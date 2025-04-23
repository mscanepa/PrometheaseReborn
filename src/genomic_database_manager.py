#!/usr/bin/env python3
"""
Genomic Database Manager

This script manages a PostgreSQL database of genomic variants from multiple sources:
- GWAS Catalog
- ClinVar
- dbSNP
"""

import os
import sys
import logging
import gzip
import requests
import pandas as pd
import hashlib
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
import json
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, UniqueConstraint, Index, inspect, event
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
import gc

# Load environment variables
load_dotenv(dotenv_path='../../relationship-calculator-api/.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler('genomic_database.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/genealogy')
CHECKPOINT_FILE = 'database_checkpoint.json'
CHUNK_SIZE = 50000
MAX_RETRIES = 3
RETRY_DELAY = 5

# Data source URLs and configurations
DATA_SOURCES = {
    'gwas': {
        'url': "https://www.ebi.ac.uk/gwas/api/search/downloads/alternative",
        'p_value_threshold': 5e-8
    },
    'clinvar': {
        'url': "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz",
        'significance': ['Pathogenic', 'Likely pathogenic', 'Risk factor']
    },
    'dbsnp': {
        'url': "https://ftp.ncbi.nih.gov/snp/latest_release/VCF/GCF_000001405.25.gz",
        'chunk_size': 100000
    }
}

class GenomicDatabaseManager:
    def __init__(self, db_url: str = DATABASE_URL):
        """Initialize the database manager."""
        logger.info("Initializing GenomicDatabaseManager...")
        self.db_url = db_url
        self.engine = None
        self.metadata = MetaData()
        self._test_connection()
        self._setup_database()
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load the last successful update checkpoint."""
        try:
            if os.path.exists(CHECKPOINT_FILE):
                with open(CHECKPOINT_FILE, 'r') as f:
                    self.checkpoint = json.load(f)
                logger.info(f"Loaded checkpoint from {self.checkpoint['timestamp']}")
            else:
                self.checkpoint = {
                    'timestamp': datetime.now().isoformat(),
                    'gwas_last_update': None,
                    'clinvar_last_update': None,
                    'dbsnp_last_update': None,
                    'gwas_checksum': None,
                    'clinvar_checksum': None,
                    'dbsnp_checksum': None
                }
                logger.info("Created new checkpoint file")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            self.checkpoint = {
                'timestamp': datetime.now().isoformat(),
                'gwas_last_update': None,
                'clinvar_last_update': None,
                'dbsnp_last_update': None,
                'gwas_checksum': None,
                'clinvar_checksum': None,
                'dbsnp_checksum': None
            }
            logger.info("Created new checkpoint after error")
    
    def _save_checkpoint(self):
        """Save the current state to checkpoint file."""
        try:
            self.checkpoint['timestamp'] = datetime.now().isoformat()
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(self.checkpoint, f)
            logger.info("Checkpoint saved successfully")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
    
    def _calculate_checksum(self, data: pd.DataFrame) -> str:
        """Calculate checksum for data validation."""
        return hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()
    
    def _test_connection(self):
        """Test the database connection and verify the database exists."""
        logger.info("Testing database connection...")
        try:
            # Create engine without connecting to a specific database
            base_url = self.db_url.rsplit('/', 1)[0]
            logger.info(f"Attempting to connect to base URL: {base_url}")
            temp_engine = create_engine(base_url)
            
            # Get database name from URL
            db_name = self.db_url.rsplit('/', 1)[1]
            logger.info(f"Checking if database '{db_name}' exists...")
            
            # Check if database exists
            with temp_engine.connect() as conn:
                result = conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :db_name"), 
                                   {"db_name": db_name})
                if not result.scalar():
                    logger.error(f"Database '{db_name}' does not exist. Please create it first.")
                    sys.exit(1)
                
                # Test connection to the specific database
                logger.info("Database exists, testing specific database connection...")
                self.engine = create_engine(self.db_url)
                with self.engine.connect() as db_conn:
                    # Test write access
                    test_table = "test_write_access"
                    db_conn.execute(text(f"CREATE TEMP TABLE {test_table} (id INTEGER)"))
                    db_conn.execute(text(f"INSERT INTO {test_table} VALUES (1)"))
                    result = db_conn.execute(text(f"SELECT * FROM {test_table}")).scalar()
                    if result != 1:
                        raise Exception("Write test failed")
                    db_conn.execute(text(f"DROP TABLE {test_table}"))
                    logger.info(f"Successfully connected to database '{db_name}' with write access")
            
        except SQLAlchemyError as e:
            logger.error(f"Database connection error: {str(e)}")
            logger.error("Please verify:")
            logger.error("1. PostgreSQL server is running")
            logger.error("2. Database credentials are correct")
            logger.error("3. Database exists and is accessible")
            logger.error("4. User has write permissions")
            sys.exit(1)
    
    def _setup_database(self):
        """Set up the PostgreSQL database with required tables and indexes."""
        logger.info("Setting up database tables and indexes...")
        try:
            # Define tables with optimized indexes
            logger.info("Defining table schemas...")
            
            # Check if tables exist before creating them
            with self.engine.connect() as conn:
                inspector = inspect(self.engine)
                existing_tables = inspector.get_table_names()
                logger.info(f"Existing tables: {existing_tables}")
            
            # GWAS variants table
            gwas_variants = Table(
                'gwas_variants', self.metadata,
                Column('id', Integer, primary_key=True),
                Column('snp_id', String),
                Column('chromosome', String),
                Column('position', Integer),
                Column('disease_trait', String),
                Column('p_value', Float),
                Column('or_beta', Float),
                Index('idx_gwas_snp', 'snp_id'),
                Index('idx_gwas_chr_pos', 'chromosome', 'position'),
                UniqueConstraint('snp_id', 'disease_trait', name='uq_gwas_snp_trait')
            )
            
            # ClinVar variants table
            clinvar_variants = Table(
                'clinvar_variants', self.metadata,
                Column('id', Integer, primary_key=True),
                Column('variation_id', Integer, unique=True),
                Column('rsid', String),
                Column('chromosome', String),
                Column('position', Integer),
                Column('reference_allele', String),
                Column('alternate_allele', String),
                Column('clinical_significance', String),
                Column('phenotype', String),
                Index('idx_clinvar_rsid', 'rsid'),
                Index('idx_clinvar_chr_pos', 'chromosome', 'position'),
                Index('idx_clinvar_var_id', 'variation_id')
            )
            
            # dbSNP variants table
            dbsnp_variants = Table(
                'dbsnp_variants', self.metadata,
                Column('id', Integer, primary_key=True),
                Column('rs_id', String, unique=True),
                Column('chromosome', String),
                Column('position', Integer),
                Column('reference_allele', String),
                Column('alternate_allele', String),
                Column('allele_frequency', Float),
                Column('population', String),
                Index('idx_dbsnp_rs', 'rs_id'),
                Index('idx_dbsnp_chr_pos', 'chromosome', 'position')
            )
            
            # Create tables if they don't exist
            tables_to_create = {
                'gwas_variants': gwas_variants,
                'clinvar_variants': clinvar_variants,
                'dbsnp_variants': dbsnp_variants
            }
            
            for table_name, table in tables_to_create.items():
                if table_name not in existing_tables:
                    logger.info(f"Creating {table_name} table...")
                    table.create(self.engine)
                    logger.info(f"Created {table_name} table successfully")
                else:
                    logger.info(f"{table_name} table already exists")
                    
                    # Verify table structure
                    columns = inspector.get_columns(table_name)
                    logger.info(f"{table_name} columns: {[col['name'] for col in columns]}")
            
            # Create optimized indexes
            self._create_optimized_indexes()
            
            # Create maintenance function
            self._create_maintenance_function()
            
            logger.info("Database setup completed successfully")
            
        except SQLAlchemyError as e:
            logger.error(f"Error setting up database: {str(e)}")
            raise
    
    def _create_optimized_indexes(self):
        """Create optimized indexes for the main tables."""
        try:
            with self.engine.connect() as conn:
                # Índices para gwas_variants
                conn.execute(text("""
                    -- Optimiza búsquedas por cromosoma y posición
                    CREATE INDEX IF NOT EXISTS idx_gwas_chr_pos ON gwas_variants (chromosome, position);
                    
                    -- Optimiza búsquedas rápidas por SNP ID
                    CREATE INDEX IF NOT EXISTS idx_gwas_snp_id ON gwas_variants (snp_id);
                """))
                
                # Índices para dbsnp_variants
                conn.execute(text("""
                    -- Búsqueda eficiente por cromosoma y posición
                    CREATE INDEX IF NOT EXISTS idx_dbsnp_chr_pos ON dbsnp_variants (chromosome, position);
                    
                    -- Búsqueda rápida por rsID
                    CREATE INDEX IF NOT EXISTS idx_dbsnp_rs_id ON dbsnp_variants (rs_id);
                    
                    -- Mejora consultas frecuentes para población específica
                    CREATE INDEX IF NOT EXISTS idx_dbsnp_population ON dbsnp_variants (population);
                """))
                
                # Índices para clinvar_variants
                conn.execute(text("""
                    -- Consultas rápidas por cromosoma y posición
                    CREATE INDEX IF NOT EXISTS idx_clinvar_chr_pos ON clinvar_variants (chromosome, position);
                    
                    -- Búsqueda eficiente por Variation ID
                    CREATE INDEX IF NOT EXISTS idx_clinvar_variation_id ON clinvar_variants (variation_id);
                    
                    -- Consultas frecuentes por relevancia clínica
                    CREATE INDEX IF NOT EXISTS idx_clinvar_clinical_significance ON clinvar_variants (clinical_significance);
                """))
                
                conn.commit()
                logger.info("Índices creados exitosamente")
                
        except Exception as e:
            logger.error(f"Error al crear índices: {str(e)}")
            raise
    
    def _create_maintenance_function(self):
        """Create a function for database maintenance."""
        try:
            with self.engine.connect() as conn:
                # Check if function exists
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT 1 
                        FROM pg_proc 
                        WHERE proname = 'maintain_database'
                    );
                """))
                function_exists = result.scalar()
                
                if not function_exists:
                    logger.info("Creating database maintenance function...")
                    conn.execute(text("""
                        CREATE OR REPLACE FUNCTION maintain_database()
                        RETURNS void AS $$
                        BEGIN
                            -- Vacuum analyze tables
                            VACUUM ANALYZE gwas_variants;
                            VACUUM ANALYZE clinvar_variants;
                            
                            -- Update statistics
                            ANALYZE gwas_variants;
                            ANALYZE clinvar_variants;
                        END;
                        $$ LANGUAGE plpgsql;
                    """))
                    logger.info("Database maintenance function created successfully")
                else:
                    logger.info("Database maintenance function already exists")
        except Exception as e:
            logger.error(f"Error creating maintenance function: {str(e)}")
            logger.warning("Database maintenance function could not be created. Maintenance tasks will be skipped.")
    
    def _run_maintenance(self):
        """Run database maintenance tasks."""
        try:
            logger.info("Running database maintenance...")
            with self.engine.connect() as conn:
                # Check if function exists before trying to execute it
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT 1 
                        FROM pg_proc 
                        WHERE proname = 'maintain_database'
                    );
                """))
                function_exists = result.scalar()
                
                if function_exists:
                    # Run maintenance function
                    logger.info("Executing maintenance function...")
                    conn.execute(text("SELECT maintain_database()"))
                    logger.info("Maintenance function executed successfully")
                else:
                    logger.warning("Maintenance function not found. Running basic maintenance tasks...")
                    # Run basic maintenance tasks directly
                    # Note: VACUUM cannot run inside a transaction block
                    conn.execute(text("COMMIT"))  # End current transaction
                    conn.execute(text("VACUUM ANALYZE gwas_variants"))
                    conn.execute(text("VACUUM ANALYZE clinvar_variants"))
                    conn.execute(text("VACUUM ANALYZE dbsnp_variants"))
                    logger.info("Basic maintenance tasks completed")
                
                # Get table sizes
                sizes = conn.execute(text("""
                    SELECT 
                        table_name,
                        pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) as total_size,
                        pg_size_pretty(pg_relation_size(quote_ident(table_name))) as table_size,
                        pg_size_pretty(pg_total_relation_size(quote_ident(table_name)) - pg_relation_size(quote_ident(table_name))) as index_size
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name IN ('gwas_variants', 'clinvar_variants', 'dbsnp_variants');
                """))
                
                for size in sizes:
                    logger.info(f"Table {size[0]}: Total={size[1]}, Table={size[2]}, Indexes={size[3]}")
            
            logger.info("Database maintenance completed")
        except Exception as e:
            logger.error(f"Error running maintenance: {str(e)}")
            logger.warning("Some maintenance tasks may not have completed successfully")
    
    def download_gwas_data(self) -> Optional[pd.DataFrame]:
        """Download and process GWAS Catalog data."""
        logger.info("Downloading GWAS data...")
        try:
            # Create a temporary file to store the downloaded data
            temp_file = io.StringIO()
            
            # Download the data in chunks with progress tracking
            response = requests.get(DATA_SOURCES['gwas']['url'], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 8192
            downloaded = 0
            
            logger.info(f"Starting download of GWAS data (total size: {total_size/1024/1024:.2f} MB)")
            
            for chunk in response.iter_content(chunk_size=chunk_size, decode_unicode=True):
                if chunk:
                    temp_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        logger.info(f"Download progress: {progress:.1f}%")
            
            # Reset the file pointer to the beginning
            temp_file.seek(0)
            
            # Read the data into a DataFrame in chunks to manage memory
            chunks = []
            chunk_size = 100000  # Process 100k rows at a time
            
            for chunk in pd.read_csv(
                temp_file,
                sep='\t',
                usecols=['SNPS', 'CHR_ID', 'CHR_POS', 'DISEASE/TRAIT', 'P-VALUE', 'OR or BETA'],
                dtype='str',
                chunksize=chunk_size
            ):
                # Process each chunk
                chunk['P-VALUE'] = pd.to_numeric(chunk['P-VALUE'], errors='coerce')
                chunk = chunk[chunk['P-VALUE'] < DATA_SOURCES['gwas']['p_value_threshold']]
                
                if not chunk.empty:
                    chunks.append(chunk)
                
                logger.info(f"Processed chunk: {len(chunk)} rows")
            
            # Close the temporary file
            temp_file.close()
            
            if not chunks:
                logger.warning("No significant GWAS variants found after filtering")
                return None
            
            # Concatenate all chunks
            df = pd.concat(chunks, ignore_index=True)
            
            initial_count = len(df)
            logger.info(f"Initial GWAS variants count: {initial_count}")
            
            # Process OR/BETA values
            logger.info("Processing OR/BETA values...")
            def process_or_beta(x):
                if pd.isna(x):
                    return None
                try:
                    if ' x ' in str(x):
                        return float(str(x).split(' x ')[0])
                    return float(str(x).replace(',', ''))
                except (ValueError, TypeError):
                    return None
            
            df['OR or BETA'] = df['OR or BETA'].apply(process_or_beta)
            
            # Remove rows with invalid OR/BETA values
            invalid_or_beta = df['OR or BETA'].isna().sum()
            if invalid_or_beta > 0:
                logger.info(f"Removing {invalid_or_beta} rows with invalid OR/BETA values")
                df = df.dropna(subset=['OR or BETA'])
            
            # Rename columns
            df = df.rename(columns={
                'SNPS': 'snp_id',
                'CHR_ID': 'chromosome',
                'CHR_POS': 'position',
                'DISEASE/TRAIT': 'disease_trait',
                'P-VALUE': 'p_value',
                'OR or BETA': 'or_beta'
            })
            
            # Convert position to integer
            df['position'] = pd.to_numeric(df['position'], errors='coerce').astype('Int64')
            
            # Clean up memory
            del chunks
            gc.collect()
            
            logger.info(f"GWAS data processing completed: {len(df)} significant variants")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading GWAS data: {str(e)}")
            return None
    
    def _verify_insertion(self, table_name: str, id_column: str, inserted_ids: set) -> bool:
        """Verify that all records were actually saved in the database."""
        try:
            with self.engine.connect() as conn:
                # Get all IDs from the database
                db_ids = pd.read_sql(
                    f"SELECT {id_column} FROM {table_name}",
                    conn
                )[id_column].tolist()
                db_id_set = set(db_ids)
                
                # Check for missing records
                missing_ids = inserted_ids - db_id_set
                if missing_ids:
                    logger.error(f"Missing {len(missing_ids)} records in {table_name}")
                    logger.error(f"Sample of missing IDs: {list(missing_ids)[:10]}")
                    return False
                
                # Check for extra records
                extra_ids = db_id_set - inserted_ids
                if extra_ids:
                    logger.error(f"Found {len(extra_ids)} unexpected records in {table_name}")
                    logger.error(f"Sample of extra IDs: {list(extra_ids)[:10]}")
                    return False
                
                logger.info(f"Successfully verified {len(inserted_ids)} records in {table_name}")
                return True
        except Exception as e:
            logger.error(f"Error verifying insertion in {table_name}: {str(e)}")
            return False

    def _insert_gwas_data(self, df: pd.DataFrame) -> bool:
        """Insert GWAS data into the database."""
        if df is None or df.empty:
            logger.warning("No GWAS data to insert")
            return False

        logger.info(f"Starting insertion of {len(df)} GWAS variants")
        
        try:
            # Checkpoint 1: Data cleaning
            logger.info("=== CHECKPOINT 1: Data Cleaning ===")
            initial_count = len(df)
            df = df.replace({pd.NA: None, np.nan: None})
            df = df.dropna(subset=['snp_id', 'chromosome', 'position', 'disease_trait', 'p_value', 'or_beta'])
            
            if len(df) < initial_count:
                logger.warning(f"Removed {initial_count - len(df)} records with missing values")
                logger.info(f"Sample of removed records: {df[df.isna().any(axis=1)].head(5).to_dict('records')}")
            
            if df.empty:
                logger.info("No valid records after cleaning")
                return True
            
            logger.info(f"Records after cleaning: {len(df)}")
            
            # Checkpoint 2: Remove internal duplicates
            logger.info("=== CHECKPOINT 2: Internal Duplicate Removal ===")
            initial_count = len(df)
            df = df.drop_duplicates(subset=['snp_id', 'disease_trait'], keep='first')
            
            if len(df) < initial_count:
                logger.warning(f"Removed {initial_count - len(df)} duplicate records within the dataset")
                logger.info(f"Sample of duplicate records: {df[df.duplicated(subset=['snp_id', 'disease_trait'], keep=False)].head(5).to_dict('records')}")
            
            logger.info(f"Records after internal duplicate removal: {len(df)}")
            
            if df.empty:
                logger.info("No valid records after duplicate removal")
                return True
            
            # Checkpoint 3: Verify against database
            logger.info("=== CHECKPOINT 3: Database Verification ===")
            with self.engine.begin() as conn:
                # Get existing records
                existing_records = pd.read_sql("""
                    SELECT snp_id, disease_trait 
                    FROM gwas_variants
                """, conn)
                
                if not existing_records.empty:
                    logger.info(f"Found {len(existing_records)} existing records in database")
                    # Create a set of existing pairs for fast lookup
                    existing_pairs = set(zip(existing_records['snp_id'], existing_records['disease_trait']))
                    
                    # Filter out existing records
                    df['exists'] = df.apply(lambda row: (str(row['snp_id']), str(row['disease_trait'])) in existing_pairs, axis=1)
                    new_records = df[~df['exists']].drop('exists', axis=1)
                    
                    if len(new_records) == 0:
                        logger.info("No new records to insert - all records already exist in database")
                        return True
                        
                    logger.info(f"Found {len(new_records)} new records to insert")
                    df = new_records
                else:
                    logger.info("No existing records found in database - all records are new")
            
            # Create a temporary table and insert data in a single transaction
            with self.engine.begin() as conn:
                # Checkpoint 4: Create temporary table
                logger.info("=== CHECKPOINT 4: Temporary Table Creation ===")
                conn.execute(text("""
                    CREATE TEMP TABLE temp_gwas_variants (
                        snp_id TEXT,
                        chromosome TEXT,
                        position BIGINT,
                        disease_trait TEXT,
                        p_value FLOAT,
                        or_beta FLOAT,
                        UNIQUE(snp_id, disease_trait)
                    ) ON COMMIT DROP;
                """))
                
                # Checkpoint 5: Batch insertion
                logger.info("=== CHECKPOINT 5: Batch Insertion ===")
                batch_size = 10000
                total_rows = len(df)
                inserted_rows = 0
                
                for i in range(0, total_rows, batch_size):
                    batch = df.iloc[i:i + batch_size]
                    
                    # Convert DataFrame to list of dictionaries for bulk insert
                    records = []
                    for _, row in batch.iterrows():
                        try:
                            record = {
                                'snp_id': str(row['snp_id']),
                                'chromosome': str(row['chromosome']),
                                'position': int(row['position']),
                                'disease_trait': str(row['disease_trait']),
                                'p_value': float(row['p_value']),
                                'or_beta': float(row['or_beta']) if pd.notna(row['or_beta']) else None
                            }
                            records.append(record)
                        except Exception as e:
                            logger.error(f"Error processing record: {str(e)}")
                            logger.error(f"Record data: {row.to_dict()}")
                            continue
                    
                    # Bulk insert into temporary table
                    conn.execute(text("""
                        INSERT INTO temp_gwas_variants 
                        (snp_id, chromosome, position, disease_trait, p_value, or_beta)
                        VALUES (:snp_id, :chromosome, :position, :disease_trait, :p_value, :or_beta)
                    """), records)
                    
                    inserted_rows += len(records)
                    logger.info(f"Inserted batch: {inserted_rows}/{total_rows} rows")
                
                # Checkpoint 6: Final insertion
                logger.info("=== CHECKPOINT 6: Final Insertion ===")
                conn.execute(text("""
                    INSERT INTO gwas_variants 
                    (snp_id, chromosome, position, disease_trait, p_value, or_beta)
                    SELECT 
                        snp_id, 
                        chromosome, 
                        position, 
                        disease_trait, 
                        p_value, 
                        or_beta
                    FROM temp_gwas_variants;
                """))
                
                # Checkpoint 7: Verification
                logger.info("=== CHECKPOINT 7: Final Verification ===")
                result = conn.execute(text("SELECT COUNT(*) FROM gwas_variants"))
                final_count = result.scalar()
                logger.info(f"Final count in gwas_variants table: {final_count}")
                
                if final_count < inserted_rows:
                    logger.error(f"Insertion verification failed: expected {inserted_rows} records, got {final_count}")
                    return False
            
            logger.info("GWAS data insertion completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting GWAS data: {str(e)}")
            logger.error("Stopping script due to error")
            raise
    
    def download_clinvar_data(self) -> Optional[pd.DataFrame]:
        """Download and process ClinVar data."""
        logger.info("Downloading ClinVar data...")
        try:
            response = requests.get(DATA_SOURCES['clinvar']['url'], stream=True)
            response.raise_for_status()
            
            # Process data in chunks
            chunks = []
            total_variants = 0
            significant_variants = 0
            
            with gzip.open(io.BytesIO(response.content), 'rt') as f:
                # Read header
                header = f.readline().strip().split('\t')
                
                # Process data in smaller chunks
                chunk_size = 10000  # Reduced chunk size
                chunk = []
                
                for line in f:
                    chunk.append(line.strip().split('\t'))
                    
                    if len(chunk) >= chunk_size:
                        # Process chunk
                        df_chunk = pd.DataFrame(chunk, columns=header)
                        chunk = []  # Clear chunk to free memory
                        
                        # Filter significant variants
                        df_chunk = df_chunk[df_chunk['ClinicalSignificance'].isin(DATA_SOURCES['clinvar']['significance'])]
                        
                        if not df_chunk.empty:
                            # Process chunk
                            df_chunk = df_chunk[[
                                'VariationID', 'Name', 'ClinicalSignificance',
                                'PhenotypeList', 'Chromosome', 'Start',
                                'ReferenceAllele', 'AlternateAllele'
                            ]]
                            df_chunk.columns = [
                                'variation_id', 'name', 'clinical_significance',
                                'phenotype', 'chromosome', 'position',
                                'reference_allele', 'alternate_allele'
                            ]
                            
                            # Clean data
                            df_chunk = df_chunk.replace('', pd.NA)
                            df_chunk = df_chunk.dropna(subset=['variation_id', 'chromosome', 'position'])
                            
                            chunks.append(df_chunk)
                            significant_variants += len(df_chunk)
                        
                        total_variants += chunk_size
                        logger.info(f"Processed {total_variants:,} variants, found {significant_variants:,} significant variants")
                        
                        # Force garbage collection after each chunk
                        import gc
                        gc.collect()
            
            # Process remaining chunk if any
            if chunk:
                df_chunk = pd.DataFrame(chunk, columns=header)
                df_chunk = df_chunk[df_chunk['ClinicalSignificance'].isin(DATA_SOURCES['clinvar']['significance'])]
                
                if not df_chunk.empty:
                    df_chunk = df_chunk[[
                        'VariationID', 'Name', 'ClinicalSignificance',
                        'PhenotypeList', 'Chromosome', 'Start',
                        'ReferenceAllele', 'AlternateAllele'
                    ]]
                    df_chunk.columns = [
                        'variation_id', 'name', 'clinical_significance',
                        'phenotype', 'chromosome', 'position',
                        'reference_allele', 'alternate_allele'
                    ]
                    
                    df_chunk = df_chunk.replace('', pd.NA)
                    df_chunk = df_chunk.dropna(subset=['variation_id', 'chromosome', 'position'])
                    
                    chunks.append(df_chunk)
                    significant_variants += len(df_chunk)
                
                total_variants += len(chunk)
                logger.info(f"Processed {total_variants:,} variants, found {significant_variants:,} significant variants")
            
            if chunks:
                # Concatenate chunks
                df = pd.concat(chunks, ignore_index=True)
                logger.info(f"Successfully downloaded {len(df):,} significant ClinVar variants")
                return df
            else:
                logger.warning("No significant ClinVar variants found")
                return None
            
        except Exception as e:
            logger.error(f"Error downloading ClinVar data: {str(e)}")
            return None
    
    def download_dbsnp_data(self) -> Optional[pd.DataFrame]:
        """Download and process dbSNP data in batches with validation."""
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Starting dbSNP data download (attempt {attempt + 1}/{MAX_RETRIES})...")
                response = requests.get(DATA_SOURCES['dbsnp']['url'], stream=True)
                response.raise_for_status()
                
                logger.info("Processing dbSNP data in batches...")
                total_variants = 0
                total_inserted = 0
                batch_size = 100000  # Process 100k variants at a time
                
                with gzip.GzipFile(fileobj=response.raw) as f:
                    # First, read the header to get column names
                    header = None
                    for line in f:
                        if line.startswith(b'#CHROM'):
                            header = line.decode('utf-8').strip().split('\t')
                            logger.info(f"Found VCF header with columns: {header}")
                            break
                    
                    if not header:
                        raise ValueError("Could not find VCF header in the file")
                    
                    # Create a mapping of standard VCF columns
                    vcf_columns = {
                        '#CHROM': 'chromosome',
                        'POS': 'position',
                        'ID': 'rs_id',
                        'REF': 'reference_allele',
                        'ALT': 'alternate_allele',
                        'INFO': 'info'
                    }
                    
                    # Find which columns are actually present
                    available_columns = [col for col in vcf_columns.keys() if col in header]
                    logger.info(f"Using columns: {available_columns}")
                    
                    # Read the data in chunks
                    for chunk in pd.read_csv(
                        f,
                        sep='\t',
                        chunksize=batch_size,
                        names=header,
                        usecols=available_columns,
                        dtype={col: 'str' for col in available_columns}
                    ):
                        # Step 1: Process the batch
                        logger.info(f"=== Processing batch of {len(chunk)} variants ===")
                        
                        # Step 2: Clean invalid data
                        initial_count = len(chunk)
                        chunk = chunk.replace({pd.NA: None, np.nan: None})
                        chunk = chunk.dropna(subset=['ID', '#CHROM', 'POS', 'REF', 'ALT'])
                        
                        if len(chunk) < initial_count:
                            logger.info(f"Removed {initial_count - len(chunk)} invalid records from batch")
                        
                        if chunk.empty:
                            logger.info("No valid records in this batch")
                            continue
                        
                        # Extract AF and POP from INFO column if present
                        if 'INFO' in chunk.columns:
                            chunk['AF'] = chunk['INFO'].str.extract(r'AF=([^;]+)')
                            chunk['POP'] = chunk['INFO'].str.extract(r'POP=([^;]+)')
                            chunk = chunk.drop('INFO', axis=1)
                            chunk['AF'] = pd.to_numeric(chunk['AF'], errors='coerce')
                        
                        # Rename columns according to our mapping
                        chunk = chunk.rename(columns={k: v for k, v in vcf_columns.items() if k in chunk.columns})
                        
                        # Step 3: Validate against database
                        with self.engine.begin() as conn:
                            # Create temporary table for validation
                            conn.execute(text("""
                                CREATE TEMP TABLE temp_check_rs_ids (
                                    rs_id TEXT
                                ) ON COMMIT DROP;
                            """))
                            
                            # Insert RS IDs to check
                            check_records = [{'rs_id': str(rs_id)} for rs_id in chunk['rs_id']]
                            conn.execute(text("""
                                INSERT INTO temp_check_rs_ids (rs_id)
                                VALUES (:rs_id)
                            """), check_records)
                            
                            # Find existing records
                            existing_records = pd.read_sql("""
                                SELECT t.rs_id
                                FROM temp_check_rs_ids t
                                JOIN dbsnp_variants m ON t.rs_id = m.rs_id
                            """, conn)
                            
                            if not existing_records.empty:
                                logger.info(f"Found {len(existing_records)} existing records in database")
                                # Filter out existing records
                                chunk = chunk[~chunk['rs_id'].isin(existing_records['rs_id'])]
                            
                            if chunk.empty:
                                logger.info("No new records to insert in this batch")
                                continue
                            
                            # Step 4: Insert new records
                            logger.info(f"Inserting {len(chunk)} new records")
                            records = []
                            for _, row in chunk.iterrows():
                                try:
                                    record = {
                                        'rs_id': str(row['rs_id']),
                                        'chromosome': str(row['chromosome']),
                                        'position': int(row['position']),
                                        'reference_allele': str(row['reference_allele']),
                                        'alternate_allele': str(row['alternate_allele']),
                                        'allele_frequency': float(row['AF']) if 'AF' in row and pd.notna(row['AF']) else None,
                                        'population': str(row['POP']) if 'POP' in row and pd.notna(row['POP']) else None
                                    }
                                    records.append(record)
                                except Exception as e:
                                    logger.error(f"Error processing record: {str(e)}")
                                    logger.error(f"Record data: {row.to_dict()}")
                                    continue
                            
                            # Bulk insert new records
                            conn.execute(text("""
                                INSERT INTO dbsnp_variants 
                                (rs_id, chromosome, position, reference_allele, alternate_allele, allele_frequency, population)
                                VALUES (:rs_id, :chromosome, :position, :reference_allele, :alternate_allele, :allele_frequency, :population)
                            """), records)
                            
                            total_inserted += len(records)
                        
                        total_variants += len(chunk)
                        if total_variants % 1000000 == 0:
                            logger.info(f"Processed {total_variants:,} variants so far, inserted {total_inserted:,} new records")
                        
                        # Clear memory
                        del chunk
                        gc.collect()
                
                # Step 6: Final report
                logger.info("=== dbSNP Data Processing Summary ===")
                logger.info(f"Total variants processed: {total_variants:,}")
                logger.info(f"Total new records inserted: {total_inserted:,}")
                logger.info(f"Percentage of new records: {(total_inserted/total_variants*100):.2f}%")
                
                return None  # Return None since we've already inserted the data
                
            except Exception as e:
                logger.error(f"Error downloading dbSNP data (attempt {attempt + 1}): {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    return None
    
    def _insert_data_with_verification(self, table_name: str, data: pd.DataFrame, id_column: str) -> bool:
        """Insert data with verification and error handling."""
        if data is None or data.empty:
            logger.error("ERROR: No data provided for insertion")
            raise ValueError("No data provided for insertion")

        logger.info(f"Starting insertion process for {table_name}")
        logger.info(f"Initial record count: {len(data)}")
        logger.info(f"Using ID column for validation: {id_column}")
        
        try:
            # Checkpoint 1: Data cleaning
            logger.info("=== CHECKPOINT 1: Data Cleaning ===")
            initial_count = len(data)
            data = data.replace({pd.NA: None, np.nan: None})
            data = data.dropna(subset=[id_column])
            
            if len(data) < initial_count:
                logger.warning(f"Removed {initial_count - len(data)} records with missing {id_column}")
            
            if data.empty:
                logger.error("ERROR: No valid records after cleaning")
                raise ValueError("No valid records after cleaning")
            
            logger.info(f"Records after cleaning: {len(data)}")
            
            # Checkpoint 2: Remove duplicates
            logger.info("=== CHECKPOINT 2: Duplicate Removal ===")
            initial_count = len(data)
            data = data.drop_duplicates(subset=[id_column], keep='first')
            
            if len(data) < initial_count:
                logger.warning(f"Removed {initial_count - len(data)} duplicate records from input data")
                logger.info(f"Sample of duplicate IDs: {data[id_column].head(5).tolist()}")
            
            logger.info(f"Records after duplicate removal: {len(data)}")
            
            # Create a temporary table and insert data in a single transaction
            with self.engine.begin() as conn:
                # Checkpoint 3: Verify existing records
                logger.info("=== CHECKPOINT 3: Existing Records Verification ===")
                
                # Create a temporary table with the new records to check
                # Use appropriate data type based on the table
                id_type = "INTEGER" if table_name == "clinvar_variants" else "TEXT"
                conn.execute(text(f"""
                    CREATE TEMP TABLE temp_check_records (
                        {id_column} {id_type}
                    ) ON COMMIT DROP;
                """))
                
                # Insert the records to check, converting types if necessary
                check_records = []
                for _, row in data.iterrows():
                    try:
                        if table_name == "clinvar_variants":
                            # Convert to integer for ClinVar
                            check_records.append({id_column: int(row[id_column])})
                        else:
                            # Keep as text for other tables
                            check_records.append({id_column: str(row[id_column])})
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error converting {id_column} value: {row[id_column]}")
                        logger.error(f"Error details: {str(e)}")
                        continue
                
                if not check_records:
                    logger.error("No valid records after type conversion")
                    raise ValueError("No valid records after type conversion")
                
                conn.execute(text(f"""
                    INSERT INTO temp_check_records ({id_column})
                    VALUES (:{id_column})
                """), check_records)
                
                # Find which records already exist using a single SQL query
                existing_records = pd.read_sql(f"""
                    SELECT t.{id_column}
                    FROM temp_check_records t
                    JOIN {table_name} m ON t.{id_column} = m.{id_column}
                """, conn)
                
                if not existing_records.empty:
                    logger.info(f"Found {len(existing_records)} existing records in database")
                    logger.info(f"Sample of existing IDs: {existing_records[id_column].head(5).tolist()}")
                    
                    # Create a set of existing IDs for fast lookup
                    existing_ids = set(existing_records[id_column])
                    
                    # Filter out existing records
                    data['exists'] = data[id_column].isin(existing_ids)
                    new_records = data[~data['exists']].drop('exists', axis=1)
                    
                    if len(new_records) == 0:
                        logger.info("No new records to insert - all records already exist")
                        return True
                        
                    logger.info(f"Found {len(new_records)} new records to insert")
                    logger.info(f"Sample of new IDs: {new_records[id_column].head(5).tolist()}")
                    data = new_records
                else:
                    logger.info("No existing records found in database - all records are new")
                
                # Checkpoint 4: Create temporary table
                logger.info("=== CHECKPOINT 4: Temporary Table Creation ===")
                conn.execute(text(f"""
                    CREATE TEMP TABLE temp_{table_name} (
                        LIKE {table_name} INCLUDING DEFAULTS
                    ) ON COMMIT DROP;
                """))
                
                # Checkpoint 5: Batch insertion
                logger.info("=== CHECKPOINT 5: Batch Insertion ===")
                batch_size = 10000
                total_rows = len(data)
                inserted_rows = 0
                
                for i in range(0, total_rows, batch_size):
                    batch = data.iloc[i:i + batch_size]
                    batch_start = i + 1
                    batch_end = min(i + batch_size, total_rows)
                    
                    logger.info(f"Processing batch {batch_start}-{batch_end} of {total_rows}")
                    
                    # Convert DataFrame to list of dictionaries
                    records = []
                    for _, row in batch.iterrows():
                        try:
                            record = {}
                            for col in data.columns:
                                if col == id_column and table_name == "clinvar_variants":
                                    record[col] = int(row[col])
                                else:
                                    record[col] = str(row[col]) if pd.notna(row[col]) else None
                            records.append(record)
                        except Exception as e:
                            logger.error(f"ERROR: Failed to process record with {id_column}: {row[id_column]}")
                            logger.error(f"Record data: {row.to_dict()}")
                            logger.error(f"Error details: {str(e)}")
                            continue
                    
                    # Bulk insert into temporary table
                    columns = ', '.join(data.columns)
                    values = ', '.join([f':{col}' for col in data.columns])
                    conn.execute(text(f"""
                        INSERT INTO temp_{table_name} ({columns})
                        VALUES ({values})
                    """), records)
                    
                    inserted_rows += len(records)
                    logger.info(f"Successfully inserted batch: {inserted_rows}/{total_rows} rows")
                
                # Checkpoint 6: Final insertion
                logger.info("=== CHECKPOINT 6: Final Insertion ===")
                try:
                    conn.execute(text(f"""
                        INSERT INTO {table_name}
                        SELECT * FROM temp_{table_name};
                    """))
                except SQLAlchemyError as e:
                    if "UniqueViolation" in str(e) and table_name == "clinvar_variants":
                        # For ClinVar, duplicate key violations are expected and can be ignored
                        logger.info("Some records already exist in the database - this is expected for ClinVar data")
                    else:
                        # For other tables or other types of errors, raise the exception
                        raise
                
                # Checkpoint 7: Verification
                logger.info("=== CHECKPOINT 7: Final Verification ===")
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                final_count = result.scalar()
                logger.info(f"Final count in {table_name} table: {final_count}")
                
                if final_count < inserted_rows:
                    logger.error(f"ERROR: Insertion verification failed")
                    logger.error(f"Expected {inserted_rows} records, got {final_count}")
                    raise ValueError("Insertion verification failed")
                
                logger.info("Insertion verification successful")
            
            logger.info(f"=== {table_name} data insertion completed successfully ===")
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Failed to insert {table_name} data")
            logger.error(f"Error details: {str(e)}")
            logger.error("Stopping script due to error")
            raise

    def update_database(self):
        """Update the database with latest data from all sources."""
        logger.info("Starting database update process...")
        success = True
        failed_sources = []
        
        try:
            # Download and process data from all sources
            data_sources = {
                'gwas': {
                    'download_func': self.download_gwas_data,
                    'table_name': 'gwas_variants',
                    'id_column': 'snp_id',
                    'insert_func': self._insert_gwas_data
                },
                'clinvar': {
                    'download_func': self.download_clinvar_data,
                    'table_name': 'clinvar_variants',
                    'id_column': 'variation_id',
                    'insert_func': self._insert_data_with_verification
                },
                'dbsnp': {
                    'download_func': self.download_dbsnp_data,
                    'table_name': 'dbsnp_variants',
                    'id_column': 'rs_id',
                    'insert_func': self._insert_data_with_verification
                }
            }
            
            for source_name, source_config in data_sources.items():
                try:
                    logger.info(f"=== Processing {source_name} data ===")
                    data = source_config['download_func']()
                    
                    if data is not None:
                        # Verify table exists and is accessible
                        inspector = inspect(self.engine)
                        if source_config['table_name'] not in inspector.get_table_names():
                            logger.error(f"Table {source_config['table_name']} does not exist!")
                            failed_sources.append(source_name)
                            continue  # Continue with next source
                        
                        # Get existing IDs
                        logger.info(f"Checking for existing {source_name} records...")
                        try:
                            with self.engine.connect() as conn:
                                existing_ids = pd.read_sql(
                                    f"SELECT {source_config['id_column']} FROM {source_config['table_name']}",
                                    conn
                                )
                                existing_id_set = set(existing_ids[source_config['id_column']].tolist())
                                logger.info(f"Found {len(existing_id_set)} existing {source_name} records")
                        except Exception as e:
                            logger.error(f"Error checking existing records: {str(e)}")
                            failed_sources.append(source_name)
                            continue  # Continue with next source
                        
                        # Filter out existing records
                        new_data = data[~data[source_config['id_column']].isin(existing_id_set)]
                        logger.info(f"Found {len(new_data)} new {source_name} records to insert")
                        
                        if len(new_data) > 0:
                            try:
                                if source_name == 'gwas':
                                    if not source_config['insert_func'](new_data):
                                        logger.error(f"Failed to insert {source_name} data")
                                        failed_sources.append(source_name)
                                        continue
                                elif source_name == 'dbsnp':
                                    if not source_config['insert_func'](new_data):
                                        logger.error(f"Failed to insert {source_name} data")
                                        failed_sources.append(source_name)
                                        continue
                                else:  # clinvar
                                    if not source_config['insert_func'](
                                        source_config['table_name'],
                                        new_data,
                                        source_config['id_column']
                                    ):
                                        logger.error(f"Failed to insert {source_name} data")
                                        failed_sources.append(source_name)
                                        continue
                                logger.info(f"Successfully inserted {source_name} data")
                            except Exception as e:
                                logger.error(f"Error during {source_name} insertion: {str(e)}")
                                failed_sources.append(source_name)
                                continue  # Continue with next source
                        else:
                            logger.info(f"No new {source_name} records to insert")
                    else:
                        logger.error(f"Failed to download {source_name} data")
                        failed_sources.append(source_name)
                        continue  # Continue with next source
                        
                except Exception as e:
                    logger.error(f"Error processing {source_name} data: {str(e)}")
                    failed_sources.append(source_name)
                    continue  # Continue with next source
            
            # Run maintenance after update
            if not failed_sources:
                self._run_maintenance()
                logger.info("Database update completed successfully for all sources")
            else:
                logger.warning(f"Database update completed with errors in sources: {', '.join(failed_sources)}")
                logger.info(f"Successfully processed sources: {', '.join(set(data_sources.keys()) - set(failed_sources))}")
            
            return len(failed_sources) == 0
            
        except Exception as e:
            logger.error(f"Critical error updating database: {str(e)}")
            return False
    
    def query_variants(self, rsids: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """Query variants by RSIDs."""
        logger.info(f"Querying variants for {len(rsids)} RSIDs...")
        try:
            gwas_results = []
            clinvar_results = []
            
            with self.engine.connect() as conn:
                # Query GWAS variants
                logger.info("Querying GWAS variants...")
                gwas_query = text("""
                    SELECT * FROM gwas_variants 
                    WHERE snp_id = ANY(:rsids)
                """)
                gwas_result = conn.execute(gwas_query, {"rsids": rsids})
                gwas_results = [dict(row) for row in gwas_result]
                logger.info(f"Found {len(gwas_results)} matching GWAS variants")
                
                # Query ClinVar variants
                logger.info("Querying ClinVar variants...")
                clinvar_query = text("""
                    SELECT * FROM clinvar_variants 
                    WHERE name = ANY(:rsids)
                """)
                clinvar_result = conn.execute(clinvar_query, {"rsids": rsids})
                clinvar_results = [dict(row) for row in clinvar_result]
                logger.info(f"Found {len(clinvar_results)} matching ClinVar variants")
            
            return gwas_results, clinvar_results
            
        except Exception as e:
            logger.error(f"Error querying variants: {str(e)}")
            return [], []
    
    def close(self):
        """Close the database connection."""
        if self.engine:
            logger.info("Closing database connection...")
            self.engine.dispose()
            logger.info("Database connection closed")

    def query_gwas_data(self, rs_ids: List[str], chromosome: str = None, position: int = None) -> pd.DataFrame:
        """Query GWAS data using rsid first, then chromosome and position if needed."""
        try:
            # First try to query by rsid
            query = text("""
                SELECT 
                    g.snp_id as rsid,
                    g.chromosome,
                    g.position,
                    g.disease_trait,
                    g.p_value,
                    g.or_beta
                FROM gwas_variants g
                WHERE g.snp_id IN :rs_ids
            """)
            
            # Execute query and convert to DataFrame
            df = pd.read_sql(query, self.engine, params={'rs_ids': tuple(rs_ids)})
            
            # If no results found by rsid and chromosome/position provided, try that
            if df.empty and chromosome and position:
                query = text("""
                    SELECT 
                        g.snp_id as rsid,
                        g.chromosome,
                        g.position,
                        g.disease_trait,
                        g.p_value,
                        g.or_beta
                    FROM gwas_variants g
                    WHERE g.chromosome = :chromosome 
                    AND g.position = :position
                """)
                
                df = pd.read_sql(query, self.engine, params={
                    'chromosome': chromosome,
                    'position': position
                })
            
            return df
            
        except Exception as e:
            logger.error(f"Error querying GWAS data: {str(e)}")
            return pd.DataFrame()

    def query_clinvar_data(self, rs_ids: List[str], chromosome: str = None, position: int = None) -> pd.DataFrame:
        """Query ClinVar data using rsid first, then chromosome and position if needed."""
        try:
            logger.info(f"Starting ClinVar query for {len(rs_ids)} rs_ids...")
            
            # Process in larger batches with parallel processing
            batch_size = 10000  # Increased batch size
            all_results = []
            
            # Create a temporary table for batch processing
            with self.engine.connect() as conn:
                conn.execute(text("""
                    CREATE TEMP TABLE temp_rs_ids (
                        rs_id TEXT,
                        batch_id INTEGER
                    ) ON COMMIT DROP;
                """))
                
                # Insert rs_ids in batches with batch_id
                for i in range(0, len(rs_ids), batch_size):
                    batch = rs_ids[i:i + batch_size]
                    batch_id = i // batch_size
                    records = [{'rs_id': rs_id, 'batch_id': batch_id} for rs_id in batch]
                    conn.execute(text("""
                        INSERT INTO temp_rs_ids (rs_id, batch_id)
                        VALUES (:rs_id, :batch_id)
                    """), records)
                
                # Create index on temporary table
                conn.execute(text("""
                    CREATE INDEX idx_temp_rs_ids ON temp_rs_ids (rs_id, batch_id);
                """))
                
                # Query 1: Search by rsid
                logger.info("Executing ClinVar query by rsid...")
                df_rsid = pd.read_sql(text("""
                    SELECT 
                        c.variation_id,
                        c.rsid,
                        c.chromosome,
                        c.position,
                        c.clinical_significance,
                        c.phenotype,
                        c.reference_allele,
                        c.alternate_allele,
                        t.batch_id
                    FROM clinvar_variants c
                    JOIN temp_rs_ids t ON c.rsid = t.rs_id
                """), conn)
                
                # Query 2: Search by name
                logger.info("Executing ClinVar query by name...")
                df_name = pd.read_sql(text("""
                    SELECT 
                        c.variation_id,
                        c.name as rsid,
                        c.chromosome,
                        c.position,
                        c.clinical_significance,
                        c.phenotype,
                        c.reference_allele,
                        c.alternate_allele,
                        t.batch_id
                    FROM clinvar_variants c
                    JOIN temp_rs_ids t ON c.name = t.rs_id
                """), conn)
                
                # Combine results and remove duplicates
                logger.info("Combining ClinVar results...")
                df = pd.concat([df_rsid, df_name], ignore_index=True)
                df = df.drop_duplicates(subset=['variation_id'])
                
                if not df.empty:
                    logger.info(f"Found {len(df)} ClinVar variants")
                    
                    # If chromosome and position provided, also check those
                    if chromosome and position:
                        # Create a temporary table with the found variation_ids
                        conn.execute(text("""
                            CREATE TEMP TABLE found_variations (
                                variation_id INTEGER
                            ) ON COMMIT DROP;
                        """))
                        
                        # Insert found variation_ids
                        if not df.empty:
                            variation_ids = [{'variation_id': int(vid)} for vid in df['variation_id'].unique()]
                            conn.execute(text("""
                                INSERT INTO found_variations (variation_id)
                                VALUES (:variation_id)
                            """), variation_ids)
                        
                        # Query for position-based variants
                        logger.info("Executing position-based ClinVar query...")
                        position_df = pd.read_sql(text("""
                            SELECT 
                                c.variation_id,
                                CASE 
                                    WHEN c.rsid IS NOT NULL THEN c.rsid
                                    ELSE c.name
                                END as rsid,
                                c.chromosome,
                                c.position,
                                c.clinical_significance,
                                c.phenotype,
                                c.reference_allele,
                                c.alternate_allele
                            FROM clinvar_variants c
                            WHERE c.chromosome = :chromosome 
                            AND c.position = :position
                            AND NOT EXISTS (
                                SELECT 1 FROM found_variations f
                                WHERE f.variation_id = c.variation_id
                            )
                        """), conn, params={
                            'chromosome': chromosome,
                            'position': position
                        })
                        
                        if not position_df.empty:
                            logger.info(f"Found additional {len(position_df)} variants by position")
                            df = pd.concat([df, position_df], ignore_index=True)
                
                return df
            
        except Exception as e:
            logger.error(f"Error querying ClinVar data: {str(e)}")
            return pd.DataFrame()

    def query_dbsnp_data(self, rs_ids: List[str], chromosome: str = None, position: int = None) -> pd.DataFrame:
        """Query dbSNP data using rsid first, then chromosome and position if needed."""
        try:
            # First try to query by rsid
            query = text("""
                SELECT 
                    d.rs_id as rsid,
                    d.chromosome,
                    d.position,
                    d.reference_allele,
                    d.alternate_allele,
                    d.allele_frequency,
                    d.population
                FROM dbsnp_variants d
                WHERE d.rs_id IN :rs_ids
            """)
            
            # Execute query and convert to DataFrame
            df = pd.read_sql(query, self.engine, params={'rs_ids': tuple(rs_ids)})
            
            # If no results found by rsid and chromosome/position provided, try that
            if df.empty and chromosome and position:
                query = text("""
                    SELECT 
                        d.rs_id as rsid,
                        d.chromosome,
                        d.position,
                        d.reference_allele,
                        d.alternate_allele,
                        d.allele_frequency,
                        d.population
                    FROM dbsnp_variants d
                    WHERE d.chromosome = :chromosome 
                    AND d.position = :position
                """)
                
                df = pd.read_sql(query, self.engine, params={
                    'chromosome': chromosome,
                    'position': position
                })
            
            # Ensure rsid column exists and is properly formatted
            if not df.empty and 'rsid' in df.columns:
                df['rsid'] = df['rsid'].astype(str)
            
            return df
            
        except Exception as e:
            logger.error(f"Error querying dbSNP data: {str(e)}")
            return pd.DataFrame()

    def procesar_datos_geneticos(self, rs_ids: List[str], chromosome: str = None, position: int = None) -> pd.DataFrame:
        """Procesa datos genéticos de múltiples fuentes y los combina en un único DataFrame."""
        try:
            logger.info(f"Iniciando procesamiento de datos genéticos para {len(rs_ids)} rs_ids...")
            
            # 1. Definir estructura base del DataFrame final
            logger.info("Definiendo estructura de columnas...")
            base_columns = ['rsid', 'chromosome', 'position']
            gwas_columns = ['gwas_disease_trait', 'gwas_p_value', 'gwas_or_beta']
            clinvar_columns = [
                'clinvar_variation_id', 'clinvar_clinical_significance',
                'clinvar_phenotype', 'clinvar_reference_allele', 'clinvar_alternate_allele'
            ]
            dbsnp_columns = [
                'dbsnp_reference_allele', 'dbsnp_alternate_allele',
                'dbsnp_allele_frequency', 'dbsnp_population'
            ]
            
            all_columns = base_columns + gwas_columns + clinvar_columns + dbsnp_columns
            logger.info(f"Columnas definidas: {all_columns}")
            
            # 2. Crear DataFrame base con todos los SNPs
            logger.info("Creando DataFrame base...")
            data = {col: [] for col in all_columns}
            for rsid in rs_ids:
                data['rsid'].append(rsid)
                data['chromosome'].append(chromosome)
                data['position'].append(position)
                # Inicializar el resto de columnas como NaN
                for col in gwas_columns + clinvar_columns + dbsnp_columns:
                    data[col].append(None)
            
            df_final = pd.DataFrame(data)
            logger.info(f"DataFrame base creado con forma: {df_final.shape}")
            logger.info(f"Columnas del DataFrame base: {df_final.columns.tolist()}")
            
            # 3. Función para actualizar datos de una fuente
            def update_source_data(source_df, prefix, column_mapping):
                if source_df.empty:
                    logger.info(f"No hay datos para la fuente {prefix}")
                    return
                
                logger.info(f"Procesando datos de {prefix}...")
                logger.info(f"Forma del DataFrame de origen: {source_df.shape}")
                logger.info(f"Columnas del DataFrame de origen: {source_df.columns.tolist()}")
                
                # Crear un diccionario para mapear rsids a sus datos
                rsid_data = {}
                for _, row in source_df.iterrows():
                    rsid = row['rsid']
                    if rsid not in rsid_data:
                        rsid_data[rsid] = {}
                    
                    for source_col, target_col in column_mapping.items():
                        if source_col in row:
                            target_col_name = f"{prefix}_{target_col}"
                            rsid_data[rsid][target_col_name] = row[source_col]
                
                # Actualizar todos los campos de una vez para cada rsid
                for rsid, data in rsid_data.items():
                    mask = df_final['rsid'] == rsid
                    for col, value in data.items():
                        df_final.loc[mask, col] = value
                
                logger.info(f"Actualizados {len(rsid_data)} rsids para la fuente {prefix}")
            
            # 4. Obtener y procesar datos de cada fuente
            logger.info("=== Procesando datos GWAS ===")
            gwas_df = self.query_gwas_data(rs_ids, chromosome, position)
            logger.info(f"Resultados GWAS: {gwas_df.shape if gwas_df is not None else 'None'}")
            if gwas_df is not None and not gwas_df.empty:
                logger.info(f"Columnas GWAS: {gwas_df.columns.tolist()}")
            update_source_data(gwas_df, 'gwas', {
                'disease_trait': 'disease_trait',
                'p_value': 'p_value',
                'or_beta': 'or_beta'
            })
            
            logger.info("=== Procesando datos ClinVar ===")
            clinvar_df = self.query_clinvar_data(rs_ids, chromosome, position)
            logger.info(f"Resultados ClinVar: {clinvar_df.shape if clinvar_df is not None else 'None'}")
            if clinvar_df is not None and not clinvar_df.empty:
                logger.info(f"Columnas ClinVar: {clinvar_df.columns.tolist()}")
            update_source_data(clinvar_df, 'clinvar', {
                'variation_id': 'variation_id',
                'clinical_significance': 'clinical_significance',
                'phenotype': 'phenotype',
                'reference_allele': 'reference_allele',
                'alternate_allele': 'alternate_allele'
            })
            
            logger.info("=== Procesando datos dbSNP ===")
            dbsnp_df = self.query_dbsnp_data(rs_ids, chromosome, position)
            logger.info(f"Resultados dbSNP: {dbsnp_df.shape if dbsnp_df is not None else 'None'}")
            if dbsnp_df is not None and not dbsnp_df.empty:
                logger.info(f"Columnas dbSNP: {dbsnp_df.columns.tolist()}")
            update_source_data(dbsnp_df, 'dbsnp', {
                'reference_allele': 'reference_allele',
                'alternate_allele': 'alternate_allele',
                'allele_frequency': 'allele_frequency',
                'population': 'population'
            })
            
            # 5. Verificar y reportar resultados
            logger.info("=== Verificando resultados finales ===")
            logger.info(f"Columnas finales: {df_final.columns.tolist()}")
            logger.info(f"Forma final: {df_final.shape}")
            
            total_variants = len(df_final)
            gwas_variants = df_final['gwas_disease_trait'].notna().sum()
            clinvar_variants = df_final['clinvar_variation_id'].notna().sum()
            dbsnp_variants = df_final['dbsnp_reference_allele'].notna().sum()
            
            logger.info(f"""
            Resumen de procesamiento:
            - Total de SNPs válidos: {total_variants}
            - Variantes GWAS encontradas: {gwas_variants}
            - Variantes ClinVar encontradas: {clinvar_variants}
            - Variantes dbSNP encontradas: {dbsnp_variants}
            """)
            
            return df_final
            
        except Exception as e:
            logger.error(f"Error al procesar datos genéticos: {str(e)}")
            logger.error(f"Tipo de error: {type(e)}")
            logger.error(f"Stack trace: {e.__traceback__}")
            raise

def main():
    """Main function to update the genomic database."""
    logger.info("Starting genomic database update process...")
    try:
        db_manager = GenomicDatabaseManager()
        db_manager.update_database()
        db_manager.close()
        logger.info("Genomic database update completed successfully")
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 