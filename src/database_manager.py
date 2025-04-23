#!/usr/bin/env python3
"""
Database Manager for TheModernPromethease

This script manages the local databases used by the application:
- Downloads and updates data sources
- Maintains local copies of databases
- Provides access to the data
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, Optional
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define data sources and their configurations
DATA_SOURCES: Dict[str, Dict] = {
    'gwas': {
        'url': 'ftp://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest/gwas-catalog-associations.tsv',
        'raw_filename': 'gwas-catalog-associations.tsv',
        'processed_columns': ['SNPS', 'P-VALUE', 'OR or BETA', 'DISEASE/TRAIT'],
        'update_frequency': timedelta(days=7),  # Update weekly
        'last_update': None
    },
    'clinvar': {
        'url': 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz',
        'raw_filename': 'variant_summary.txt.gz',
        'processed_columns': ['Name', 'ClinicalSignificance', 'PhenotypeList', 
                            'Chromosome', 'Start', 'ReferenceAllele', 'AlternateAllele'],
        'update_frequency': timedelta(days=7),
        'last_update': None
    },
    'dbsnp': {
        'url': 'https://ftp.ncbi.nih.gov/snp/latest_release/JSON/data/refsnp-chr1.json.bz2',
        'raw_filename': 'refsnp-chr1.json.bz2',
        'processed_columns': ['refsnp_id', 'position', 'alleles'],
        'update_frequency': timedelta(days=14),  # Update bi-weekly
        'last_update': None
    }
}

class DatabaseManager:
    def __init__(self, base_dir: str = "data"):
        """Initialize the database manager."""
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.metadata_file = self.base_dir / "database_metadata.json"
        
        # Create necessary directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load metadata from file or create new if not exists."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {source: {'last_update': None} for source in DATA_SOURCES}
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)
    
    def download_source(self, source: str) -> bool:
        """Download a data source."""
        try:
            config = DATA_SOURCES[source]
            url = config['url']
            filename = config['raw_filename']
            output_path = self.raw_dir / filename
            
            logger.info(f"Downloading {source} data...")
            
            # Download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            
            with open(output_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True
            ) as pbar:
                for data in response.iter_content(block_size):
                    pbar.update(len(data))
                    f.write(data)
            
            # Update metadata
            self.metadata[source]['last_update'] = datetime.now().isoformat()
            self._save_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {source}: {str(e)}")
            return False
    
    def process_source(self, source: str) -> bool:
        """Process a downloaded data source."""
        try:
            config = DATA_SOURCES[source]
            input_path = self.raw_dir / config['raw_filename']
            output_path = self.processed_dir / f"{source}_processed.parquet"
            
            logger.info(f"Processing {source} data...")
            
            # Define efficient data types
            dtypes = {
                'CHR_ID': 'category',
                'CHR_POS': 'int32',
                'SNPS': 'category',
                'P-VALUE': 'float32',
                'OR or BETA': 'string'
            } if source == 'gwas' else {
                'Chromosome': 'category',
                'Start': 'int32',
                'ReferenceAllele': 'category',
                'AlternateAllele': 'category',
                'ClinicalSignificance': 'category',
                'PhenotypeList': 'string'
            } if source == 'clinvar' else {
                'refsnp_id': 'category',
                'position': 'int32',
                'alleles': 'string'
            }
            
            # Read and process data
            df = pd.read_csv(
                input_path,
                sep='\t',
                low_memory=False,
                dtype=dtypes,
                usecols=config['processed_columns']
            )
            
            # Convert to Parquet
            table = pa.Table.from_pandas(df)
            pq.write_table(
                table,
                output_path,
                compression='zstd',
                version='2.6'
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {source}: {str(e)}")
            return False
    
    def needs_update(self, source: str) -> bool:
        """Check if a source needs to be updated."""
        if source not in self.metadata or not self.metadata[source]['last_update']:
            return True
        
        last_update = datetime.fromisoformat(self.metadata[source]['last_update'])
        update_frequency = DATA_SOURCES[source]['update_frequency']
        
        return datetime.now() - last_update > update_frequency
    
    def update_all_sources(self, force: bool = False) -> None:
        """Update all data sources if needed."""
        for source in DATA_SOURCES:
            if force or self.needs_update(source):
                logger.info(f"Updating {source}...")
                if self.download_source(source):
                    self.process_source(source)
    
    def get_data(self, source: str) -> Optional[pd.DataFrame]:
        """Get processed data for a source."""
        try:
            file_path = self.processed_dir / f"{source}_processed.parquet"
            if not file_path.exists():
                return None
            
            return pd.read_parquet(file_path)
            
        except Exception as e:
            logger.error(f"Error reading {source} data: {str(e)}")
            return None

def main():
    """Main function to update all databases."""
    manager = DatabaseManager()
    manager.update_all_sources()

if __name__ == "__main__":
    main() 