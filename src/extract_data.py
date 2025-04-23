#!/usr/bin/env python3
"""
Data Extraction Module for TheModernPromethease

This module handles the extraction and preprocessing of genetic data from various sources:
- GWAS Catalog
- ClinVar
- dbSNP

It downloads, decompresses, and processes the data into standardized formats with compression.
"""

import os
import gzip
import bz2
import json
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
import shutil
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define data sources and their configurations
DATA_SOURCES: Dict[str, Dict] = {
    'gwas': {
        'url': 'ftp://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest/gwas-catalog-associations.tsv',
        'raw_filename': 'gwas-catalog-associations.tsv',
        'processed_columns': ['SNPS', 'P-VALUE', 'OR or BETA', 'DISEASE/TRAIT'],
        'compression': None
    },
    'clinvar': {
        'url': 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz',
        'raw_filename': 'variant_summary.txt.gz',
        'processed_columns': ['Name', 'ClinicalSignificance', 'PhenotypeList', 
                            'Chromosome', 'Start', 'ReferenceAllele', 'AlternateAllele'],
        'compression': 'gz'
    },
    'dbsnp': {
        'url': 'https://ftp.ncbi.nih.gov/snp/latest_release/JSON/data/refsnp-chr1.json.bz2',
        'raw_filename': 'refsnp-chr1.json.bz2',
        'processed_columns': ['refsnp_id', 'position', 'alleles'],
        'compression': 'bz2'
    }
}

def download_file(url: str, output_path: str) -> bool:
    """
    Download a file from a URL with progress bar and error handling.
    
    Args:
        url: URL of the file to download
        output_path: Path where to save the downloaded file
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        logger.info(f"Iniciando descarga de {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=f"Descargando {os.path.basename(output_path)}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        
        logger.info(f"Descarga completada: {output_path}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error al descargar {url}: {str(e)}")
        return False

def decompress_file(input_path: str, compression: Optional[str]) -> str:
    """
    Decompress a file if needed and return the path to the decompressed file.
    
    Args:
        input_path: Path to the compressed file
        compression: Type of compression ('gz', 'bz2', or None)
        
    Returns:
        str: Path to the decompressed file
    """
    if compression is None:
        return input_path
        
    output_path = input_path.rstrip('.gz').rstrip('.bz2')
    
    try:
        logger.info(f"Descomprimiendo archivo: {input_path}")
        if compression == 'gz':
            with gzip.open(input_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    f_out.write(f_in.read())
        elif compression == 'bz2':
            with bz2.open(input_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    f_out.write(f_in.read())
        
        logger.info(f"Descompresión completada: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error al descomprimir {input_path}: {str(e)}")
        return input_path

def process_gwas_data(input_path: str, output_path: str) -> bool:
    """
    Process GWAS Catalog data and save in efficient format.
    
    Args:
        input_path: Path to the raw GWAS data
        output_path: Path to save the processed data
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        logger.info("Procesando datos del GWAS Catalog...")
        
        # Define column types for memory efficiency
        dtypes = {
            'CHR_ID': 'category',
            'CHR_POS': 'int32',
            'SNPS': 'category',
            'P-VALUE': 'float32',  # Use float32 instead of float64 where precision is sufficient
            'OR or BETA': 'string'
        }
        
        # Read data in chunks
        chunksize = 100000
        chunks = []
        
        for chunk in pd.read_csv(
            input_path,
            sep='\t',
            low_memory=False,
            dtype=dtypes,
            chunksize=chunksize
        ):
            # Process each chunk
            chunk = chunk[list(dtypes.keys())]
            chunks.append(chunk)
            
        # Combine chunks
        df = pd.concat(chunks, ignore_index=True)
        
        # Convert to PyArrow table
        table = pa.Table.from_pandas(df)
        
        # Save as Parquet with compression
        pq.write_table(
            table,
            output_path.replace('.csv', '.parquet'),
            compression='zstd',
            version='2.6'
        )
        
        logger.info(f"Datos del GWAS Catalog guardados en formato Parquet: {output_path.replace('.csv', '.parquet')}")
        return True
        
    except Exception as e:
        logger.error(f"Error al procesar datos del GWAS Catalog: {str(e)}")
        return False

def process_clinvar_data(input_path: str, output_path: str) -> bool:
    """
    Process ClinVar data and save in efficient format with compression.
    
    Args:
        input_path: Path to the raw ClinVar data
        output_path: Path to save the processed data
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        logger.info("Procesando datos de ClinVar...")
        
        # Define column types for memory efficiency
        dtypes = {
            'Chromosome': 'category',  # Use category for string columns with few unique values
            'Start': 'int32',         # Use int32 instead of int64 where possible
            'ReferenceAllele': 'category',
            'AlternateAllele': 'category',
            'ClinicalSignificance': 'category',
            'PhenotypeList': 'string'  # Keep as string for free text
        }
        
        # Read data in chunks
        chunksize = 100000
        chunks = []
        
        for chunk in pd.read_csv(
            input_path,
            sep='\t',
            low_memory=False,
            dtype=dtypes,
            chunksize=chunksize
        ):
            # Process each chunk
            chunk = chunk[list(dtypes.keys())]
            chunks.append(chunk)
            
        # Combine chunks
        df = pd.concat(chunks, ignore_index=True)
        
        # Convert to PyArrow table
        table = pa.Table.from_pandas(df)
        
        # Save as Parquet with compression
        pq.write_table(
            table,
            output_path.replace('.csv', '.parquet'),
            compression='zstd',  # Better compression than gzip
            version='2.6'        # Latest stable version
        )
        
        logger.info(f"Datos de ClinVar guardados en formato Parquet: {output_path.replace('.csv', '.parquet')}")
        return True
        
    except Exception as e:
        logger.error(f"Error al procesar datos de ClinVar: {str(e)}")
        return False

def process_dbsnp_data(input_path: str, output_path: str) -> bool:
    """
    Process dbSNP data and save in efficient format.
    
    Args:
        input_path: Path to the raw dbSNP data
        output_path: Path to save the processed data
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        logger.info("Procesando datos de dbSNP...")
        
        # Read and process JSON data
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        processed_data = []
        for entry in data:
            if 'refsnp_id' in entry and 'position' in entry and 'alleles' in entry:
                processed_data.append({
                    'refsnp_id': entry['refsnp_id'],
                    'position': entry['position'],
                    'alleles': json.dumps(entry['alleles'])
                })
        
        # Create DataFrame with efficient types
        df = pd.DataFrame(processed_data, dtype={
            'refsnp_id': 'category',
            'position': 'int32',
            'alleles': 'string'
        })
        
        # Convert to PyArrow table
        table = pa.Table.from_pandas(df)
        
        # Save as Parquet with compression
        pq.write_table(
            table,
            output_path.replace('.csv', '.parquet'),
            compression='zstd',
            version='2.6'
        )
        
        logger.info(f"Datos de dbSNP guardados en formato Parquet: {output_path.replace('.csv', '.parquet')}")
        return True
        
    except Exception as e:
        logger.error(f"Error al procesar datos de dbSNP: {str(e)}")
        return False

def download_clinvar_data(output_dir: str) -> Optional[str]:
    """
    Download ClinVar data.
    
    Args:
        output_dir: Directory to save the downloaded file
        
    Returns:
        Optional[str]: Path to the downloaded file if successful, None otherwise
    """
    try:
        logger.info("Iniciando descarga de datos de ClinVar...")
        
        # Verificar si ya existe el archivo
        output_file = os.path.join(output_dir, 'variant_summary.txt.gz')
        if os.path.exists(output_file):
            logger.info("Los datos de ClinVar ya existen, omitiendo descarga")
            return output_file
        
        # URL para ClinVar
        url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
        
        # Descargar archivo
        if not download_file(url, output_file):
            raise Exception("Error al descargar datos de ClinVar")
        
        logger.info("Descarga de datos de ClinVar completada exitosamente")
        return output_file
        
    except Exception as e:
        logger.error(f"Error al descargar datos de ClinVar: {str(e)}")
        return None

def download_gwas_data(output_dir: str) -> Optional[str]:
    """
    Download GWAS Catalog data.
    
    Args:
        output_dir: Directory to save the downloaded file
        
    Returns:
        Optional[str]: Path to the downloaded file if successful, None otherwise
    """
    try:
        logger.info("Iniciando descarga de datos del GWAS Catalog...")
        
        # Verificar si ya existe el archivo
        output_file = os.path.join(output_dir, 'gwas-catalog-associations.tsv')
        if os.path.exists(output_file):
            logger.info("Los datos del GWAS Catalog ya existen, omitiendo descarga")
            return output_file
        
        # URL para GWAS Catalog
        url = "https://www.ebi.ac.uk/gwas/api/search/downloads/alternative"
        
        # Descargar archivo
        if not download_file(url, output_file):
            raise Exception("Error al descargar datos del GWAS Catalog")
        
        logger.info("Descarga de datos del GWAS Catalog completada exitosamente")
        return output_file
        
    except Exception as e:
        logger.error(f"Error al descargar datos del GWAS Catalog: {str(e)}")
        return None

def download_dbsnp_data(output_dir: str) -> Optional[str]:
    """
    Download dbSNP data.
    
    Args:
        output_dir: Directory to save the downloaded file
        
    Returns:
        Optional[str]: Path to the downloaded file if successful, None otherwise
    """
    try:
        logger.info("Iniciando descarga de datos de dbSNP...")
        
        # Verificar si ya existe el archivo
        output_file = os.path.join(output_dir, 'refsnp-chr1.json.bz2')
        if os.path.exists(output_file):
            logger.info("Los datos de dbSNP ya existen, omitiendo descarga")
            return output_file
        
        # URL para dbSNP
        url = "https://ftp.ncbi.nih.gov/snp/latest_release/JSON/data/refsnp-chr1.json.bz2"
        
        # Descargar archivo
        if not download_file(url, output_file):
            raise Exception("Error al descargar datos de dbSNP")
        
        logger.info("Descarga de datos de dbSNP completada exitosamente")
        return output_file
        
    except Exception as e:
        logger.error(f"Error al descargar datos de dbSNP: {str(e)}")
        return None

def main():
    """Función principal para extraer y procesar datos."""
    try:
        logger.info("Iniciando proceso de extracción y procesamiento de datos...")
        
        # Crear directorios necesarios
        raw_dir = 'data/raw'
        processed_dir = 'data/processed'
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        logger.info(f"Directorios creados: {raw_dir}, {processed_dir}")
        
        # Descargar y procesar datos de ClinVar
        logger.info("Procesando datos de ClinVar...")
        clinvar_file = download_clinvar_data(raw_dir)
        if not clinvar_file:
            raise Exception("Error al descargar datos de ClinVar")
        
        clinvar_processed = os.path.join(processed_dir, 'clinvar_processed.csv')
        if not process_clinvar_data(clinvar_file, clinvar_processed):
            raise Exception("Error al procesar datos de ClinVar")
        
        # Descargar y procesar datos del GWAS Catalog
        logger.info("Procesando datos del GWAS Catalog...")
        gwas_file = download_gwas_data(raw_dir)
        if not gwas_file:
            raise Exception("Error al descargar datos del GWAS Catalog")
        
        gwas_processed = os.path.join(processed_dir, 'gwas_catalog_processed.csv')
        if not process_gwas_data(gwas_file, gwas_processed):
            raise Exception("Error al procesar datos del GWAS Catalog")
        
        # Descargar y procesar datos de dbSNP (opcional)
        logger.info("Procesando datos de dbSNP...")
        dbsnp_file = download_dbsnp_data(raw_dir)
        if dbsnp_file:
            dbsnp_processed = os.path.join(processed_dir, 'dbsnp_processed.csv')
            if not process_dbsnp_data(dbsnp_file, dbsnp_processed):
                logger.warning("Error al procesar datos de dbSNP, continuando sin ellos")
        else:
            logger.warning("No se pudieron descargar datos de dbSNP, continuando sin ellos")
        
        logger.info("Proceso de extracción y procesamiento de datos completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error en el proceso principal: {str(e)}")
        raise

if __name__ == "__main__":
    main() 