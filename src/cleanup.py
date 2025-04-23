#!/usr/bin/env python3
"""
Data Cleanup and Management Script

This script helps manage data files by:
1. Converting old CSV files to Parquet format
2. Cleaning up temporary files
3. Managing storage space
"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import logging
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_csv_to_parquet(csv_path: str, parquet_path: str) -> bool:
    """
    Convert a CSV file to Parquet format with compression.
    
    Args:
        csv_path: Path to the CSV file
        parquet_path: Path where the Parquet file will be saved
        
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        logger.info(f"Converting {csv_path} to Parquet format...")
        
        # Read CSV in chunks
        chunksize = 100000
        chunks = []
        
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            chunks.append(chunk)
        
        # Combine chunks
        df = pd.concat(chunks, ignore_index=True)
        
        # Convert to PyArrow table
        table = pa.Table.from_pandas(df)
        
        # Save as Parquet with compression
        pq.write_table(
            table,
            parquet_path,
            compression='zstd',
            version='2.6'
        )
        
        logger.info(f"Successfully converted to {parquet_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error converting {csv_path}: {str(e)}")
        return False

def cleanup_directory(directory: str, max_size_mb: int = 1000) -> None:
    """
    Clean up a directory by converting large CSV files to Parquet
    and removing temporary files.
    
    Args:
        directory: Path to the directory to clean up
        max_size_mb: Maximum size in MB before converting to Parquet
    """
    try:
        logger.info(f"Cleaning up directory: {directory}")
        
        # Convert large CSV files to Parquet
        for file in Path(directory).glob("*.csv"):
            file_size_mb = file.stat().st_size / (1024 * 1024)
            
            if file_size_mb > max_size_mb:
                parquet_path = file.with_suffix('.parquet')
                if convert_csv_to_parquet(str(file), str(parquet_path)):
                    # Remove original CSV if conversion was successful
                    file.unlink()
                    logger.info(f"Removed original CSV file: {file}")
        
        # Remove temporary files
        for file in Path(directory).glob("*.tmp"):
            file.unlink()
            logger.info(f"Removed temporary file: {file}")
        
        logger.info("Cleanup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

def main():
    """Main function to run cleanup operations."""
    # Define directories to clean up
    directories = [
        "data/raw",
        "data/processed"
    ]
    
    # Clean up each directory
    for directory in directories:
        if os.path.exists(directory):
            cleanup_directory(directory)
        else:
            logger.warning(f"Directory not found: {directory}")

if __name__ == "__main__":
    main() 