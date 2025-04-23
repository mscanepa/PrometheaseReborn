#!/usr/bin/env python3
"""
Test script for verifying the updated ClinVar query functionality.
"""

import sys
import logging
from src.genomic_database_manager import GenomicDatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting test...")
        
        # Initialize database manager
        db_manager = GenomicDatabaseManager()
        
        # Test with a small set of rs_ids
        test_rs_ids = ['rs1234', 'rs5678']
        
        logger.info("Testing ClinVar query...")
        results = db_manager.query_clinvar_data(test_rs_ids)
        
        logger.info(f"Query completed. Results shape: {results.shape if results is not None else 'None'}")
        if results is not None and not results.empty:
            logger.info("Columns in results:")
            for col in results.columns:
                logger.info(f"  - {col}")
        
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        sys.exit(1)
    finally:
        if 'db_manager' in locals():
            db_manager.close()

if __name__ == "__main__":
    main() 