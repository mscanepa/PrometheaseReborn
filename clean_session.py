#!/usr/bin/env python3
"""
Script to ensure a clean Python session and test the updated code.
"""

import sys
import os
import importlib
import logging
from src.genomic_database_manager import GenomicDatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_session():
    """Ensure a clean Python session."""
    try:
        # Clear any existing modules
        for module in list(sys.modules.keys()):
            if module.startswith('src.'):
                del sys.modules[module]
        
        # Force reload the module
        importlib.invalidate_caches()
        from src.genomic_database_manager import GenomicDatabaseManager
        
        logger.info("Session cleaned successfully")
        return True
    except Exception as e:
        logger.error(f"Error cleaning session: {str(e)}")
        return False

def test_updated_code():
    """Test the updated code with a clean session."""
    try:
        if not clean_session():
            raise Exception("Failed to clean session")
        
        logger.info("Testing with clean session...")
        db_manager = GenomicDatabaseManager()
        
        # Test with real data
        test_rs_ids = ['rs1234', 'rs5678']
        logger.info(f"Testing with {len(test_rs_ids)} rs_ids")
        
        # Test ClinVar query
        results = db_manager.query_clinvar_data(test_rs_ids)
        logger.info(f"ClinVar query results shape: {results.shape if results is not None else 'None'}")
        
        if results is not None and not results.empty:
            logger.info("Columns in results:")
            for col in results.columns:
                logger.info(f"  - {col}")
            
            # Verify no duplicate columns
            if any('_x' in col or '_y' in col for col in results.columns):
                raise Exception("Found duplicate columns in results")
        
        logger.info("Test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False
    finally:
        if 'db_manager' in locals():
            db_manager.close()

if __name__ == "__main__":
    success = test_updated_code()
    sys.exit(0 if success else 1) 