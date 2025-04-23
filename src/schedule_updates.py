#!/usr/bin/env python3
"""
Database Update Scheduler

This script schedules automatic updates of the genetic databases.
It can be run as a cron job or scheduled task.
"""

import os
import sys
import logging
from datetime import datetime
from database_manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_updates.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to update databases."""
    try:
        logger.info("Starting scheduled database update...")
        
        # Initialize database manager
        manager = DatabaseManager()
        
        # Check if any updates are needed
        needs_update = any(manager.needs_update(source) for source in manager.metadata)
        
        if needs_update:
            logger.info("Updates needed, starting update process...")
            manager.update_all_sources()
            logger.info("Database update completed successfully")
        else:
            logger.info("No updates needed at this time")
        
        # Log update status
        for source in manager.metadata:
            last_update = manager.metadata[source]['last_update']
            if last_update:
                last_update = datetime.fromisoformat(last_update)
                logger.info(f"{source} last updated: {last_update}")
        
    except Exception as e:
        logger.error(f"Error during scheduled update: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 