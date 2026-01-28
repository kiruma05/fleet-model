import time
import logging
from db_utils import init_db
from scheduler import start_scheduler, stop_scheduler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing Fleet ML POC...")
    
    # 1. Initialize Database
    logger.info("Initializing database...")
    init_db()
    
    # 2. Start Scheduler
    logger.info("Starting background scheduler...")
    start_scheduler()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down...")
        stop_scheduler()

if __name__ == "__main__":
    main()
