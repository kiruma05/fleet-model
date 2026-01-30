import logging
from datetime import datetime, timedelta
import pandas as pd
from app.services.collector import FleetCoteDataCollector
from app.database.db_utils import IOTSessionLocal, FleetSessionLocal, init_db
from app.database.models import IOTTelemetry, Trip
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def backfill_data(days=30):
    """Backfill disabled in read-only mode"""
    print("Backfill skipped: Read-only mode active.")


if __name__ == "__main__":
    import sys
    days_to_pull = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    backfill_data(days=days_to_pull)

