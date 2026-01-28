from apscheduler.schedulers.background import BackgroundScheduler
from collector import FleetCoteDataCollector
from db_utils import IOTSessionLocal, FleetSessionLocal
from models import IOTTelemetry, Trip, Violation
import logging
from datetime import datetime, timedelta
import safety_score
import violations
import maintenance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

collector = FleetCoteDataCollector()
scheduler = BackgroundScheduler()

def collect_telemetry_all_vehicles():
    """Telemetry collection disabled in read-only mode"""
    logger.info("Telemetry collection skipped: Read-only mode active.")

def run_ml_pipelines():
    """ML inference pipelines (write-back) disabled in read-only mode"""
    logger.info("ML pipelines (write operations) skipped: Read-only mode active.")


def start_scheduler():
    # Schedule collection jobs (if still needed)
    scheduler.add_job(collect_telemetry_all_vehicles, 'interval', minutes=5)
    
    # Schedule ML jobs
    scheduler.add_job(run_ml_pipelines, 'interval', minutes=15)
    
    scheduler.start()
    logger.info("Scheduler started.")

def stop_scheduler():
    scheduler.shutdown()
    logger.info("Scheduler stopped.")

