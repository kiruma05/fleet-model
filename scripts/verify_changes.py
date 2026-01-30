import sys
import os

# Ensure we can import from local dir
sys.path.append(os.getcwd())

try:
    print("Importing modules...")
    from ml_pipeline import train_safety_model
    from app.database.models import IOTTelemetry, Trip
    from app.database.db_utils import get_fleet_db, get_iot_db, get_ml_db
    import xgboost
    print(f"XGBoost version: {xgboost.__version__}")
    print("Imports successful.")
    
    # Optional: Dry run connection check
    # print("Checking DB connections...")
    # fleet_db = next(get_fleet_db())
    # iot_db = next(get_iot_db())
    # print("DB connections initialized (but not queried).")
    
except Exception as e:
    print(f"Verification Failed: {e}")
    sys.exit(1)
