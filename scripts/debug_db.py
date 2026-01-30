import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("IOT_DATABASE_URL")
if not DB_URL:
    print("No IOT_DATABASE_URL")
    exit(1)

engine = create_engine(DB_URL)

from sqlalchemy.orm import sessionmaker
from app.database.models import IOTTelemetry

SessionLocal = sessionmaker(bind=engine)

try:
    with SessionLocal() as session:
        print("Connected with Session.")
        import uuid
        test_trip_id = uuid.UUID('366c8117-0319-4349-a36a-60fdf78781fa')
        print(f"Querying for trip_id: {test_trip_id}")
        
        telemetry = session.query(IOTTelemetry).filter(
            IOTTelemetry.trip_id == test_trip_id
        ).limit(5).all()
        
        print(f"Found {len(telemetry)} records (showing max 5).")
        
        for t in telemetry:
            print("--- Record ---")
            print(f"Time: {t.time}")
            print(f"Speed: {t.speed}")
            print(f"Ignition: {t.ignition}")
            print(f"Fuel Data: {t.fuel_data}")
            print(f"Vehicle Telemetry: {t.vehicle_telemetry}")
            
except Exception as e:
    print(f"ORM Error: {e}")
    import traceback
    traceback.print_exc()
