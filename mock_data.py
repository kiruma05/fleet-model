import random
from datetime import datetime, timedelta
from db_utils import IOTSessionLocal, FleetSessionLocal, init_db
from models import IOTTelemetry, Trip, Violation, MaintenancePrediction
import uuid

def generate_mock_data():
    iot_db = IOTSessionLocal()
    fleet_db = FleetSessionLocal()
    init_db()
    
    # Use UUIDs to match the models
    vehicle_id = uuid.uuid4()
    driver_id = uuid.uuid4()
    
    print(f"Generating mock data for {vehicle_id}...")
    
    # 1. Create a Trip in Fleet DB (Supabase)
    trip_id = uuid.uuid4()
    start_time = datetime.utcnow() - timedelta(hours=2)
    end_time = datetime.utcnow() - timedelta(minutes=5)
    
    new_trip = Trip(
        id=trip_id,
        main_driver_id=driver_id,
        vehicle_id=vehicle_id,
        start_time=start_time,
        end_time=end_time,
        distance_km="50.5",
        fuel_used="5.2",
        status="completed"
    )
    fleet_db.add(new_trip)
    
    # 2. Generate IOT Telemetry in IOT DB (VPS)
    current_time = start_time
    while current_time <= end_time:
        speed = random.uniform(40, 80)
        harsh_braking = False
        if random.random() > 0.95:
            speed = random.uniform(105, 120)
        if random.random() > 0.98:
            harsh_braking = True
            
        telemetry = IOTTelemetry(
            vehicle_id=str(vehicle_id),
            trip_id=str(trip_id),
            timestamp=current_time,
            speed=speed,
            harsh_braking=harsh_braking,
            fuel_level=random.uniform(20, 80),
            battery_voltage=random.uniform(11.5, 12.8),
            engine_status="ON",
            latitude=-1.2833 + random.uniform(-0.01, 0.01),
            longitude=36.8167 + random.uniform(-0.01, 0.01)
        )
        iot_db.add(telemetry)
        current_time += timedelta(minutes=1)
        
    fleet_db.commit()
    iot_db.commit()
    print(f"Mock data generated. Trip ID: {trip_id}")
    iot_db.close()
    fleet_db.close()
    return trip_id

if __name__ == "__main__":
    generate_mock_data()
