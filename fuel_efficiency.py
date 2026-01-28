from models import IOTTelemetry, Trip
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np

def calculate_trip_fuel_efficiency(iot_db: Session, fleet_db: Session, trip_id: str):
    """
    Calculate fuel efficiency for a specific trip using IOT data and trip details.
    """
    # 1. Fetch trip details from Fleet DB
    trip = fleet_db.query(Trip).filter(Trip.id == trip_id).first()
    if not trip:
        return None
        
    try:
        distance = float(trip.distance_km) if trip.distance_km else 0.0
    except (ValueError, TypeError):
        distance = 0.0
        
    if distance <= 0:
        return None
    
    # 2. Fetch fuel data from IOT DB for this trip
    # Ensure trip_id is handled as UUID if possible
    try:
        import uuid
        if isinstance(trip_id, str):
            trip_search_id = uuid.UUID(trip_id)
        else:
            trip_search_id = trip_id
    except:
        trip_search_id = trip_id # Fallback
        
    telemetry = iot_db.query(IOTTelemetry).filter(
        IOTTelemetry.trip_id == trip_search_id
    ).order_by(IOTTelemetry.time.asc()).all()
    
    if not telemetry:
        return None
        
    # Manually extract fields because __dict__ doesn't include properties
    data = []
    for t in telemetry:
        row = {
            'timestamp': t.time,
            'fuel_level': t.fuel_level
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # Method A: Use direct fuel_level delta
    if 'fuel_level' in df.columns:
        series = pd.to_numeric(df['fuel_level'], errors='coerce').dropna()
        if len(series) < 2:
            return None
        consumed = series.iloc[0] - series.iloc[-1]
        if consumed > 0:
            efficiency = distance / consumed
            return round(efficiency, 2)
            
    return None


def get_vehicle_fuel_efficiency_summary(iot_db: Session, fleet_db: Session, vehicle_id: str):
    """
    Get aggregated fuel efficiency for a vehicle across all trips.
    """
    trips = fleet_db.query(Trip).filter(Trip.vehicle_id == vehicle_id).all()
    efficiencies = []
    
    for trip in trips:
        eff = calculate_trip_fuel_efficiency(iot_db, fleet_db, trip.id)
        if eff:
            efficiencies.append(eff)
            
    if not efficiencies:
        return {"vehicle_id": vehicle_id, "avg_efficiency": None, "trip_count": 0}
        
    return {
        "vehicle_id": vehicle_id,
        "avg_efficiency": round(np.mean(efficiencies), 2),
        "trip_count": len(efficiencies)
    }
def get_driver_fuel_efficiency_summary(iot_db: Session, fleet_db: Session, driver_id: str):
    """
    Get aggregated fuel efficiency for a driver across all trips.
    """
    trips = fleet_db.query(Trip).filter(Trip.main_driver_id == driver_id).all()
    efficiencies = []
    
    for trip in trips:
        eff = calculate_trip_fuel_efficiency(iot_db, fleet_db, trip.id)
        if eff:
            efficiencies.append(eff)
            
    if not efficiencies:
        return {"driver_id": driver_id, "avg_efficiency": None, "trip_count": 0}
        
    return {
        "driver_id": driver_id,
        "avg_efficiency": round(np.mean(efficiencies), 2),
        "trip_count": len(efficiencies)
    }
