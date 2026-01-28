import pandas as pd
import numpy as np
from models import IOTTelemetry, Event, Trip
from sqlalchemy.orm import Session

def calculate_trip_features(iot_db: Session, fleet_db: Session, trip_id: str):
    """
    Calculate safety features for a completed trip using IOT data.
    """
    # 1. Fetch trip data from Fleet DB
    trip = fleet_db.query(Trip).filter(Trip.id == trip_id).first()
    if not trip:
        return None
    
    # 2. Fetch IOT Telemetry for this trip
    telemetry = iot_db.query(IOTTelemetry).filter(
        IOTTelemetry.trip_id == trip_id
    ).order_by(IOTTelemetry.timestamp.asc()).all()
    
    if not telemetry:
        return None
        
    df = pd.DataFrame([t.__dict__ for t in telemetry])
    
    # 3. Calculate harsh events from IOT data
    harsh_braking = df['harsh_braking'].sum() if 'harsh_braking' in df.columns else 0
    harsh_accel = df['harsh_acceleration'].sum() if 'harsh_acceleration' in df.columns else 0
    crash_count = df['crash_detected'].sum() if 'crash_detected' in df.columns else 0
    
    # 4. Speed compliance
    speed_limit = 100 
    speeding_instances = len(df[df['speed'] > speed_limit]) if 'speed' in df.columns else 0
    speed_compliance = 1 - (speeding_instances / len(df)) if len(df) > 0 else 1.0
        
    # Handle distance_km as string from Supabase
    try:
        distance = float(trip.distance_km) if trip.distance_km else 1.0
    except (ValueError, TypeError):
        distance = 1.0
    
    features = {
        'harsh_braking_rate': harsh_braking / distance,
        'harsh_accel_rate': harsh_accel / distance,
        'crash_events': crash_count,
        'speed_compliance': speed_compliance,
        'avg_speed': 0 # Not directly in production trip table
    }
    
    return features

def compute_safety_score(features):
    """
    Compute a weighted safety score (0-100) based on features.
    """
    if not features:
        return 0.0
    
    # Penalties
    braking_pen = min(features['harsh_braking_rate'] * 15, 30)
    accel_pen = min(features['harsh_accel_rate'] * 10, 20)
    crash_pen = 100 if features['crash_events'] > 0 else 0
    
    speed_score = features['speed_compliance'] * 40
    
    base_score = 60
    total_penalty = braking_pen + accel_pen + crash_pen
    
    final_score = (speed_score) + (base_score - total_penalty)
    
    return max(0, min(100, final_score))


