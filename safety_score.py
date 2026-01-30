from sqlalchemy import func
import pandas as pd
import numpy as np
from models import IOTTelemetry, Event, Trip, Violation, DriverSafetyScore
from sqlalchemy.orm import Session
import datetime

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
    # Fallback if rates are missing but counts/distance exist
    if 'harsh_braking_rate' not in features and 'harsh_braking_count' in features and 'distance_km' in features:
        features['harsh_braking_rate'] = features['harsh_braking_count'] / max(features['distance_km'], 0.1)
    
    if 'harsh_accel_rate' not in features and 'harsh_accel_count' in features and 'distance_km' in features:
        features['harsh_accel_rate'] = features['harsh_accel_count'] / max(features['distance_km'], 0.1)

    if 'crash_events' not in features:
        features['crash_events'] = 0

    if 'speed_compliance' not in features:
        features['speed_compliance'] = 1.0

    braking_pen = min(features.get('harsh_braking_rate', 0) * 15, 30)
    accel_pen = min(features.get('harsh_accel_rate', 0) * 10, 20)
    crash_pen = 100 if features['crash_events'] > 0 else 0
    
    speed_score = features['speed_compliance'] * 40
    
    base_score = 60
    total_penalty = braking_pen + accel_pen + crash_pen
    
    final_score = (speed_score) + (base_score - total_penalty)
    
    return max(0, min(100, final_score))

def calculate_driver_safety_score(db: Session, driver_id: str):
    """
    Calculate safety score for a driver based on their trips and violations.
    """
    # 1. Fetch all trips for the driver
    trips = db.query(Trip).filter(Trip.main_driver_id == driver_id).all()
    
    if not trips:
        return 0.0
        
    total_distance = 0.0
    for trip in trips:
        try:
            dist = float(trip.distance_km) if trip.distance_km else 0.0
            total_distance += dist
        except (ValueError, TypeError):
            continue
            
    if total_distance < 1.0:
        # Fallback for POC/Mock data if distance_km is not populated in Trip table
        # Assume average of 10km per trip if trips exist
        if trips:
            total_distance = len(trips) * 10.0
        else:
            return 0.0 # Truly no data
        
    # 2. Fetch all violations for the driver
    violations = db.query(Violation).filter(Violation.driver_id == driver_id).all()
    
    harsh_braking_count = 0
    harsh_accel_count = 0
    crash_count = 0
    speeding_count = 0 
    
    for v in violations:
        if v.violation_type == 'HARSH_BRAKING':
            harsh_braking_count += 1
        elif v.violation_type == 'HARSH_ACCELERATION':
            harsh_accel_count += 1
        elif v.violation_type == 'CRASH_INCIDENT':
            crash_count += 1
        elif v.violation_type == 'SPEEDING':
            speeding_count += 1
            
    # 3. Compute metrics
    # Speed compliance is tricky without raw points, 
    # but we can approximate it by saying more speeding violations = lower compliance
    # Let's say 1 speeding violation per 100km reduces compliance by 10%
    speeding_rate_per_100km = (speeding_count / total_distance) * 100
    speed_compliance = max(0.0, 1.0 - (speeding_rate_per_100km * 0.1))
    
    features = {
        'harsh_braking_count': harsh_braking_count,
        'harsh_accel_count': harsh_accel_count,
        'crash_events': crash_count,
        'distance_km': total_distance,
        'speed_compliance': speed_compliance
    }
    
    return compute_safety_score(features)

def update_driver_scores(db: Session):
    """
    Update safety scores for all drivers.
    """
    # Get all distinct driver IDs from Trip table
    driver_ids = db.query(Trip.main_driver_id).distinct().all()
    driver_ids = [d[0] for d in driver_ids if d[0]]
    
    for d_id in driver_ids:
        score = calculate_driver_safety_score(db, str(d_id))
        
        # Update or create record
        existing = db.query(DriverSafetyScore).filter(DriverSafetyScore.driver_id == d_id).first()
        
        # Count trips
        trip_count = db.query(Trip).filter(Trip.main_driver_id == d_id).count() 
        
        if existing:
            existing.score = score
            existing.trip_count = trip_count
            existing.last_updated = datetime.datetime.utcnow()
        else:
            new_score = DriverSafetyScore(
                driver_id=d_id,
                score=score,
                trip_count=trip_count
            )
            db.add(new_score)
            
    db.commit()
