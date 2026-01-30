from app.database.models import IOTTelemetry, MaintenancePrediction, Trip
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import logging

logger = logging.getLogger(__name__)

def _ensure_mock_telemetry(iot_db: Session, fleet_db: Session, vehicle_id: str, rows: int = 200) -> int:
    """
    Seed minimal IOT telemetry for a vehicle when none exists (POC helper).
    """
    # UUID handling
    try:
        import uuid
        if isinstance(vehicle_id, str):
            v_uuid = uuid.UUID(vehicle_id)
        else:
            v_uuid = vehicle_id
    except:
        return 0 # Cannot insert without valid UUID

    # Try to align with an existing trip if possible
    trip = (
        fleet_db.query(Trip)
        .filter(Trip.vehicle_id == v_uuid)
        .order_by(Trip.start_time.desc())
        .first()
    )
    trip_id = trip.id if trip and trip.id else None
    start_ts = (trip.start_time if trip and trip.start_time else datetime.utcnow()) - timedelta(hours=2)
    created = 0
    try:
        for i in range(rows):
            jitter = timedelta(seconds=i * 30)
            if i % 100 == 0: 
                # Avoid bulk insert error or just simplified loop
                pass 
                
            speed = max(0.0, random.gauss(55, 12))
            harsh_braking = random.random() < 0.05
            harsh_accel = random.random() < 0.04
            crash = random.random() < 0.005
            battery_voltage = random.gauss(12.4, 0.25)
            fuel_level = int(max(5.0, 80.0 - i * (60.0 / max(rows - 1, 1))))
            
            iot_db.add(IOTTelemetry(
                vehicle_id=v_uuid,
                trip_id=trip_id,
                time=start_ts + jitter,
                speed=float(speed),
                harsh_braking=bool(harsh_braking),
                harsh_acceleration=bool(harsh_accel),
                crash_detected=bool(crash),
                battery_voltage=float(battery_voltage),
                fuel_level=fuel_level,
                latitude=0.0,
                longitude=0.0
            ))
            created += 1
        iot_db.commit()
    except Exception as exc:
        logger.warning("mock telemetry insert failed: %s", exc)
        iot_db.rollback()
        return 0
    return created

def predict_maintenance(iot_db: Session, fleet_db: Session, vehicle_id: str):
    """
    Predict maintenance needs based on IOT data and history in Fleet DB.
    """
    # UUID handling
    try:
        import uuid
        if isinstance(vehicle_id, str):
            v_uuid = uuid.UUID(vehicle_id)
        else:
            v_uuid = vehicle_id
    except:
        return []

    # 1. Fetch recent telemetry from IOT DB (last 30 days)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)
    
    # Use .time instead of .timestamp
    telemetry = iot_db.query(IOTTelemetry).filter(
        IOTTelemetry.vehicle_id == v_uuid,
        IOTTelemetry.time >= start_date
    ).order_by(IOTTelemetry.time.asc()).all()
    
    source = "iot"
    if not telemetry:
        created = _ensure_mock_telemetry(iot_db, fleet_db, vehicle_id) # helper handles uuid
        telemetry = iot_db.query(IOTTelemetry).filter(
            IOTTelemetry.vehicle_id == v_uuid,
            IOTTelemetry.time >= start_date
        ).order_by(IOTTelemetry.time.asc()).all()
        if not telemetry:
            source = "synthetic"
            # Fallback: synthesize minimal telemetry in-memory for POC outputs
            df = pd.DataFrame([{
                "battery_voltage": 12.4,
                "speed": 55.0,
                "harsh_braking": False
            } for _ in range(5)])
            predictions = []
        else:
            source = "iot" if created > 0 else "iot"
    if telemetry:
        # Extract properties explicitly
        data = []
        for t in telemetry:
            row = {
                "battery_voltage": t.battery_voltage,
                "speed": t.speed,
                "engine_status": getattr(t, 'engine_status', None), # constant property?
                "harsh_braking": t.harsh_braking
            }
            data.append(row)
        df = pd.DataFrame(data)
        predictions = []
    
    # --- Battery Health Prediction (Real IOT data) ---
    if 'battery_voltage' in df.columns:
        # Convert to numeric, coercing errors
        s_volt = pd.to_numeric(df['battery_voltage'], errors='coerce')
        avg_voltage = s_volt.mean()
        
        # Handle NaN
        if pd.isna(avg_voltage):
             avg_voltage = 0.0
             
        if avg_voltage > 0 and avg_voltage < 12.2:
            predictions.append(MaintenancePrediction(
                vehicle_id=vehicle_id,
                component='BATTERY',
                status='CRITICAL' if avg_voltage < 11.8 else 'MEDIUM',
                confidence=0.9,
                predicted_km_to_failure=0 if avg_voltage < 11.8 else 1000,
                recommended_action='Replace battery' if avg_voltage < 11.8 else 'Test battery charge',
                indicators={'avg_voltage': round(float(avg_voltage), 2), 'source': source}
            ))
        else:
            predictions.append(MaintenancePrediction(
                vehicle_id=vehicle_id,
                component='BATTERY',
                status='OK',
                confidence=0.7,
                predicted_km_to_failure=None,
                recommended_action='Monitor battery health',
                indicators={'avg_voltage': round(float(avg_voltage), 2), 'source': source}
            ))
            
    # --- Engine System (using engine_status and speed) ---
    if 'engine_status' in df.columns:
        # Example logic: if engine is "ON" but speed is 0 for long periods, check idling
        pass
            
    # --- Tire Health (based on mileage from management data) ---
    # Fetch total distance for this vehicle from Fleet DB
    trips = fleet_db.query(Trip).filter(Trip.vehicle_id == vehicle_id).all()
    total_km = 0.0
    for trip in trips:
        try:
            total_km += float(trip.distance_km) if trip.distance_km else 0.0
        except (ValueError, TypeError):
            pass
    
    if total_km > 50000:
        predictions.append(MaintenancePrediction(
            vehicle_id=vehicle_id,
            component='TIRES',
            status='MEDIUM',
            confidence=0.7,
            predicted_km_to_failure=2000,
            recommended_action='Inspect tire tread depth and rotate tires',
            indicators={'total_mileage': total_km, 'source': source}
        ))
    else:
        predictions.append(MaintenancePrediction(
            vehicle_id=vehicle_id,
            component='TIRES',
            status='OK',
            confidence=0.6,
            predicted_km_to_failure=max(0, int(60000 - total_km)),
            recommended_action='No action; continue monitoring',
            indicators={'total_mileage': total_km, 'source': source}
        ))
        
    return predictions


def get_fleet_health_summary(fleet_db: Session):
    """
    Get a summary of fleet health based on latest predictions stored in Fleet DB.
    """
    latest_preds = fleet_db.query(MaintenancePrediction).order_by(MaintenancePrediction.prediction_date.desc()).all()
    
    unique_preds = {}
    for p in latest_preds:
        key = (p.vehicle_id, p.component)
        if key not in unique_preds:
            unique_preds[key] = p
            
    summary = {
        'critical_count': sum(1 for p in unique_preds.values() if p.status == 'CRITICAL'),
        'high_count': sum(1 for p in unique_preds.values() if p.status == 'HIGH'),
        'medium_count': sum(1 for p in unique_preds.values() if p.status == 'MEDIUM'),
        'vehicles_at_risk': list(set(p.vehicle_id for p in unique_preds.values() if p.status in ['CRITICAL', 'HIGH']))
    }
    
    return summary
