import datetime
import pickle
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

import fuel_efficiency
import safety_score
import violations
from models import IOTTelemetry, Trip, ModelArtifact

MIN_SAMPLES = 5

def generate_mock_telemetry(
    iot_db: Session,
    fleet_db: Session,
    trips_limit: Optional[int] = 50,
    rows_per_trip: int = 20,
    reset: bool = False,
    seed: int = 42
):
    if reset:
        iot_db.query(IOTTelemetry).delete()
        iot_db.commit()
    trips_query = fleet_db.query(Trip).order_by(Trip.start_time.desc())
    if trips_limit:
        trips_query = trips_query.limit(trips_limit)
    trips = trips_query.all()
    if not trips:
        return {"status": "skipped", "reason": "no_trips"}
    random.seed(seed)
    created = 0
    for trip in trips:
        if not trip.id or not trip.vehicle_id:
            continue
        start_ts = trip.start_time or datetime.datetime.utcnow() - datetime.timedelta(hours=1)
        for i in range(rows_per_trip):
            jitter = datetime.timedelta(seconds=i * 30)
            speed = max(0.0, random.gauss(55, 12))
            harsh_braking = random.random() < 0.05
            harsh_accel = random.random() < 0.04
            crash = random.random() < 0.005
            battery_voltage = random.gauss(12.6, 0.3)
            iot_db.add(IOTTelemetry(
                vehicle_id=str(trip.vehicle_id),
                trip_id=str(trip.id),
                timestamp=start_ts + jitter,
                speed=float(speed),
                harsh_braking=bool(harsh_braking),
                harsh_acceleration=bool(harsh_accel),
                crash_detected=bool(crash),
                battery_voltage=float(battery_voltage),
                latitude=0.0,
                longitude=0.0
            ))
            created += 1
    iot_db.commit()
    return {"status": "ok", "rows_created": created, "trips_used": len(trips)}

def _save_model(ml_db: Session, name: str, model, feature_names: List[str], metrics: Dict):
    payload = pickle.dumps(model)
    latest = (
        ml_db.query(ModelArtifact)
        .filter(ModelArtifact.name == name)
        .order_by(ModelArtifact.version.desc())
        .first()
    )
    version = 1 if not latest else latest.version + 1
    artifact = ModelArtifact(
        name=name,
        version=version,
        trained_at=datetime.datetime.utcnow(),
        metrics=metrics,
        feature_names=feature_names,
        model_blob=payload
    )
    ml_db.add(artifact)
    ml_db.commit()
    return artifact

def load_latest_model(ml_db: Session, name: str):
    artifact = (
        ml_db.query(ModelArtifact)
        .filter(ModelArtifact.name == name)
        .order_by(ModelArtifact.version.desc())
        .first()
    )
    if not artifact:
        return None, None, None
    model = pickle.loads(artifact.model_blob)
    return model, artifact.feature_names or [], artifact.metrics or {}

import json
from xgboost import XGBClassifier, XGBRegressor

# ... (Previous imports kept if not replacing whole file, but I am replacing chunks)

def build_trip_safety_features(iot_db: Session, fleet_db: Session, trip_id: str) -> Optional[Dict]:
    # Ensure trip_id is treated as UUID if needed, but the model expects UUID object or string depending on driver. 
    # SQLAlchemy with Uuid(as_uuid=True) expects UUID objects usually.
    try:
        if isinstance(trip_id, str):
            import uuid
            trip_uuid = uuid.UUID(trip_id)
        else:
            trip_uuid = trip_id
    except ValueError:
        return None

    # Fetch IOT Telemetry
    telemetry = (
        iot_db.query(IOTTelemetry)
        .filter(IOTTelemetry.trip_id == trip_uuid)
        .order_by(IOTTelemetry.time.asc())
        .all()
    )
    
    if not telemetry:
        return None
        
    # Convert to DataFrame
    # Note: accessing .time alias or timestamp, and JSON fields
    data = []
    for t in telemetry:
        row = {
            "timestamp": t.time,
            "speed": t.speed or 0.0,
            "heading": t.heading or 0.0,
            "ignition": t.ignition,
            "harsh_braking": t.harsh_braking,
            "harsh_acceleration": t.harsh_acceleration,
            "crash_detected": t.crash_detected,
            "vehicle_telemetry": t.vehicle_telemetry
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # --- Feature Engineering ---
    
    # 1. Idling Ratio
    # Speed < 1km/h AND Ignition = True
    # We need to assume regular sampling or weighted by time. Simple count ratio for POC.
    idling_count = len(df[(df['speed'] < 1.0) & (df['ignition'] == True)])
    idling_ratio = idling_count / len(df) if len(df) > 0 else 0.0
    
    # 2. Cornering Intensity
    # Rate of change of heading. Handle 360-0 wrap around if sophisticated, but simple diff for now.
    # wrapping: diff of 359 to 1 is 2, not 358. 
    # crude fix: min(abs(diff), 360-abs(diff))
    if 'heading' in df.columns:
        df['heading_diff'] = df['heading'].diff().fillna(0.0)
        df['heading_change'] = df['heading_diff'].apply(lambda x: min(abs(x), 360 - abs(x)))
        cornering_intensity = df['heading_change'].mean()
    else:
        cornering_intensity = 0.0
        
    # 3. RPM Stress
    # Extract RPM from JSONB if exists
    def get_rpm(val):
        if not val: return 0
        if isinstance(val, dict): return val.get('rpm', 0)
        try:
            d = json.loads(val) if isinstance(val, str) else val
            return d.get('rpm', 0)
        except:
            return 0
            
    df['rpm'] = df['vehicle_telemetry'].apply(get_rpm)
    rpm_stress_count = len(df[df['rpm'] > 3000])
    rpm_stress_ratio = rpm_stress_count / len(df) if len(df) > 0 else 0.0
    
    trip = fleet_db.query(Trip).filter(Trip.id == trip_uuid).first()
    try:
        distance = float(trip.distance_km) if trip and trip.distance_km else 1.0
    except:
        distance = 1.0
        
    # Safe conversion helpers
    avg_speed = float(df["speed"].mean())
    if pd.isna(avg_speed): avg_speed = 0.0
    
    speed_std = float(df["speed"].std())
    if pd.isna(speed_std): speed_std = 0.0
    
    # Speed compliance calculation
    speed_limit = 100
    speeding_instances = len(df[df['speed'] > speed_limit]) if 'speed' in df.columns else 0
    speed_compliance = 1.0 - (speeding_instances / len(df)) if len(df) > 0 else 1.0
    
    features = {
        "avg_speed": avg_speed,
        "speed_std": speed_std,
        "harsh_braking_count": int(df["harsh_braking"].sum()) if "harsh_braking" in df.columns else 0,
        "harsh_accel_count": int(df["harsh_acceleration"].sum()) if "harsh_acceleration" in df.columns else 0,
        "idling_ratio": float(idling_ratio) if not pd.isna(idling_ratio) else 0.0,
        "cornering_intensity": float(cornering_intensity) if not pd.isna(cornering_intensity) else 0.0,
        "rpm_stress_ratio": float(rpm_stress_ratio) if not pd.isna(rpm_stress_ratio) else 0.0,
        "distance_km": distance,
        "speed_compliance": float(speed_compliance)
    }
    return features

    return features

def build_trip_fuel_features(iot_db: Session, fleet_db: Session, trip_id: str) -> Optional[Dict]:
    # UUID handling
    try:
        import uuid
        if isinstance(trip_id, str):
            trip_search_id = uuid.UUID(trip_id)
        else:
            trip_search_id = trip_id
    except:
        return None

    trip = fleet_db.query(Trip).filter(Trip.id == trip_search_id).first()
    if not trip:
        return None
        
    telemetry = (
        iot_db.query(IOTTelemetry)
        .filter(IOTTelemetry.trip_id == trip_search_id)
        .order_by(IOTTelemetry.time.asc())
        .all()
    )
    if not telemetry:
        return None
        
    # Extract properties explicitly
    data = []
    for t in telemetry:
        row = {
            "speed": t.speed,
            "harsh_braking": t.harsh_braking,
            "harsh_acceleration": t.harsh_acceleration,
            "timestamp": t.time
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    try:
        distance = float(trip.distance_km) if trip.distance_km else 0.0
    except (ValueError, TypeError):
        distance = 0.0
        
    if distance <= 0:
        return None
        
    avg_speed = float(df["speed"].mean())
    if pd.isna(avg_speed): avg_speed = 0.0
    
    speed_std = float(df["speed"].std())
    if pd.isna(speed_std): speed_std = 0.0

    features = {
        "avg_speed": avg_speed,
        "speed_std": speed_std,
        "harsh_braking_count": int(df["harsh_braking"].sum()) if "harsh_braking" in df.columns else 0,
        "harsh_accel_count": int(df["harsh_acceleration"].sum()) if "harsh_acceleration" in df.columns else 0,
        "distance_km": distance
    }
    return features

def build_vehicle_maintenance_features(iot_db: Session, fleet_db: Session, vehicle_id: str) -> Optional[Dict]:
    # UUID handling
    try:
        import uuid
        if isinstance(vehicle_id, str):
            v_uuid = uuid.UUID(vehicle_id)
        else:
            v_uuid = vehicle_id
    except:
        return None

    telemetry = (
        iot_db.query(IOTTelemetry)
        .filter(IOTTelemetry.vehicle_id == v_uuid)
        .order_by(IOTTelemetry.time.asc())
        .limit(5000)
        .all()
    )
    if not telemetry:
        return None
        
    # Extract properties explicit
    data = []
    for t in telemetry:
        row = {
            "battery_voltage": t.battery_voltage,
            "speed": t.speed,
            "harsh_braking": t.harsh_braking
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # Calculate totals from trips
    trips = fleet_db.query(Trip).filter(Trip.vehicle_id == v_uuid).all()
    total_km = 0.0
    for trip in trips:
        try:
            total_km += float(trip.distance_km) if trip.distance_km else 0.0
        except (ValueError, TypeError):
            pass
            
    features = {
        "avg_battery_voltage": float(df["battery_voltage"].mean()) if "battery_voltage" in df.columns else 0.0,
        "min_battery_voltage": float(df["battery_voltage"].min()) if "battery_voltage" in df.columns else 0.0,
        "avg_speed": float(df["speed"].mean()) if "speed" in df.columns else 0.0,
        "harsh_events": int(df["harsh_braking"].sum()) if "harsh_braking" in df.columns else 0,
        "total_km": total_km
    }
    return features
# Implementing Training Functions with XGBoost

def train_safety_model(iot_db: Session, fleet_db: Session, ml_db: Session):
    trips = fleet_db.query(Trip).all()
    rows = []
    targets = []
    for trip in trips:
        features = build_trip_safety_features(iot_db, fleet_db, str(trip.id))
        if not features:
            continue
        # Use existing scoring logic or new one. Using existing for target generation.
        score = safety_score.compute_safety_score(features) 
        rows.append(features)
        targets.append(score)
        
    if len(rows) < MIN_SAMPLES:
        return {"name": "safety_score", "status": "skipped", "reason": "not_enough_samples", "samples": len(rows)}
        
    df = pd.DataFrame(rows).fillna(0.0)
    feature_names = list(df.columns)
    X = df.values
    y = np.array(targets, dtype=float)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost Regressor
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    metrics = {"mae": float(mean_absolute_error(y_test, preds)), "r2": float(r2_score(y_test, preds))}
    
    artifact = _save_model(ml_db, "safety_score", model, feature_names, metrics)
    return {"name": "safety_score", "status": "trained", "version": artifact.version, "metrics": metrics, "samples": len(rows)}

def train_violations_risk_model(iot_db: Session, fleet_db: Session, ml_db: Session):
    trips = fleet_db.query(Trip).all()
    rows = []
    targets = []
    for trip in trips:
        features = build_trip_safety_features(iot_db, fleet_db, str(trip.id))
        if not features:
            continue
        # Detect violations (requires updating violations.py likely, but sticking to interface)
        detected = violations.detect_violations_from_iot(iot_db, fleet_db, str(trip.id))
        rows.append(features)
        targets.append(1 if detected else 0)
        
    if len(rows) < MIN_SAMPLES:
        return {"name": "violations_risk", "status": "skipped", "reason": "not_enough_samples", "samples": len(rows)}
        
    df = pd.DataFrame(rows).fillna(0.0)
    feature_names = list(df.columns)
    X = df.values
    y = np.array(targets, dtype=int)
    
    if len(set(y)) < 2:
        return {"name": "violations_risk", "status": "skipped", "reason": "insufficient_class_variance", "samples": len(rows)}
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # XGBoost Classifier
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    metrics = {"accuracy": float(accuracy_score(y_test, preds))}
    
    artifact = _save_model(ml_db, "violations_risk", model, feature_names, metrics)
    return {"name": "violations_risk", "status": "trained", "version": artifact.version, "metrics": metrics, "samples": len(rows)}

# Keeping fuel/maintenance similar but switching provided snippet to cover the safety logic first.
# I will do a partial replacement of the specific functions to keep the file valid.


def train_fuel_efficiency_model(iot_db: Session, fleet_db: Session, ml_db: Session):
    trips = fleet_db.query(Trip).all()
    rows = []
    targets = []
    for trip in trips:
        features = build_trip_fuel_features(iot_db, fleet_db, str(trip.id))
        if not features:
            continue
        efficiency = fuel_efficiency.calculate_trip_fuel_efficiency(iot_db, fleet_db, str(trip.id))
        if efficiency is None:
            continue
        rows.append(features)
        targets.append(efficiency)
    if len(rows) < MIN_SAMPLES:
        return {"name": "fuel_efficiency", "status": "skipped", "reason": "not_enough_samples", "samples": len(rows)}
    df = pd.DataFrame(rows).fillna(0.0)
    feature_names = list(df.columns)
    X = df.values
    y = np.array(targets, dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost Regressor
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    metrics = {"mae": float(mean_absolute_error(y_test, preds)), "r2": float(r2_score(y_test, preds))}
    artifact = _save_model(ml_db, "fuel_efficiency", model, feature_names, metrics)
    return {"name": "fuel_efficiency", "status": "trained", "version": artifact.version, "metrics": metrics, "samples": len(rows)}

def train_maintenance_risk_model(iot_db: Session, fleet_db: Session, ml_db: Session):
    vehicles = fleet_db.query(Trip.vehicle_id).distinct().all()
    rows = []
    targets = []
    for (vehicle_id,) in vehicles:
        if not vehicle_id:
            continue
        # Ensure ID handled correctly
        features = build_vehicle_maintenance_features(iot_db, fleet_db, str(vehicle_id))
        if not features:
            continue
        avg_voltage = features.get("avg_battery_voltage")
        total_km = features.get("total_km")
        risk = 1 if (avg_voltage and avg_voltage < 12.2) or (total_km and total_km > 50000) else 0
        rows.append(features)
        targets.append(risk)
    if len(rows) < MIN_SAMPLES:
        return {"name": "maintenance_risk", "status": "skipped", "reason": "not_enough_samples", "samples": len(rows)}
    df = pd.DataFrame(rows).fillna(0.0)
    feature_names = list(df.columns)
    X = df.values
    y = np.array(targets, dtype=int)
    if len(set(y)) < 2:
        return {"name": "maintenance_risk", "status": "skipped", "reason": "insufficient_class_variance", "samples": len(rows)}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # XGBoost Classifier
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    metrics = {"accuracy": float(accuracy_score(y_test, preds))}
    artifact = _save_model(ml_db, "maintenance_risk", model, feature_names, metrics)
    return {"name": "maintenance_risk", "status": "trained", "version": artifact.version, "metrics": metrics, "samples": len(rows)}

def predict_safety_score(features: Dict, ml_db: Session) -> Optional[float]:
    model, feature_names, _ = load_latest_model(ml_db, "safety_score")
    if not model or not features:
        return None
    row = [features.get(name, 0.0) for name in feature_names]
    return float(model.predict([row])[0])

def predict_fuel_efficiency(features: Dict, ml_db: Session) -> Optional[float]:
    model, feature_names, _ = load_latest_model(ml_db, "fuel_efficiency")
    if not model or not features:
        return None
    row = [features.get(name, 0.0) for name in feature_names]
    return float(model.predict([row])[0])

def predict_violations_risk(features: Dict, ml_db: Session) -> Optional[float]:
    model, feature_names, _ = load_latest_model(ml_db, "violations_risk")
    if not model or not features:
        return None
    row = [features.get(name, 0.0) for name in feature_names]
    proba = model.predict_proba([row])
    if proba.shape[1] < 2:
        return None
    return float(proba[0][1])

def predict_maintenance_risk(features: Dict, ml_db: Session) -> Optional[float]:
    model, feature_names, _ = load_latest_model(ml_db, "maintenance_risk")
    if not model or not features:
        return None
    row = [features.get(name, 0.0) for name in feature_names]
    proba = model.predict_proba([row])
    if proba.shape[1] < 2:
        return None
    return float(proba[0][1])
