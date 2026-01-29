from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from db_utils import get_iot_db, get_fleet_db, get_ml_db, init_db
from models import Trip, Violation, DriverSafetyScore, IOTTelemetry
import safety_score
import violations
import maintenance
import fuel_efficiency
import ml_pipeline
import datetime
import traceback
import sys
import uuid
try:
    import numpy as np
except Exception:
    np = None

app = FastAPI(title="Fleet ML Services POC")

@app.on_event("startup")
def ensure_ml_tables():
    init_db()

@app.middleware("http")
async def log_exceptions_middleware(request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise e

@app.get("/", summary="API Root", description="Verifies that the Fleet ML API is running and accessible.")
def read_root():
    return {"message": "Fleet ML Services POC API is running"}

# --- Shared Helpers ---

def _get_or_compute_driver_safety_score(driver_id: uuid.UUID, fleet_db: Session, iot_db: Session, ml_db: Session):
    # Always recompute with new logic for now to ensure freshness
    score = safety_score.calculate_driver_safety_score(fleet_db, str(driver_id))
    
    # Update or Create in ML DB
    db_score = ml_db.query(DriverSafetyScore).filter(DriverSafetyScore.driver_id == driver_id).first()
    trip_count = fleet_db.query(Trip).filter(Trip.main_driver_id == driver_id).count() 
    
    if db_score:
        db_score.score = score
        db_score.trip_count = trip_count
        db_score.last_updated = datetime.datetime.utcnow()
    else:
        db_score = DriverSafetyScore(
            driver_id=driver_id,
            score=score,
            trip_count=trip_count,
            last_updated=datetime.datetime.utcnow()
        )
        ml_db.add(db_score)
    
    ml_db.commit()
    
    return {
        "driver_id": driver_id,
        "score": score,
        "trip_count": trip_count,
        "last_updated": db_score.last_updated
    }

def _serialize_violation(v: Violation):
    return {
        "id": v.id,
        "trip_id": v.trip_id,
        "driver_id": v.driver_id,
        "vehicle_id": v.vehicle_id,
        "violation_type": v.violation_type,
        "severity": v.severity,
        "timestamp": v.timestamp.isoformat() if v.timestamp else None,
        "latitude": v.latitude,
        "longitude": v.longitude,
        "metadata": v.metadata_json
    }

def _summarize_violations(violation_list):
    summary = {
        "total_count": len(violation_list),
        "by_type": {},
        "by_severity": {1: 0, 2: 0, 3: 0}
    }
    for v in violation_list:
        summary["by_type"][v.violation_type] = summary["by_type"].get(v.violation_type, 0) + 1
        if v.severity in summary["by_severity"]:
            summary["by_severity"][v.severity] += 1
        else:
            summary["by_severity"][v.severity] = 1
    return summary

def _to_native(value):
    if np is not None:
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    # Fallback for numpy scalars/arrays when numpy isn't importable
    try:
        module_name = type(value).__module__
    except Exception:
        module_name = ""
    if module_name.startswith("numpy"):
        if hasattr(value, "item") and callable(value.item):
            return value.item()
        if hasattr(value, "tolist") and callable(value.tolist):
            return value.tolist()
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_native(v) for v in value]
    return value

# --- Consolidated Summary Endpoints (Only Public APIs) ---

@app.get("/api/v1/trips/{trip_id}/summary", 
          summary="Get Trip Safety Summary", 
          description="Analyzes a specific trip to provide a safety score, detect violations, and calculate fuel efficiency features.")
def get_trip_summary(trip_id: uuid.UUID,
                     fleet_db: Session = Depends(get_fleet_db),
                     iot_db: Session = Depends(get_iot_db),
                     ml_db: Session = Depends(get_ml_db)):
    """
    Returns a comprehensive summary of a trip, including:
    - **Safety Score**: 0-100 (100 is best).
    - **Violations**: List of speeding or harsh driving events.
    - **Risk Analysis**: AI-predicted risk of future violations.
    - **Fuel Efficiency**: Estimated KM per Liter.
    """
    trip = fleet_db.query(Trip).filter(Trip.id == trip_id).first()
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    
    features = ml_pipeline.build_trip_safety_features(iot_db, fleet_db, str(trip_id))
    if not features:
        features = ml_pipeline.build_trip_safety_features(ml_db, fleet_db, str(trip_id))
    trip_score = ml_pipeline.predict_safety_score(features, ml_db)
    if trip_score is None:
        trip_score = safety_score.compute_safety_score(features)
    violations_risk = ml_pipeline.predict_violations_risk(features, ml_db)
    driver_id = str(trip.main_driver_id) if trip.main_driver_id else None
    trip_violations = violations.detect_violations_from_iot(iot_db, fleet_db, str(trip_id), driver_id=driver_id)
    if not trip_violations:
        trip_violations = violations.detect_violations_from_iot(ml_db, fleet_db, str(trip_id), driver_id=driver_id)
    fuel_features = ml_pipeline.build_trip_fuel_features(iot_db, fleet_db, str(trip_id))
    if not fuel_features:
        fuel_features = ml_pipeline.build_trip_fuel_features(ml_db, fleet_db, str(trip_id))
    efficiency = ml_pipeline.predict_fuel_efficiency(fuel_features, ml_db)
    if efficiency is None:
        efficiency = fuel_efficiency.calculate_trip_fuel_efficiency(iot_db, fleet_db, str(trip_id))
    
    driver_score = None
    if trip.main_driver_id:
        driver_score = _get_or_compute_driver_safety_score(trip.main_driver_id, fleet_db, iot_db, ml_db)
    
    return _to_native({
        "trip_id": trip_id,
        "driver_id": trip.main_driver_id,
        "trip_safety_score": trip_score,
        "trip_safety_features": features,
        "trip_violations_risk": violations_risk,
        "trip_violations": [_serialize_violation(v) for v in trip_violations],
        "trip_violations_count": len(trip_violations),
        "trip_fuel_efficiency_kpl": efficiency,
        "driver_safety_score": driver_score
    })

@app.get("/api/v1/drivers/{driver_id}/summary", 
          summary="Get Driver Safety Profile", 
          description="Aggregates all trips for a specific driver to compute an overall safety score and violation history.")
def get_driver_summary(driver_id: uuid.UUID,
                       fleet_db: Session = Depends(get_fleet_db),
                       iot_db: Session = Depends(get_iot_db),
                       ml_db: Session = Depends(get_ml_db)):
    """
    Returns the driver's safety profile:
    - **Safety Score**: Aggregate score across all trips.
    - **Violations Summary**: Counts of violations by type and severity.
    - **Trips**: List of trips with their individual safety scores and violation counts.
    """
    driver_score = _get_or_compute_driver_safety_score(driver_id, fleet_db, iot_db, ml_db)
    trips = fleet_db.query(Trip).filter(Trip.main_driver_id == driver_id).order_by(Trip.start_time.desc()).all()
    
    trip_summaries = []
    all_violations = [] # Still needed for summary aggregation
    
    for trip in trips:
        # Calculate scores and violations for trip
        features = ml_pipeline.build_trip_safety_features(iot_db, fleet_db, str(trip.id))
        trip_score = safety_score.compute_safety_score(features)
        
        trip_violations = violations.detect_violations_from_iot(
            iot_db,
            fleet_db,
            str(trip.id),
            driver_id=str(driver_id)
        )
        if not trip_violations:
            trip_violations = violations.detect_violations_from_iot(
                ml_db,
                fleet_db,
                str(trip.id),
                driver_id=str(driver_id)
            )
        all_violations.extend(trip_violations)
        
        trip_summaries.append({
            "trip_id": trip.id,
            "start_time": trip.start_time.isoformat() if trip.start_time else None,
            "end_time": trip.end_time.isoformat() if trip.end_time else None,
            "distance_km": trip.distance_km,
            "score": trip_score,
            "violation_count": len(trip_violations)
        })
        
    summary = _summarize_violations(all_violations)
    
    return _to_native({
        "driver_id": driver_id,
        "safety_score": driver_score,
        "trips": trip_summaries,
        "violations_summary": summary
    })

@app.get("/api/v1/drivers/{driver_id}/violations", 
          summary="Get Driver Violations", 
          description="Returns a full list of violations committed by this driver.")
def get_driver_violations(driver_id: uuid.UUID,
                          fleet_db: Session = Depends(get_fleet_db),
                          iot_db: Session = Depends(get_iot_db),
                          ml_db: Session = Depends(get_ml_db)):
    """
    Returns a comprehensive list of all violations for the driver.
    """
    trips = fleet_db.query(Trip).filter(Trip.main_driver_id == driver_id).all()
    all_violations = []
    for trip in trips:
        trip_violations = violations.detect_violations_from_iot(
            iot_db,
            fleet_db,
            str(trip.id),
            driver_id=str(driver_id)
        )
        if not trip_violations:
            trip_violations = violations.detect_violations_from_iot(
                ml_db,
                fleet_db,
                str(trip.id),
                driver_id=str(driver_id)
            )
        all_violations.extend(trip_violations)
        
    return _to_native([_serialize_violation(v) for v in all_violations])

@app.get("/api/v1/trips/{trip_id}/violations", 
          summary="Get Trip Violations", 
          description="Returns a full list of violations for a specific trip.")
def get_trip_violations(trip_id: uuid.UUID,
                        fleet_db: Session = Depends(get_fleet_db),
                        iot_db: Session = Depends(get_iot_db),
                        ml_db: Session = Depends(get_ml_db)):
    """
    Returns a comprehensive list of all violations for the trip.
    """
    trip = fleet_db.query(Trip).filter(Trip.id == trip_id).first()
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
        
    driver_id = str(trip.main_driver_id) if trip.main_driver_id else None
    trip_violations = violations.detect_violations_from_iot(
        iot_db,
        fleet_db,
        str(trip_id),
        driver_id=driver_id
    )
    if not trip_violations:
        trip_violations = violations.detect_violations_from_iot(
            ml_db,
            fleet_db,
            str(trip_id),
            driver_id=driver_id
        )
        
    return _to_native([_serialize_violation(v) for v in trip_violations])

@app.get("/api/v1/analytics/maintenance/predict/{vehicle_id}", 
          summary="Predict Vehicle Maintenance", 
          description="Analyzes vehicle telemetry to predict upcoming maintenance needs and failure risks.")
def get_vehicle_maintenance_prediction(vehicle_id: uuid.UUID, 
                                     fleet_db: Session = Depends(get_fleet_db),
                                     iot_db: Session = Depends(get_iot_db),
                                     ml_db: Session = Depends(get_ml_db)):
    """
    Predictive maintenance analysis:
    - **Predictions**: List of predicted component failures (e.g., "Brake Pad Wear").
    - **Maintenance Risk Score**: 0.0 - 1.0 probability of immediate attention needed.
    """
    preds = maintenance.predict_maintenance(iot_db, fleet_db, str(vehicle_id))
    risk_features = ml_pipeline.build_vehicle_maintenance_features(iot_db, fleet_db, str(vehicle_id))
    risk_score = ml_pipeline.predict_maintenance_risk(risk_features, ml_db)
    if risk_score is None and risk_features:
        # Simple heuristic fallback for POC
        avg_v = risk_features.get("avg_battery_voltage")
        min_v = risk_features.get("min_battery_voltage")
        harsh = risk_features.get("harsh_events", 0) or 0
        total_km = risk_features.get("total_km", 0) or 0
        score = 0.1
        if avg_v is not None and avg_v < 12.2:
            score += 0.5
        if min_v is not None and min_v < 11.8:
            score += 0.2
        if harsh > 5:
            score += 0.1
        if total_km > 50000:
            score += 0.2
        risk_score = min(1.0, float(score))
    return {
        "vehicle_id": vehicle_id,
        "predictions": preds,
        "maintenance_risk_score": risk_score
    }

@app.get("/api/v1/maintenance/fleet/summary", 
          summary="Get Fleet Health Overview", 
          description="Provides a high-level summary of the maintenance status across the entire fleet.")
def get_fleet_maintenance_summary(db: Session = Depends(get_fleet_db)):
    """
    Returns:
    - Count of healthy vs. at-risk vehicles.
    - List of vehicles requiring immediate service.
    """
    return maintenance.get_fleet_health_summary(db)

@app.post("/api/v1/ml/train", 
           summary="Trigger Model Training", 
           description="Triggers the ML pipeline to retrain all models using the latest trip data.")
def train_models(use_mock: bool = False,
                 trips_limit: int = 50,
                 rows_per_trip: int = 20,
                 reset_mock: bool = False,
                 fleet_db: Session = Depends(get_fleet_db),
                 iot_db: Session = Depends(get_iot_db),
                 ml_db: Session = Depends(get_ml_db)):
    """
    **Admin Only**: Retrains the machine learning models.
    - **use_mock**: If True, generates synthetic data for training (dev mode).
    - **trips_limit**: Number of trips to process.
    - **reset_mock**: If True, wipes existing mock data before generating new data.
    """
    mock_info = None
    trips_query = fleet_db.query(Trip).order_by(Trip.start_time.desc())
    if trips_limit:
        trips_query = trips_query.limit(trips_limit)
    trips = trips_query.all()
    if not trips:
        raise HTTPException(status_code=400, detail="no trips available in fleet DB for training")
    trip_ids = [str(trip.id) for trip in trips if trip and trip.id]
    if use_mock:
        mock_info = ml_pipeline.generate_mock_telemetry(
            iot_db,
            fleet_db,
            trips_limit=trips_limit,
            rows_per_trip=rows_per_trip,
            reset=reset_mock
        )
    else:
        telemetry_exists = (
            iot_db.query(IOTTelemetry.id)
            .filter(IOTTelemetry.trip_id.in_(trip_ids))
            .first()
        )
        if not telemetry_exists:
            mock_info = ml_pipeline.generate_mock_telemetry(
                iot_db,
                fleet_db,
                trips_limit=trips_limit,
                rows_per_trip=rows_per_trip,
                reset=False
            )
        else:
            mock_info = {"status": "skipped", "reason": "telemetry_exists"}
    results = [
        ml_pipeline.train_safety_model(iot_db, fleet_db, ml_db),
        ml_pipeline.train_violations_risk_model(iot_db, fleet_db, ml_db),
        ml_pipeline.train_fuel_efficiency_model(iot_db, fleet_db, ml_db),
        ml_pipeline.train_maintenance_risk_model(iot_db, fleet_db, ml_db)
    ]
    return {"trained_at": datetime.datetime.utcnow(), "mock_info": mock_info, "results": results}
