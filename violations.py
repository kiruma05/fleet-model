from models import IOTTelemetry, Violation, Trip
from sqlalchemy.orm import Session
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def detect_violations_from_iot(iot_db: Session, fleet_db: Session, trip_id: str, driver_id: Optional[str] = None):
    """
    Detect violations for a specific trip using raw IOT sensor flags.
    """
    try:
        import uuid
        if isinstance(trip_id, str):
            trip_uuid = uuid.UUID(trip_id)
        else:
            trip_uuid = trip_id
    except ValueError:
        return []

    telemetry = iot_db.query(IOTTelemetry).filter(
        IOTTelemetry.trip_id == trip_uuid
    ).all()
    
    if not telemetry:
        return []
        
    violations_detected = []
    speed_limit = 100
    
    for tel in telemetry:
        # 1. Speeding
        if tel.speed and tel.speed > speed_limit:
            diff = tel.speed - speed_limit
            if diff > 30:
                severity = 3
            elif diff > 15:
                severity = 2
            else:
                severity = 1
                
            violations_detected.append(Violation(
                trip_id=trip_id,
                driver_id=driver_id,
                vehicle_id=tel.vehicle_id,
                violation_type='SPEEDING',
                severity=severity,
                timestamp=tel.timestamp,
                latitude=tel.latitude,
                longitude=tel.longitude,
                metadata_json={'actual_speed': tel.speed, 'limit': speed_limit}
            ))
            
                
        # 2. Harsh Behavior (Direct from IOT flags)
        if tel.harsh_braking:
            violations_detected.append(Violation(
                trip_id=trip_id,
                driver_id=driver_id,
                vehicle_id=tel.vehicle_id,
                violation_type='HARSH_BRAKING',
                severity=2,  # Harsh braking is moderately severe
                timestamp=tel.timestamp,
                latitude=tel.latitude,
                longitude=tel.longitude
            ))

        if tel.harsh_acceleration:
            violations_detected.append(Violation(
                trip_id=trip_id,
                driver_id=driver_id,
                vehicle_id=tel.vehicle_id,
                violation_type='HARSH_ACCELERATION',
                severity=2,
                timestamp=tel.timestamp,
                latitude=tel.latitude,
                longitude=tel.longitude
            ))
            
        if tel.crash_detected:
            violations_detected.append(Violation(
                trip_id=trip_id,
                driver_id=driver_id,
                vehicle_id=tel.vehicle_id,
                violation_type='CRASH_INCIDENT',
                severity=3,
                timestamp=tel.timestamp,
                latitude=tel.latitude,
                longitude=tel.longitude
            ))
            
    return violations_detected

def aggregate_violations(db: Session, entity_type: str, entity_id: str):
    """
    Aggregate violations for a driver or vehicle from Management DB.
    """
    query = db.query(Violation)
    if entity_type == 'driver':
        query = query.filter(Violation.driver_id == entity_id)
    elif entity_type == 'vehicle':
        query = query.filter(Violation.vehicle_id == entity_id)
        
    vols = query.all()
    
    summary = {
        'total_count': len(vols),
        'by_type': {},
        'by_severity': {1: 0, 2: 0, 3: 0}
    }
    
    for v in vols:
        summary['by_type'][v.violation_type] = summary['by_type'].get(v.violation_type, 0) + 1
        summary['by_severity'][v.severity] = summary['by_severity'].get(v.severity, 0) + 1
        
    return summary
