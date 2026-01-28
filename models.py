from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean, LargeBinary
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.types import Uuid as GenericUuid

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

class Telemetry(Base):
    __tablename__ = 'telemetry'
    id = Column(Integer, primary_key=True)
    vehicle_id = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    speed = Column(Float)
    fuel_level = Column(Float)
    engine_temp = Column(Float)
    rpm = Column(Integer)
    latitude = Column(Float)
    longitude = Column(Float)
    metadata_json = Column(JSON)

class IOTTelemetry(Base):
    __tablename__ = 'vehicle_locations'
    
    # We use PG UUID to ensure compatibility with existing DB columns
    time = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    vehicle_id = Column(PgUUID(as_uuid=True), primary_key=True, nullable=False) 
    
    source_type = Column(String)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    speed = Column(Float)
    heading = Column(Float)
    
    # JSONB Fields
    vehicle_telemetry = Column(JSON)
    fuel_data = Column(JSON)
    driver_behavior = Column(JSON)
    power_data = Column(JSON)
    io_data = Column(JSON)
    motion_events = Column(JSON)
    device_status = Column(JSON)

    # Derived Properties from JSON fields (Handling missing columns)
    
    @property
    def fuel_level(self):
        if self.fuel_data and isinstance(self.fuel_data, dict):
            return self.fuel_data.get('fuelLevel') or self.fuel_data.get('fuel_level')
        return None

    @property
    def battery_voltage(self):
        if self.power_data and isinstance(self.power_data, dict):
            return self.power_data.get('batteryVoltage') or self.power_data.get('battery_voltage')
        if self.vehicle_telemetry and isinstance(self.vehicle_telemetry, dict):
             return self.vehicle_telemetry.get('battery_voltage')
        return None

    @property
    def ignition(self):
        if self.io_data and isinstance(self.io_data, dict):
            return self.io_data.get('ignition')
        return False

    @property
    def harsh_braking(self):
        if self.driver_behavior and isinstance(self.driver_behavior, dict):
            return self.driver_behavior.get('harsh_braking') or self.driver_behavior.get('harshBraking')
        return False

    @property
    def harsh_acceleration(self):
        if self.driver_behavior and isinstance(self.driver_behavior, dict):
            return self.driver_behavior.get('harsh_acceleration') or self.driver_behavior.get('harshAcceleration')
        return False

    @property
    def crash_detected(self):
        if self.motion_events and isinstance(self.motion_events, dict):
            return self.motion_events.get('crash_detected')
        return False

    # Constants/Defaults for missing columns
    engine_status = None
    mobile_battery = None
    network_type = None
    is_moving = None
    
    trip_id = Column(PgUUID(as_uuid=True), nullable=True)

    # Alias for compatibility with code expecting 'timestamp'
    @property
    def timestamp(self):
        return self.time


class Event(Base):
    __tablename__ = 'events'
    id = Column(Integer, primary_key=True)
    vehicle_id = Column(String, nullable=False)
    event_type = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    latitude = Column(Float)
    longitude = Column(Float)
    severity = Column(Integer)
    metadata_json = Column(JSON)

class Trip(Base):
    __tablename__ = 'trips'
    id = Column(PgUUID(as_uuid=True), primary_key=True)
    vehicle_id = Column(PgUUID(as_uuid=True), nullable=False, index=True)
    main_driver_id = Column(PgUUID(as_uuid=True), index=True)
    substitute_driver_id = Column(PgUUID(as_uuid=True))
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    distance_km = Column(String) # For Supabase compatibility
    fuel_used = Column(String) # For Supabase compatibility
    status = Column(String)
    
    # Supabase specific fields
    start_location = Column(String)
    end_location = Column(String)
    notes = Column(String)
    company_id = Column(PgUUID(as_uuid=True))
    duration_minutes = Column(String)

class Vehicle(Base):
    __tablename__ = 'vehicles'
    id = Column(PgUUID(as_uuid=True), primary_key=True)
    registration_number = Column(String, name='registrationNumber', nullable=False) # name matching Supabase
    model = Column(String)
    manufacturer = Column(String)
    vin = Column(String)
    
class Driver(Base):
    __tablename__ = 'drivers'
    id = Column(PgUUID(as_uuid=True), primary_key=True)
    first_name = Column(String, name='firstName')
    last_name = Column(String, name='lastName')
    phone_number = Column(String, name='phoneNumber')
    license_number = Column(String, name='licenseNumber')

class Violation(Base):
    __tablename__ = 'violations'
    id = Column(Integer, primary_key=True)
    trip_id = Column(PgUUID(as_uuid=True))
    driver_id = Column(PgUUID(as_uuid=True), nullable=False)
    vehicle_id = Column(PgUUID(as_uuid=True), nullable=False)
    violation_type = Column(String, nullable=False)
    severity = Column(Integer)
    timestamp = Column(DateTime)
    latitude = Column(Float)
    longitude = Column(Float)
    metadata_json = Column(JSON)

class MaintenancePrediction(Base):
    __tablename__ = 'maintenance_predictions'
    id = Column(Integer, primary_key=True)
    vehicle_id = Column(PgUUID(as_uuid=True), nullable=False)
    prediction_date = Column(DateTime, default=datetime.datetime.utcnow)
    component = Column(String, nullable=False)
    status = Column(String, nullable=False)
    confidence = Column(Float)
    predicted_km_to_failure = Column(Integer)
    recommended_action = Column(String)
    indicators = Column(JSON)

# --- Local ML Metadata Models ---

class DriverSafetyScore(Base):
    __tablename__ = 'driver_safety_scores'
    id = Column(Integer, primary_key=True)
    driver_id = Column(PgUUID(as_uuid=True), nullable=False, index=True)
    score = Column(Float, nullable=False)
    trip_count = Column(Integer, default=0)
    last_updated = Column(DateTime, default=datetime.datetime.utcnow)
    metadata_json = Column(JSON)



class FeatureCache(Base):
    __tablename__ = 'feature_cache'
    id = Column(Integer, primary_key=True)
    entity_type = Column(String, nullable=False)
    entity_id = Column(String, nullable=False)
    feature_name = Column(String, nullable=False)
    feature_value = Column(Float)
    window_period = Column(String, nullable=False)
    computed_at = Column(DateTime, default=datetime.datetime.utcnow)

class ModelArtifact(Base):
    __tablename__ = 'ml_models'
    id = Column(Integer, primary_key=True)
    name = Column(String, index=True, nullable=False)
    version = Column(Integer, default=1)
    trained_at = Column(DateTime, default=datetime.datetime.utcnow)
    metrics = Column(JSON)
    feature_names = Column(JSON)
    model_blob = Column(LargeBinary, nullable=False)
