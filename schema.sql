-- Telemetry Data Table
CREATE TABLE telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    speed REAL,
    fuel_level REAL,
    engine_temp REAL,
    rpm INTEGER,
    latitude REAL,
    longitude REAL,
    metadata JSON
);

-- Vehicle Events Table
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    latitude REAL,
    longitude REAL,
    severity INTEGER,
    metadata JSON
);

-- Trip Summary Table
CREATE TABLE trips (
    id TEXT PRIMARY KEY,
    driver_id TEXT NOT NULL,
    vehicle_id TEXT NOT NULL,
    start_time DATETIME,
    end_time DATETIME,
    distance REAL,
    avg_speed REAL,
    max_speed REAL,
    fuel_consumed REAL,
    safety_score REAL,
    metadata JSON
);

-- Violations Table
CREATE TABLE violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trip_id TEXT REFERENCES trips(id),
    driver_id TEXT NOT NULL,
    vehicle_id TEXT NOT NULL,
    violation_type TEXT NOT NULL,
    severity INTEGER,
    timestamp DATETIME,
    latitude REAL,
    longitude REAL,
    metadata JSON
);

-- Maintenance Predictions Table
CREATE TABLE maintenance_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id TEXT NOT NULL,
    prediction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    component TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence REAL,
    predicted_km_to_failure INTEGER,
    recommended_action TEXT,
    indicators JSON
);

-- Feature Cache Table
CREATE TABLE feature_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL, -- 'driver', 'vehicle', 'trip'
    entity_id TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    feature_value REAL,
    window_period TEXT NOT NULL, -- '7d', '30d', '90d'
    computed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_type, entity_id, feature_name, window_period)
);
