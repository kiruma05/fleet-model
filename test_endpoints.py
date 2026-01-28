from fastapi.testclient import TestClient
from api import app
from mock_data import generate_mock_data
from db_utils import IOTSessionLocal, FleetSessionLocal
from models import Trip, IOTTelemetry
import json

client = TestClient(app)

def test_ml_endpoints():
    # 1. Generate fresh mock data
    print("\nSetting up test data...")
    trip_id = generate_mock_data()
    
    fleet_db = FleetSessionLocal()
    trip = fleet_db.query(Trip).filter(Trip.id == trip_id).first()
    driver_id = trip.driver_id
    vehicle_id = trip.vehicle_id
    fleet_db.close()
    
    print(f"Testing endpoints for Driver: {driver_id}, Vehicle: {vehicle_id}")

    # 2. Test Root
    response = client.get("/")
    assert response.status_code == 200
    print("✓ Root endpoint working")

    # 3. Test Trip Safety Score (using IOT data)
    from safety_score import calculate_trip_features, compute_safety_score
    iot_db = IOTSessionLocal()
    fleet_db = FleetSessionLocal()
    
    features = calculate_trip_features(iot_db, fleet_db, trip_id)
    score = compute_safety_score(features)
    
    # Update DB for test (Supabase)
    db_trip = fleet_db.query(Trip).filter(Trip.id == trip_id).first()
    db_trip.safety_score = score
    fleet_db.commit()
    
    fleet_db.close()
    iot_db.close()
    
    response = client.get(f"/api/v1/trips/{trip_id}/safety-score")
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    print(f"✓ Trip Safety Score: {data['score']}")

    # 4. Test Fuel Efficiency
    response = client.get(f"/api/v1/analytics/fuel/trip/{trip_id}")
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Trip Fuel Efficiency: {data['efficiency_kpl']}")

    # 5. Test Violations
    from violations import detect_violations_from_iot
    iot_db = IOTSessionLocal()
    fleet_db = FleetSessionLocal()
    
    vols = detect_violations_from_iot(iot_db, fleet_db, trip_id)
    for v in vols:
        v.driver_id = driver_id
        fleet_db.add(v)
    fleet_db.commit()
    
    iot_db.close()
    fleet_db.close()

    response = client.get(f"/api/v1/analytics/violations/trip/{trip_id}")
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Violations detected for trip: {data['count']}")

    # 6. Test Maintenance Predictions
    from maintenance import predict_maintenance
    iot_db = IOTSessionLocal()
    fleet_db = FleetSessionLocal()
    
    preds = predict_maintenance(iot_db, fleet_db, vehicle_id)
    for p in preds:
        fleet_db.add(p)
    fleet_db.commit()
    
    iot_db.close()
    fleet_db.close()

    response = client.get(f"/api/v1/analytics/maintenance/predict/{vehicle_id}")
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Maintenance alerts: {len(data['predictions'])}")

    print("\nAll ML API tests passed! Ready for production.")

if __name__ == "__main__":
    try:
        test_ml_endpoints()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
