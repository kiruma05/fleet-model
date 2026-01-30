from app.database.db_utils import FleetSessionLocal
from app.database.models import Trip
import uuid

vehicle_id = "6a5643f3-61b8-49ca-9260-e5a4221cb607"

db = FleetSessionLocal()
try:
    # Try searching as string and as UUID
    trips = db.query(Trip).filter(Trip.vehicle_id == vehicle_id).all()
    print(f"Total trips found for vehicle {vehicle_id}: {len(trips)}")
    for t in trips:
        print(f"Trip ID: {t.id}, Score: {t.safety_score}")
except Exception as e:
    print(f"Error querying Supabase: {e}")
finally:
    db.close()
