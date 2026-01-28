import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

class FleetCoteDataCollector:
    def __init__(self, api_key=None):
        self.base_url = os.getenv("API_BASE_URL", "https://coordinates.fleetcotelematics.com")
        self.api_key = api_key or os.getenv("FLEET_COTE_API_KEY")
        self.headers = {
            'Authorization': f'Bearer {self.api_key}' if self.api_key else None,
            'Content-Type': 'application/json'
        }
    
    def fetch_latest_telemetry(self, vehicle_id):
        """Fetch latest telemetry data for a vehicle"""
        url = f"{self.base_url}/api/analytics/telemetry/{vehicle_id}/latest"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def fetch_vehicle_events(self, event_type=None, start_date=None, end_date=None):
        """Fetch vehicle events, optionally filtered by type"""
        if event_type:
            url = f"{self.base_url}/api/events/by-type"
            params = {'type': event_type, 'start': start_date, 'end': end_date}
        else:
            url = f"{self.base_url}/api/events"
            params = {'start': start_date, 'end': end_date}
        
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()
    
    def fetch_trip_summary(self, trip_id):
        """Fetch trip summary with metrics"""
        url = f"{self.base_url}/api/trips/{trip_id}/summary"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def fetch_trip_locations(self, trip_id):
        """Fetch all location points for a trip"""
        url = f"{self.base_url}/api/trips/{trip_id}/locations"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def fetch_location_history(self, vehicle_id, start_date, end_date):
        """Fetch location history for a vehicle"""
        url = f"{self.base_url}/api/analytics/location/history"
        params = {
            'vehicleId': vehicle_id,
            'start': start_date,
            'end': end_date
        }
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()
    
    def fetch_all_vehicles_latest(self):
        """Get all vehicles with their latest locations"""
        url = f"{self.base_url}/api/analytics/vehicles"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def fetch_latest_location(self, vehicle_id):
        """Get latest location for a specific vehicle"""
        url = f"{self.base_url}/api/analytics/location/{vehicle_id}/latest"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
