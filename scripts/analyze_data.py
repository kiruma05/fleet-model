import sqlite3
import pandas as pd
import json

db_path = 'fleet_model.db'

def analyze_db():
    try:
        conn = sqlite3.connect(db_path)
        
        # 1. Basic Counts
        count = pd.read_sql("SELECT COUNT(*) as count FROM vehicle_telemetry", conn).iloc[0]['count']
        print(f"Total Rows in vehicle_telemetry: {count}")

        if count == 0:
            print("No data found.")
            return

        # 2. Sample Data
        df = pd.read_sql("SELECT * FROM vehicle_telemetry LIMIT 100", conn)
        print("\n--- Columns ---")
        print(df.columns.tolist())
        
        print("\n--- Sample Data (Head) ---")
        print(df[['timestamp', 'speed', 'battery_voltage', 'fuel_level']].head())

        # 3. Distributions
        print("\n--- Statistics ---")
        stats_query = """
        SELECT 
            MIN(speed) as min_speed, MAX(speed) as max_speed, AVG(speed) as avg_speed,
            MIN(battery_voltage) as min_volt, MAX(battery_voltage) as max_volt, AVG(battery_voltage) as avg_volt,
            SUM(harsh_braking) as total_harsh_braking,
            SUM(harsh_acceleration) as total_harsh_accel,
            SUM(crash_detected) as total_crashes
        FROM vehicle_telemetry
        """
        stats = pd.read_sql(stats_query, conn)
        print(stats.to_string())

        # 4. JSON Inspection
        print("\n--- JSON Fields Inspection ---")
        # Check fuel_data, io_data, metadata_json
        for col in ['fuel_data', 'io_data', 'metadata_json']:
            if col in df.columns:
                print(f"\nExample {col}:")
                sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if sample:
                    try:
                        # SQLite JSON might be stored as string
                        if isinstance(sample, str):
                            print(json.dumps(json.loads(sample), indent=2))
                        else:
                            print(sample)
                    except Exception as e:
                        print(f"Error parsing JSON: {e} | Raw: {sample}")
                else:
                    print("clean/empty")

    except Exception as e:
        print(f"Error analyzing DB: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    analyze_db()
