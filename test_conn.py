import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

FLEET_DATABASE_URL = os.getenv("FLEET_DATABASE_URL")
print(f"Testing connection to: {FLEET_DATABASE_URL.split('@')[1]}")

try:
    engine = create_engine(FLEET_DATABASE_URL)
    with engine.connect() as conn:
        print("✓ Successfully connected to Supabase!")
except Exception as e:
    print(f"✗ Connection failed: {e}")
