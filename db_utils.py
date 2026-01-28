import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from models import Base

load_dotenv()

# Database URLs from environment
IOT_DATABASE_URL = os.getenv("IOT_DATABASE_URL", "sqlite:///./iot_vps.db")
FLEET_DATABASE_URL = os.getenv("FLEET_DATABASE_URL", "sqlite:///./fleet_supabase.db")
ML_DATABASE_URL = os.getenv("ML_DATABASE_URL", "sqlite:///./fleet_model.db")

# Engines
iot_engine = create_engine(
    IOT_DATABASE_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in IOT_DATABASE_URL else {}
)
fleet_engine = create_engine(
    FLEET_DATABASE_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in FLEET_DATABASE_URL else {}
)
ml_engine = create_engine(
    ML_DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in ML_DATABASE_URL else {}
)

# Sessions
IOTSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=iot_engine)
FleetSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=fleet_engine)
MLSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=ml_engine)

def init_db():
    """Initialization check (Schema creation for local ML DB)"""
    # Create tables in the local ML database
    Base.metadata.create_all(bind=ml_engine)


def get_iot_db():
    db = IOTSessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_fleet_db():
    db = FleetSessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_ml_db():
    db = MLSessionLocal()
    try:
        yield db
    finally:
        db.close()

# Legacy alias for backward compatibility or default usage
get_db = get_fleet_db

