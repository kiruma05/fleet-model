# Fleet ML POC - Driver Safety & Predictive Maintenance

This Proof of Concept (POC) implements machine learning models for fleet management using the Fleet Cote Telematics API.

## Features

1.  **Driver Safety Scoring (0-100)**: Real-time scoring based on speeding, harsh events, and trip distance.
2.  **Violations Detection**: Automated detection and aggregation of speeding and harsh driving behaviors.
3.  **Predictive Maintenance**: Trend-based failure prediction for engine systems, batteries, and tires.
4.  **Automated Data Collection**: Background scheduler for fetching telemetry and events.
5.  **ML API Service**: FastAPI endpoints for easy integration with your dashboard.

## Technology Stack

- **Language**: Python 3.9+
- **ML/Analytics**: Pandas, NumPy, XGBoost, scikit-learn
- **API**: FastAPI + Uvicorn
- **Database**: SQLite (SQLAlchemy)
- **Scheduler**: APScheduler

## Quick Start

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure `.env`**:
    Copy `.env.example` to `.env` and add your `FLEET_COTE_API_KEY`.

3.  **Run the system**:
    ```bash
    # Terminal 1: Background Collector
    python main.py
    
    # Terminal 2: API Service
    uvicorn api:app --port 8000
    ```

## Documentation

For a detailed implementation guide and API reference, see the [Walkthrough](file:///Users/frankkiruma/.gemini/antigravity/brain/d5165473-da21-4598-8c53-63ce2e7f98bf/walkthrough.md).
