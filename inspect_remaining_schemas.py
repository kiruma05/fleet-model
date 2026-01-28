from sqlalchemy import create_engine, inspect
from db_utils import FLEET_DATABASE_URL

engine = create_engine(FLEET_DATABASE_URL)
inspector = inspect(engine)

for table_name in ['violations', 'maintenance_predictions', 'vehicles', 'drivers']:
    print(f"\nColumns in '{table_name}' table:")
    try:
        columns = inspector.get_columns(table_name)
        for column in columns:
            print(f"- {column['name']} ({column['type']})")
    except Exception as e:
        print(f"Table '{table_name}' not found or error: {e}")
