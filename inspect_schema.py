from sqlalchemy import create_engine, inspect
from db_utils import FLEET_DATABASE_URL

engine = create_engine(FLEET_DATABASE_URL)
inspector = inspect(engine)

print("Columns in 'trips' table:")
columns = inspector.get_columns('trips')
for column in columns:
    print(f"- {column['name']} ({column['type']})")
