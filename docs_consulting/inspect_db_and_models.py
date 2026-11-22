import sys
from pathlib import Path
import duckdb

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

db_path = project_root / "data" / "f1_lakehouse.duckdb"
con = duckdb.connect(str(db_path), read_only=True)

print("=== DUCKDB TABLES ===")
tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
print(tables)

for t in tables:
    cols = [r[0] for r in con.execute(f"DESCRIBE {t}").fetchall()]
    print(f"\n[{t}] -> {cols}")

print("\n=== SAMPLE TRACKS (2025) ===")
for candidate in ["dim_results", "clean_results", "races", "results"]:
    if candidate in tables:
        df = con.execute(f"SELECT DISTINCT Year, RoundNumber, EventName FROM {candidate} WHERE Year = 2025 ORDER BY RoundNumber LIMIT 5").df()
        print(df)
        break
con.close()