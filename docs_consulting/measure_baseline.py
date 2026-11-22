import time
import statistics
from pathlib import Path
import duckdb

def run_dmaic_measure_audit():
    print("=" * 65)
    print("SIX SIGMA DMAIC - PHASE 2: AUTOMATED BASELINE MEASUREMENT (MSA)")
    print("=" * 65)

    # 1. Automatically find any .duckdb or .db file in the project
    project_root = Path(__file__).resolve().parent.parent
    db_files = list(project_root.rglob("*.duckdb")) + list(project_root.rglob("*.db"))

    if not db_files:
        print(f"\n[!] No DuckDB file found inside: {project_root}")
        print("    Tip: Run your sync_telemetry_to_duckdb.py script first if the DB is not generated locally.")
        return

    db_path = db_files[0]
    db_size_mb = db_path.stat().st_size / (1024 * 1024)
    print(f"\n[+] Found Database: {db_path.relative_to(project_root)} ({db_size_mb:.2f} MB)")

    # 2. Connect to DuckDB in read-only mode
    con = duckdb.connect(str(db_path), read_only=True)
    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    if not tables:
        print("[!] Database connected, but no tables were found.")
        return

    print(f"[+] Found {len(tables)} Table(s): {', '.join(tables)}\n")
    print("-" * 65)
    print(f"{'Table Name':<28} | {'Row Count':<12} | {'Avg Query Time (ms)':<20}")
    print("-" * 65)

    # 3. Run 10 timing trials per table to calculate Mean and Std Dev (Six Sigma repeatability)
    all_latencies = []
    for table in tables:
        row_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        trial_times_ms = []
        for _ in range(10):
            start = time.perf_counter()
            # Simulate an analytical aggregation query
            con.execute(f"SELECT * FROM {table} LIMIT 5000").fetchall()
            elapsed_ms = (time.perf_counter() - start) * 1000
            trial_times_ms.append(elapsed_ms)

        mean_ms = statistics.mean(trial_times_ms)
        std_ms = statistics.stdev(trial_times_ms)
        all_latencies.append(mean_ms)
        print(f"{table:<28} | {row_count:<12,d} | {mean_ms:>6.2f} ms (±{std_ms:.2f} ms)")

    con.close()
    print("-" * 65)
    print(f"[*] Overall Mean DuckDB Query Latency (X1): {statistics.mean(all_latencies):.2f} ms")
    print("=" * 65)

if __name__ == "__main__":
    run_dmaic_measure_audit()