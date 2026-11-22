from pathlib import Path
import duckdb

def run_dmaic_analyze_audit():
    print("=" * 70)
    print("SIX SIGMA DMAIC - PHASE 3: CODE & SCHEMA ARCHITECTURE AUDIT")
    print("=" * 70)

    project_root = Path(__file__).resolve().parent.parent
    ignore_dirs = {"venv", ".venv", ".git", "__pycache__", "docs_consulting", ".streamlit"}

    # 1. Inventory all Python source files in the repository
    print("\n[1] EXISTING PYTHON MODULES IN REPOSITORY:")
    print("-" * 70)
    py_files = sorted([
        p for p in project_root.rglob("*.py")
        if not any(part in ignore_dirs for part in p.parts)
    ])
    for py_file in py_files:
        line_count = len(py_file.read_text(encoding="utf-8", errors="ignore").splitlines())
        print(f"  - {str(py_file.relative_to(project_root)):<45} ({line_count} lines)")

    # 2. Check for any saved ML model artifacts
    model_exts = ("*.pkl", "*.joblib", "*.ubj", "*.json", "*.onnx")
    model_files = []
    for ext in model_exts:
        model_files.extend([
            p for p in project_root.rglob(ext)
            if not any(part in ignore_dirs for part in p.parts)
        ])
    print("\n[2] SAVED MODEL / CONFIG ARTIFACTS:")
    print("-" * 70)
    if model_files:
        for mf in sorted(model_files):
            print(f"  - {mf.relative_to(project_root)}")
    else:
        print("  - None found on disk (models are likely trained dynamically in Python).")

    # 3. Inspect exact column names and types for all 8 DuckDB tables
    db_path = project_root / "data" / "f1_lakehouse.duckdb"
    if not db_path.exists():
        print(f"\n[!] Database not found at {db_path}")
        return

    print("\n[3] DUCKDB TABLE SCHEMAS (FOR BIGQUERY & FASTAPI MAPPING):")
    print("-" * 70)
    con = duckdb.connect(str(db_path), read_only=True)
    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    for table in tables:
        cols = con.execute(f"DESCRIBE {table}").fetchall()
        col_summary = ", ".join([f"{c[0]} ({c[1]})" for c in cols])
        print(f"\n* {table} [{len(cols)} columns]:")
        print(f"  {col_summary}")

    con.close()
    print("\n" + "=" * 70)

if __name__ == "__main__":
    run_dmaic_analyze_audit()