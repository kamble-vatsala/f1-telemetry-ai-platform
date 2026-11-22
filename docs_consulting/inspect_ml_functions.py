import ast
from pathlib import Path

def inspect_project_functions():
    print("=" * 70)
    print("WORKSTREAM 2: INSPECTING EXISTING ANALYTICS & ML FUNCTIONS")
    print("=" * 70)

    project_root = Path(__file__).resolve().parent.parent
    target_files = [
        "src/analytics/tyre_analytics.py",
        "src/analytics/telemetry_engine.py",
        "src/analytics/driver_dna.py",
        "src/analytics/race_analytics.py",
        "src/models/predictor.py",
        "src/models/ml_supervised_suite.py",
        "src/models/ml_unsupervised_bayesian_sim.py",
    ]

    for rel_path in target_files:
        file_path = project_root / rel_path
        if not file_path.exists():
            continue

        print(f"\n[FILE] {rel_path}")
        print("-" * 70)
        tree = ast.parse(file_path.read_text(encoding="utf-8", errors="ignore"))

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                args = [a.arg for a in node.args.args]
                doc = ast.get_docstring(node)
                first_line = doc.splitlines()[0] if doc else "No docstring"
                print(f"  def {node.name}({', '.join(args)})")
                print(f"      -> {first_line}")
            elif isinstance(node, ast.ClassDef):
                print(f"  class {node.name}:")
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        args = [a.arg for a in item.args.args]
                        print(f"      def {item.name}({', '.join(args)})")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    inspect_project_functions()