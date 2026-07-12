"""Run the bust-out investigation queries and print each result.

Uses data/panel.csv if present, else writes the mock panel to a temporary CSV.

    python scripts/run_investigation.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from bustout import investigation, panel_data


def feed_path() -> tuple[str, bool]:
    real = Path("data/panel.csv")
    if real.exists():
        return str(real), False
    tmp = Path(tempfile.mkdtemp())
    return str(panel_data.write_mock_panel(tmp, seed=7)), True


def main() -> None:
    path, is_mock = feed_path()
    print(f"Investigation over the {'mock panel' if is_mock else 'panel'}: {path}\n")
    for name, (desc, df) in investigation.run(path).items():
        print("=" * 80)
        print(f"{name}  --  {desc}")
        print("=" * 80)
        if df.empty:
            print("(no rows)\n")
            continue
        print(df.head(15).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
