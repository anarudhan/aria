"""Create or update repository labels from .github/labels.yml.

Usage: python .github/scripts/sync_labels.py [--dry-run]
Needs the GitHub CLI (gh) and a token in GH_TOKEN. Never deletes labels.
"""
import subprocess
import sys

import yaml

LABELS_FILE = ".github/labels.yml"


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    with open(LABELS_FILE, encoding="utf-8") as handle:
        labels = yaml.safe_load(handle)

    names = set()
    for label in labels:
        name = label["name"]
        if name in names:
            print(f"Duplicate label in {LABELS_FILE}: {name}")
            return 1
        names.add(name)
        if len(str(label["color"])) != 6:
            print(f"Bad colour for {name}: {label['color']}")
            return 1

    for label in labels:
        command = [
            "gh", "label", "create", label["name"],
            "--color", str(label["color"]),
            "--description", label.get("description", ""),
            "--force",
        ]
        print(("DRY RUN " if dry_run else "") + "sync " + label["name"])
        if not dry_run:
            subprocess.run(command, check=True)
    print(f"{len(labels)} labels in sync")
    return 0


if __name__ == "__main__":
    sys.exit(main())
