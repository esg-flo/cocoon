#!/usr/bin/env python3
"""
Calculate semantic version based on commit message tags.
Supports [MAJOR], [MINOR], [FIX], and [PATCH] tags.
"""

import re
import sys
from typing import Tuple


def parse_version(version_string: str) -> Tuple[int, int, int]:
    """Parse a version string like 'v1.2.3' into (major, minor, patch)."""
    version_string = version_string.lstrip("v")
    parts = version_string.split(".")

    major = int(parts[0]) if len(parts) > 0 else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    patch = int(parts[2]) if len(parts) > 2 else 0

    return major, minor, patch


def determine_increment_type(commit_messages: str) -> str:
    """
    Determine version increment type based on commit message tags.
    Priority: MAJOR > MINOR > PATCH/FIX
    """
    if re.search(r"\[MAJOR\]", commit_messages, re.IGNORECASE):
        return "MAJOR"
    elif re.search(r"\[MINOR\]", commit_messages, re.IGNORECASE):
        return "MINOR"
    elif re.search(r"\[(FIX|PATCH)\]", commit_messages, re.IGNORECASE):
        return "PATCH"
    else:
        return "NONE"


def calculate_new_version(current_version: str, increment_type: str) -> str:
    """Calculate the new version based on increment type."""
    major, minor, patch = parse_version(current_version)

    if increment_type == "MAJOR":
        return f"v{major + 1}.0.0"
    elif increment_type == "MINOR":
        return f"v{major}.{minor + 1}.0"
    else:  # PATCH or NONE (default to patch)
        return f"v{major}.{minor}.{patch + 1}"


def has_version_tag(commit_messages: str) -> bool:
    """Check if any commit has a version increment tag."""
    return bool(
        re.search(r"\[(MAJOR|MINOR|FIX|PATCH)\]", commit_messages, re.IGNORECASE)
    )


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: version_calculator.py <current_version> <commit_messages>",
            file=sys.stderr,
        )
        sys.exit(1)

    current_version = sys.argv[1]
    commit_messages = sys.argv[2]

    # Determine increment type
    increment_type = determine_increment_type(commit_messages)

    # If no tags found, default to PATCH
    if increment_type == "NONE":
        increment_type = "PATCH"
        print("No increment designation found, defaulting to [FIX]")
    else:
        print(f"Found [{increment_type}] tag in commits")

    # Calculate new version
    new_version = calculate_new_version(current_version, increment_type)

    # Check if we should add a note to release notes
    needs_note = not has_version_tag(commit_messages)

    # Output results
    print(f"increment_type={increment_type}")
    print(f"new_version={new_version}")
    print(f"needs_note={'true' if needs_note else 'false'}")


if __name__ == "__main__":
    main()
