"""
Forgiving league and season name matching, shared by every step that accepts --league / --season.

Names are compared case-insensitively, with "-", "/" and "_" treated as spaces, so:

    "la liga", "La-Liga" and "LA LIGA"        all match the folder "La Liga"
    "2015/2016", "2015-2016" and "2015_2016"  all match the folder "2015-2016"

Apostrophes inside names are kept, so "FA Women's Super League" matches its folder too.
"""
import os


def normalize_name(value: str) -> str:
    """Lower-cases a name, strips surrounding quotes and turns separators into single spaces."""
    value = value.strip().strip("'\"").lower()
    for ch in "-/_":
        value = value.replace(ch, " ")
    return " ".join(value.split())


def names_match(a: str, b: str) -> bool:
    """True if two league or season names refer to the same thing."""
    return normalize_name(a) == normalize_name(b)


def find_subfolder(parent_dir: str, wanted: str):
    """Returns the name of the sub-folder of parent_dir that matches `wanted`, or None."""
    if not os.path.isdir(parent_dir):
        return None
    for entry in sorted(os.listdir(parent_dir)):
        if os.path.isdir(os.path.join(parent_dir, entry)) and names_match(entry, wanted):
            return entry
    return None


def list_subfolders(parent_dir: str) -> list:
    """Sorted sub-folder names of parent_dir, used to show the valid choices in error messages."""
    if not os.path.isdir(parent_dir):
        return []
    return sorted(e for e in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, e)))
