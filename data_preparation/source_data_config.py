"""Locations of the raw clinical exports used by the data-preparation scripts.

These files contain protected health information and are not distributed with
this repository.  Point VF_SOURCE_ROOT at the directory that holds them
(see config/env.example.sh).
"""
import os
from pathlib import Path

SOURCE_ROOT = Path(os.environ.get("VF_SOURCE_ROOT", "/path/to/clinical_exports"))

# Humphrey 24-2 visual fields (R data frame "bolandSS24.20.20.33")
train_boland = str(SOURCE_ROOT / "boland190822_SS24.20.20.33.RData")

# Merged 24-2 visual field table used for the structure-function dataset
vffile = str(SOURCE_ROOT / "hfa_ongoing_merged_24-2_subset_cleaned_all_in.csv")

# Cirrus Optic Disc Cube (OCT) metadata
newoctfile = str(SOURCE_ROOT / "metadata_OpticDiscCube_ongoing_cleaned.csv")
