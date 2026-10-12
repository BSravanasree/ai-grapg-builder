"""
main_m2.py — Milestone 2 runner.

Simply calls the data_bridge pipeline which handles the full
M1 → M2 flow: CSV ingestion → NLP extraction → Neo4j graph build.

Usage:
    python main_m2.py
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_bridge import run_pipeline

if __name__ == "__main__":
    # Set limit=None to process all rows; use a number (e.g. 200) for a quick test
    run_pipeline(limit=100)