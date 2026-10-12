"""
app.py — Main entry point for the AI Knowledge Graph Builder dashboard.

Usage:
    streamlit run app.py

This file re-exports the Streamlit app defined in milestone_4/app_dashboard.py
so the project has a single, standard launch command from the project root.
"""

import os
import sys

# Add milestone_4 to path then import the dashboard module.
# Streamlit will pick up and render all st.* calls made during import.
dashboard_dir = os.path.join(os.path.dirname(__file__), "milestone_4")
sys.path.insert(0, dashboard_dir)

# Also expose milestone_2 and milestone_3 for cross-module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "milestone_2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "milestone_3"))

# Import the dashboard — Streamlit renders it automatically
import app_dashboard  # noqa: F401  (side-effect import is intentional)