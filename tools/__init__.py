"""Minimal local tool shims for evaluation scripts.

The original TravelPlanner evaluation code expects a `tools.*.apis` layout.
This repo uses CSVs under `database/`, so we provide light wrappers that load
those CSVs locally.
"""

