# SPDX-License-Identifier: Apache-2.0
"""Recompute this evidence bundle using the checked-in canonical audit."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[3] / "aclgraph_descriptor_audit.py"), run_name="__main__")
