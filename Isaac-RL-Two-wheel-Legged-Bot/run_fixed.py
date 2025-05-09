#!/usr/bin/env python3
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.getcwd())

# Now import the module using the correct path
from scripts.co_rl.core.runners import OffPolicyRunner

# Import and run the original script
script_path = os.path.join(os.getcwd(), "scripts/co_rl/train.py")
with open(script_path, 'r') as f:
    script_content = f.read()

# Replace problematic import
script_content = script_content.replace(
    "from scripts.co_rl.core.runners import OffPolicyRunner",
    "# Import already done at the top of this file"
)

# Execute the modified script
exec(script_content)
