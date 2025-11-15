# A simple python script to allow importing modules from the base_dir in notebooks
import sys
import os
from pathlib import Path
base_dir = str(Path(os.getcwd()).parents[1])
if base_dir not in sys.path:
    sys.path.append(base_dir)