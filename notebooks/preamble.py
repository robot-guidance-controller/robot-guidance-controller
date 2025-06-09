# Notebooks that want to use code in the src/ directory must import this file

import sys
import os

# Add src/ to the Python path if not already added

src_paths = [
    '../src',
    '../external/shared-language-force-embedding/src',
]

for path in src_paths:
    src_path = os.path.abspath(os.path.join(os.getcwd(), path))

    if src_path not in sys.path:
        sys.path.append(src_path)
