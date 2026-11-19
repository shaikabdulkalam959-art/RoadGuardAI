import urllib.request
import sys
import subprocess
import os

# Torch version for Python 3.14
# We need a compatible torch version - let's try 2.1.x or use the CPU-only index
url = "https://download.pytorch.org/whl/cpu/torch-2.1.0-cp314-cp314-win_amd64.whl"

print("Downloading torch wheel...")
try:
    urllib.request.urlretrieve(
        url,
        "torch_cp314.whl"
    )
    print("Download complete, installing...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "torch_cp314.whl"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    print("STDOUT:", result.stdout[-2000:])
    print("STDERR:", result.stderr[-2000:])
    print("RC:", result.returncode)
except Exception as e:
    print(f"ERROR: {e}")
