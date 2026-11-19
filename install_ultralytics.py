import subprocess
import sys

print("Starting ultralytics installation...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "ultralytics"],
    capture_output=True,
    text=True,
    timeout=180,
)
print("STDOUT (tail):", result.stdout[-2000:])
print("STDERR (tail):", result.stderr[-2000:])
print("RC:", result.returncode)
