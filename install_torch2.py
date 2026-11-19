import subprocess
import sys

print("Installing torch from wheel file...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-deps", "--no-input", "torch_cp314.whl"],
    capture_output=True,
    text=True,
    timeout=600,
)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("RC:", result.returncode)
