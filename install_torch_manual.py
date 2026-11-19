import zipfile
import os
import sys

wheel_path = "C:\\Users\\ABDUL KALAM\\OneDrive\\Desktop\\RoadGuardAI\\torch-2.10.0-cp314-cp314-win_amd64.whl"
user_site = "C:\\Users\\ABDUL KALAM\\AppData\\Roaming\\Python\\Python314\\site-packages"

print(f"Extracting wheel to {user_site}...")
try:
    with zipfile.ZipFile(wheel_path, 'r') as z:
        z.extractall(user_site)
    print("Extraction complete!")
    # List what was extracted
    extracted = os.listdir(user_site)
    torch_items = [x for x in extracted if 'torch' in x.lower()]
    print("Extracted torch items:", torch_items[:10])
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
