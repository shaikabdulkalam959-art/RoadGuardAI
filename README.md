# RoadGuard AI

A local Streamlit app for detecting objects in uploaded road-scene images and
videos. It uses a general-purpose Ultralytics YOLO model; its object counts are
not a validated collision-risk prediction.

## Run on Windows

Open PowerShell in this folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open <http://127.0.0.1:8501>. The first analysis with the default `yolov8n.pt`
model downloads its weights, so internet access is required once. To use
different weights, enter their local path in the sidebar.

For video, choose a frame sampling interval and maximum number of sampled
frames to balance detail and processing time.
