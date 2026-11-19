from __future__ import annotations

from collections import Counter
from io import BytesIO
from pathlib import Path
import tempfile
from typing import Any

import streamlit as st


st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading the object detection model...")
def load_model(model_name: str) -> Any:
    """Load and cache an Ultralytics YOLO model the first time it is needed."""
    from ultralytics import YOLO

    return YOLO(model_name)


def summarize_result(result: Any) -> Counter[str]:
    """Count the detected classes in one Ultralytics result."""
    names = result.names
    class_ids = result.boxes.cls.tolist() if result.boxes is not None else []
    return Counter(str(names[int(class_id)]) for class_id in class_ids)


def show_summary(counts: Counter[str], heading: str) -> None:
    st.subheader(heading)
    if not counts:
        st.info("No supported objects were detected.")
        return

    st.bar_chart(dict(counts))
    st.dataframe(
        [{"Class": name, "Detections": count} for name, count in counts.most_common()],
        hide_index=True,
        width="stretch",
    )
    csv = "class,detections\n" + "".join(
        f"{name},{count}\n" for name, count in counts.most_common()
    )
    st.download_button(
        "Download detection counts",
        data=csv,
        file_name="roadguard-detections.csv",
        mime="text/csv",
    )


def analyze_image(image_bytes: bytes, model: Any, confidence: float) -> None:
    from PIL import Image
    import numpy as np

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    result = model.predict(np.asarray(image), conf=confidence, verbose=False)[0]
    annotated = result.plot()[:, :, ::-1].copy()
    counts = summarize_result(result)

    original_column, result_column = st.columns(2)
    with original_column:
        st.image(image, caption="Uploaded image", width="stretch")
    with result_column:
        st.image(annotated, caption="Detected objects", width="stretch")
    show_summary(counts, "Detection summary")


def analyze_video(
    video_bytes: bytes,
    file_name: str,
    model: Any,
    confidence: float,
    sample_every: int,
    max_samples: int,
) -> None:
    import cv2

    extension = Path(file_name).suffix or ".mp4"
    temp_path: str | None = None
    capture = None
    try:
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            raise ValueError("This video could not be opened. Try an MP4, AVI, or MOV file.")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = max(1, frame_count)
        frame_index = 0
        processed = 0
        counts: Counter[str] = Counter()
        preview = None
        progress = st.progress(0, text="Analyzing sampled video frames...")

        while processed < max_samples:
            ok, frame = capture.read()
            if not ok:
                break
            current_index = frame_index
            frame_index += 1
            if current_index % sample_every:
                continue

            result = model.predict(frame, conf=confidence, verbose=False)[0]
            counts.update(summarize_result(result))
            if preview is None or (
                result.boxes is not None and len(result.boxes) > 0
            ):
                preview = result.plot()[:, :, ::-1].copy()

            processed += 1
            progress.progress(
                min(current_index / total_frames, 1.0),
                text=f"Analyzed {processed} sampled frame(s)...",
            )

        progress.empty()
        if processed == 0:
            raise ValueError("No video frames could be read from this file.")

        metrics = st.columns(2)
        metrics[0].metric("Sampled frames analyzed", processed)
        metrics[1].metric("Sampling interval", f"Every {sample_every} frame(s)")
        if preview is not None:
            st.image(preview, caption="Annotated sampled frame", width="stretch")
        show_summary(counts, "Detections across sampled frames")
    finally:
        if capture is not None:
            capture.release()
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)


def main() -> None:
    st.title("RoadGuard AI")
    st.caption("Road-scene object detection for uploaded images and videos.")

    with st.sidebar:
        st.header("Analysis settings")
        analysis_type = st.radio("Input type", ["Image", "Video"])
        model_name = st.text_input(
            "YOLO model or weights path",
            value="yolov8n.pt",
            help="The default model weights download automatically the first time you analyze a file.",
        )
        confidence = st.slider("Detection confidence", 0.1, 0.9, 0.35, 0.05)
        if analysis_type == "Video":
            sample_every = st.slider(
                "Analyze every Nth frame", min_value=1, max_value=60, value=10
            )
            max_samples = st.slider(
                "Maximum sampled frames", min_value=1, max_value=300, value=60
            )

    if analysis_type == "Image":
        uploaded_file = st.file_uploader(
            "Upload a road-scene image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a road-scene video",
            type=["mp4", "avi", "mov", "mkv"],
        )

    st.info(
        "This app reports objects recognized by a general-purpose YOLO model. "
        "It is not a validated collision-risk predictor or a substitute for safe driving."
    )

    if uploaded_file is None:
        st.write("Choose an input file to get started.")
        return

    if not model_name.strip():
        st.error("Enter a model name or a valid path to YOLO weights.")
        return

    if st.button("Analyze", type="primary"):
        try:
            model = load_model(model_name.strip())
            if analysis_type == "Image":
                analyze_image(uploaded_file.getvalue(), model, confidence)
            else:
                analyze_video(
                    uploaded_file.getvalue(),
                    uploaded_file.name,
                    model,
                    confidence,
                    sample_every,
                    max_samples,
                )
        except Exception as error:
            st.error(f"Analysis failed: {error}")
            st.caption(
                "Check that the model weights are available and that the uploaded file "
                "is valid. The default YOLO weights require internet access on first use."
            )


if __name__ == "__main__":
    main()
