from __future__ import annotations

import tempfile
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st

st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model(model_name: str) -> Any:
    from ultralytics import YOLO
    return YOLO(model_name)


def summarize_result(result: Any) -> Counter:
    if result.boxes is None:
        return Counter()
    names = result.names
    counts = Counter()
    for box in result.boxes:
        class_id = int(box.cls.item())
        name = names[class_id] if isinstance(names, dict) else str(names[class_id])
        counts[name] += 1
    return counts


def analyze_image(image_bytes: bytes, model: Any, confidence: float) -> None:
    from PIL import Image, UnidentifiedImageError
    import numpy as np

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded image is not a valid JPG, PNG, BMP, or WebP file.") from exc

    result = model.predict(
        np.asarray(image), conf=confidence, imgsz=640, verbose=False
    )[0]
    annotated = result.plot()[:, :, ::-1].copy()
    counts = summarize_result(result)

    left, right = st.columns(2)
    left.image(image, caption="Uploaded image", use_container_width=True)
    right.image(annotated, caption="Detected objects", use_container_width=True)

    show_results(counts)


def analyze_video(
    video_bytes: bytes,
    file_name: str,
    model: Any,
    confidence: float,
    sample_every: int,
    max_frames: int,
) -> None:
    import cv2

    suffix = Path(file_name).suffix.lower() or ".mp4"
    temp_path = None
    capture = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            raise ValueError(
                "This video could not be opened. Use an MP4/H264 or AVI file. "
                "MOV or MKV files may need to be converted to MP4."
            )

        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = max(frame_count, 1)

        processed = 0
        frame_index = 0
        counts = Counter()
        preview = None
        progress = st.progress(0.0, text="Analyzing video frames...")

        while processed < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % sample_every == 0:
                result = model.predict(
                    frame,
                    conf=confidence,
                    imgsz=640,
                    verbose=False,
                )[0]
                counts.update(summarize_result(result))
                preview = result.plot()[:, :, ::-1].copy()
                processed += 1

                progress.progress(
                    min(frame_index / total_frames, 1.0),
                    text=f"Analyzed {processed} sampled frame(s)",
                )

            frame_index += 1

        progress.empty()

        if processed == 0:
            raise ValueError("No readable frames were found in this video.")

        duration = frame_index / fps
        metrics = st.columns(3)
        metrics[0].metric("Frames sampled", processed)
        metrics[1].metric("Video duration", f"{duration:.1f} sec")
        metrics[2].metric("Sample interval", f"{sample_every} frame(s)")

        if preview is not None:
            st.image(preview, caption="Latest annotated sampled frame", use_container_width=True)

        show_results(counts)

    finally:
        if capture is not None:
            capture.release()
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


def show_results(counts: Counter) -> None:
    st.subheader("Detection summary")
    if not counts:
        st.info("No objects were detected at the selected confidence.")
        return

    chart_data = dict(counts)
    st.bar_chart(chart_data)
    rows = [{"Class": name, "Detections": count} for name, count in counts.most_common()]
    st.dataframe(rows, hide_index=True, use_container_width=True)

    csv = "class,detections\n" + "".join(
        f"{name},{count}\n" for name, count in counts.most_common()
    )
    st.download_button(
        "Download detection counts",
        csv,
        file_name="roadguard-detections.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("RoadGuard AI")
    st.caption("Road-scene object detection for uploaded images and videos")

    with st.sidebar:
        st.header("Analysis settings")
        input_type = st.radio("Input type", ["Image", "Video"])
        model_name = st.text_input("YOLO model or weights path", "yolov8n.pt")
        confidence = st.slider("Detection confidence", 0.10, 0.90, 0.35, 0.05)

        if input_type == "Video":
            sample_every = st.slider("Analyze every Nth frame", 1, 60, 10)
            max_frames = st.slider("Maximum sampled frames", 1, 300, 60)

    uploaded = st.file_uploader(
        "Upload a road-scene image" if input_type == "Image" else "Upload a road-scene video",
        type=["jpg", "jpeg", "png", "bmp", "webp"] if input_type == "Image"
        else ["mp4", "avi", "mov", "mkv"],
    )

    st.info(
        "This app uses a general-purpose YOLO model. It does not provide a "
        "validated collision-risk prediction or replace safe-driving judgment."
    )

    if uploaded is None:
        st.write("Choose an image or video to begin.")
        return

    if not model_name.strip():
        st.error("Enter a model name or a local weights path.")
        return

    if st.button("Analyze", type="primary"):
        try:
            model = load_model(model_name.strip())
            if input_type == "Image":
                analyze_image(uploaded.getvalue(), model, confidence)
            else:
                analyze_video(
                    uploaded.getvalue(),
                    uploaded.name,
                    model,
                    confidence,
                    sample_every,
                    max_frames,
                )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.caption(
                "Try an MP4 (H264) video or JPG/PNG image, and wait for model weights "
                "to finish downloading the first time."
            )


if __name__ == "__main__":
    main()
