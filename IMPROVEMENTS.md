# speedvision — Improvement Roadmap

## Detection Accuracy

- **YOLOv8 integration** — replace background subtraction with YOLOv8 object detection for far better vehicle recognition in varied lighting/weather
- **Deep SORT tracking** — use DeepSORT for multi-object tracking across frames instead of single-contour heuristics
- **Auto-calibration** — detect known road markings (lane width = 3.5m standard) to auto-compute `km_per_pix`

## Features

- **Real-time webcam / RTSP stream** — add `--stream rtsp://...` flag for live camera feeds
- **CSV report export** — write all detected vehicle speeds + timestamps + lane to a CSV file
- **Streamlit dashboard** — live web UI showing current speeds, lane stats, and overspeed alerts
- **License plate cropping** — crop and save the plate region for speeding vehicles

## Performance

- **GPU acceleration** — use `cv2.cuda` module for edge detection and optical flow on NVIDIA GPUs
- **Frame skip** — configurable `--frame-skip N` to process every Nth frame for faster throughput

## Distribution

- **Docker image** — `docker run vedanshu7/speedvision --input-video /data/video.mp4`
- **GitHub Actions CI** — lint + test on push
