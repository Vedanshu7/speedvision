"""
speedvision: Real-time vehicle speed detection from highway camera footage.

Pipeline overview:
    1. detect_lanes()        — Canny edge detection on a reference image to
                               locate lane boundary y-coordinates.
    2. split_video_by_lane() — Cut the input video into per-lane clips.
    3. calculate_speeds()    — Frame-diff tracking per lane; compute km/h and
                               flag overspeed vehicles.

Usage::

    python src/speed_detect.py \\
        --input-image assets/input-image.jpg \\
        --input-video assets/input.mp4 \\
        --output-dir  output \\
        --speed-limit 100

All outputs land in ``<output-dir>/split/`` (raw lane clips) and
``<output-dir>/final/`` (annotated clips + snapshots).

Import as:

    import src.speed_detect as speedvision
"""

from __future__ import annotations

import argparse
import logging
import os
import threading

import cv2

_LOG = logging.getLogger(__name__)

# =========================================================================
# CLI.
# =========================================================================


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :return: parsed namespace with input_image, input_video, output_dir,
             speed_limit, start_line, stop_line
    """
    parser = argparse.ArgumentParser(
        description="Detect vehicle speeds from highway video footage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-image", default="assets/input-image.jpg",
                        help="Reference frame used for lane boundary detection.")
    parser.add_argument("--input-video", default="assets/input.mp4",
                        help="Highway video file to process.")
    parser.add_argument("--output-dir", default="output",
                        help="Root directory for all output files.")
    parser.add_argument("--speed-limit", type=int, default=100,
                        help="Speed limit in km/h; faster vehicles are flagged.")
    parser.add_argument("--start-line", type=int, default=100,
                        help="X-coordinate of the speed-measurement start line.")
    parser.add_argument("--stop-line", type=int, default=400,
                        help="X-coordinate of the speed-measurement stop line.")
    return parser.parse_args()


# =========================================================================
# Step 1: Lane detection.
# =========================================================================


def detect_lanes(input_image_path: str) -> list[int]:
    """
    Detect horizontal lane boundaries from a reference highway image.

    Applies Canny edge detection followed by contour analysis to find
    large horizontal structures (lane dividers). Boundaries closer than
    70 pixels apart are merged.

    :param input_image_path: path to a single frame showing lane markings
    :return: sorted list of y-coordinates representing the top edge of each lane
    :raises FileNotFoundError: if the image cannot be read from disk
    """
    image = cv2.imread(input_image_path)
    if image is None:
        raise FileNotFoundError(f"Reference image not found: {input_image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Collect y-coordinates of wide contours — these correspond to lane edges.
    y_coords: list[int] = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w >= 100:
            y_coords.append(y)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    y_coords.sort()

    # Merge boundaries closer than 70px — they belong to the same edge.
    boundaries: list[int] = []
    for i in range(len(y_coords) - 1):
        if y_coords[i + 1] - y_coords[i] > 70:
            boundaries.append(y_coords[i])
    if y_coords and (not boundaries or abs(boundaries[-1] - y_coords[-1]) > 70):
        boundaries.append(y_coords[-1])

    return boundaries


# =========================================================================
# Step 2: Video splitting by lane.
# =========================================================================


def split_video_by_lane(
    input_video: str,
    lane_boundaries: list[int],
    split_dir: str,
) -> int:
    """
    Slice the input video into one clip per lane.

    :param input_video: path to the full-width highway video
    :param lane_boundaries: sorted y-coordinate list from :func:`detect_lanes`
    :param split_dir: directory where per-lane AVI files are written
    :return: number of lane clips written
    :raises FileNotFoundError: if the video cannot be opened
    """
    os.makedirs(split_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read first frame of input video.")

    # One VideoWriter per lane.
    writers: list[cv2.VideoWriter] = []
    for i in range(len(lane_boundaries) - 1):
        upper, lower = lane_boundaries[i], lane_boundaries[i + 1]
        path = os.path.join(split_dir, f"output{i + 1}.avi")
        writers.append(cv2.VideoWriter(path, fourcc, fps, (frame_width, lower - upper)))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        for i, writer in enumerate(writers):
            upper, lower = lane_boundaries[i], lane_boundaries[i + 1]
            writer.write(frame[upper:lower, 0:frame_width])
        frame_count += 1
        _LOG.info("Splitting video: %.1f%%", frame_count / total_frames * 100)

    cap.release()
    for w in writers:
        w.release()

    return len(writers)


# =========================================================================
# Step 3: Speed calculation per lane (parallel threads).
# =========================================================================


def _km_per_pixel(lane_height: int) -> float:
    """
    Estimate the real-world distance (km) represented by one pixel.

    The constant 0.0035 km is an empirical calibration for standard highway
    footage from an overhead camera at approximately 10 m height. Adjust for
    different camera setups.

    :param lane_height: height of the lane crop in pixels
    :return: kilometres per pixel ratio
    """
    return 0.0035 / lane_height if lane_height > 0 else 0.0


def calculate_speeds(
    lane_index: int,
    split_dir: str,
    final_dir: str,
    speed_limit: int,
    start_x: int,
    stop_x: int,
) -> None:
    """
    Detect and record vehicle speeds for one lane clip.

    Uses background subtraction (frame differencing) to detect moving
    vehicles. Speed is calculated from the time a vehicle takes to travel
    from start_x to stop_x, scaled by the pixel-to-km ratio.

    :param lane_index: 1-based lane number matching the filename from
                       :func:`split_video_by_lane`
    :param split_dir: directory containing the per-lane clip files
    :param final_dir: directory where annotated output and snapshots are saved
    :param speed_limit: threshold in km/h above which a vehicle is flagged
    :param start_x: left detection line x-coordinate
    :param stop_x: right detection line x-coordinate
    """
    lane_img_dir = os.path.join(final_dir, "image", str(lane_index))
    overspeed_dir = os.path.join(final_dir, "image", "overspeed")
    os.makedirs(lane_img_dir, exist_ok=True)
    os.makedirs(overspeed_dir, exist_ok=True)

    clip_path = os.path.join(split_dir, f"output{lane_index}.avi")
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        _LOG.warning("Could not open %s.", clip_path)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sec_per_frame = 1.0 / fps
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    ret, frame1 = cap.read()
    if not ret:
        cap.release()
        return

    km_per_pix = _km_per_pixel(frame1.shape[0])
    # Minimum contour area to classify as a vehicle (2/3 of lane area).
    min_vehicle_area = (frame1.shape[0] ** 2) * 2 / 3

    out = cv2.VideoWriter(
        os.path.join(final_dir, f"output{lane_index}.avi"),
        fourcc, fps, (frame1.shape[1], frame1.shape[0]),
    )

    ret, frame2 = cap.read()

    # Tracking state.
    tracking_start = False
    tracking_end = False
    frame_at_start = 0
    frame_at_end = 0
    x_at_start = 0.0
    x_at_end = 0.0
    phase = 0
    snap_pending = False
    frame_count = 0

    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < min_vehicle_area:
                continue

            (x, y, w, h) = cv2.boundingRect(contour)
            cx = x + w * 0.5

            cv2.line(frame1, (stop_x, 10), (stop_x, 100), (255, 0, 0), 5)
            cv2.line(frame1, (start_x, 10), (start_x, 100), (255, 0, 0), 5)

            # Annotate speed when the vehicle passes the stop line.
            if cx > stop_x and phase == 2:
                time_elapsed = (frame_at_end - frame_at_start) * sec_per_frame / 3600
                distance = abs(x_at_end - x_at_start) * km_per_pix
                speed = distance / time_elapsed if time_elapsed > 0 else 0.0

                cv2.putText(
                    frame1, f"speed: {speed:.1f} km/h",
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3,
                )

                # Capture one snapshot per vehicle pass.
                if snap_pending and stop_x < cx < stop_x + 100:
                    snap_pending = False
                    cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    snapshot_name = f"{speed:.1f}kmh.jpg"
                    cv2.imwrite(os.path.join(lane_img_dir, snapshot_name), frame1)
                    if speed > speed_limit:
                        cv2.imwrite(
                            os.path.join(overspeed_dir, f"lane{lane_index}_{snapshot_name}"),
                            frame1,
                        )

            if cx > stop_x or cx < start_x:
                continue

            # Mark the start of measurement.
            if cx < start_x + 100 and not tracking_start:
                tracking_start, tracking_end = True, False
                frame_at_start = frame_count
                x_at_start = cx
                phase = 1

            # Mark the end of measurement.
            if cx > stop_x - 100 and not tracking_end and tracking_start:
                tracking_end, tracking_start = True, False
                frame_at_end = frame_count
                x_at_end = cx
                phase = 2
                snap_pending = True

            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame1, (int(cx), int(y + h * 0.5)), 10, (0, 0, 255), 50)

        out.write(frame1)
        frame1 = frame2
        ret, frame2 = cap.read()
        if not ret:
            break

        frame_count += 1
        _LOG.info("Lane %d speed analysis: %.1f%%", lane_index, frame_count / total_frames * 100)

        if cv2.waitKey(40) == 27:
            break

    cap.release()
    out.release()


# =========================================================================
# Entry point.
# =========================================================================


def main() -> None:
    """
    Run the full speed-detection pipeline.

    Detects lane boundaries, splits the video by lane, then runs speed
    calculation on each lane in parallel threads.
    """
    args = parse_args()
    split_dir = os.path.join(args.output_dir, "split")
    final_dir = os.path.join(args.output_dir, "final")

    _LOG.info("Step 1/3: Detecting lane boundaries.")
    lanes = detect_lanes(args.input_image)
    if len(lanes) < 2:
        _LOG.error("Fewer than 2 lane boundaries detected. Check the reference image.")
        return
    _LOG.info("Found %d lane(s): boundaries at y=%s.", len(lanes) - 1, lanes)

    _LOG.info("Step 2/3: Splitting video by lane.")
    num_lanes = split_video_by_lane(args.input_video, lanes, split_dir)
    _LOG.info("Wrote %d lane clip(s) to %s.", num_lanes, split_dir)

    _LOG.info("Step 3/3: Calculating vehicle speeds (one thread per lane).")
    threads: list[threading.Thread] = [
        threading.Thread(
            target=calculate_speeds,
            args=(i + 1, split_dir, final_dir, args.speed_limit, args.start_line, args.stop_line),
            name=f"lane-{i + 1}",
        )
        for i in range(num_lanes)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    _LOG.info("Done. All output in: %s", args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
