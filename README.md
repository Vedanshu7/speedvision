# speedvision

Real-time vehicle speed detection from highway camera footage using OpenCV.

`speedvision` processes a highway video, automatically detects lane boundaries, splits the feed by lane, tracks vehicles across detection lines, and calculates their speed in km/h. Vehicles exceeding the speed limit are flagged and their images saved.

## Input / Output

**Input: reference frame used for lane detection**

![Input reference frame](assets/input-image.jpg)

**Lane boundary detection (greyscale + Canny edges)**

![Greyscale edge detection](assets/gray.jpeg)

**Output: annotated frames with vehicle detection and speed overlay**

| Detection output | Alt view |
|---|---|
| ![Output](assets/output.jpeg) | ![Output alt](assets/output1.jpeg) |

**Raw per-lane split clips** (pre-processed, from `assets/samples/`):

| Lane 1 | Lane 2 | Lane 3 |
|--------|--------|--------|
| ![Lane 1](assets/samples/output1.gif) | ![Lane 2](assets/samples/output2.gif) | ![Lane 3](assets/samples/output3.gif) |

| Lane 4 | Lane 5 | Lane 6 |
|--------|--------|--------|
| ![Lane 4](assets/samples/output4.gif) | ![Lane 5](assets/samples/output5.gif) | ![Lane 6](assets/samples/output6.gif) |

**Full pipeline output: annotated per-lane clips with speed overlay** (from `assets/output/`):

| Lane 1 | Lane 2 | Lane 3 |
|--------|--------|--------|
| ![Lane 1](assets/output/output1.gif) | ![Lane 2](assets/output/output2.gif) | ![Lane 3](assets/output/output3.gif) |

| Lane 4 | Lane 5 | Lane 6 |
|--------|--------|--------|
| ![Lane 4](assets/output/output4.gif) | ![Lane 5](assets/output/output5.gif) | ![Lane 6](assets/output/output6.gif) |

---

## How It Works

1. **Lane detection** — uses Canny edge detection and contour analysis on a reference image to find lane boundaries
2. **Video splitting** — splits the input video into per-lane video streams
3. **Speed calculation** — tracks vehicle centroids across two detection lines, computes km/h using pixel-to-km calibration and frame timing
4. **Flagging** — saves snapshots of vehicles exceeding the speed limit

## Tech Stack

- **Language:** Python 3
- **Vision:** OpenCV (`cv2`)
- **Parallelism:** Python `threading` (one thread per lane)

## Project Structure

```
speedvision/
├── src/
│   └── speed_detect.py     # Full pipeline: lane detection, splitting, speed calc
├── assets/
│   ├── input-image.jpg     # Reference frame for lane detection
│   ├── input.mp4           # Sample input highway video
│   ├── gray.jpeg           # Greyscale reference output
│   ├── output.jpeg         # Sample detection output
│   ├── output1.jpeg        # Sample detection output (alt)
│   ├── output/             # Annotated pipeline output GIFs (with speed overlay)
│   │   ├── output1.gif
│   │   └── ...
│   └── samples/            # Raw per-lane split clips (pre-speed detection)
│       ├── output1.avi
│       ├── output2.avi
│       ├── output3.avi
│       ├── output4.avi
│       ├── output5.avi
│       └── output6.avi
├── requirements.txt
└── LICENSE
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/speed_detect.py \
  --input-image assets/input-image.jpg \
  --input-video assets/input.mp4 \
  --output-dir output \
  --speed-limit 100
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--input-image` | `assets/input-image.jpg` | Reference frame for lane detection |
| `--input-video` | `assets/input.mp4` | Highway video to process |
| `--output-dir` | `output` | Directory for split/final videos and images |
| `--speed-limit` | `100` | Speed limit in km/h; faster vehicles are flagged |
| `--start-line` | `100` | Start detection line x-coordinate |
| `--stop-line` | `400` | Stop detection line x-coordinate |

## Output Structure

```
output/
├── split/
│   ├── output1.avi         # Per-lane split video
│   └── output2.avi
└── final/
    ├── output1.avi         # Annotated output with speed overlay
    └── image/
        ├── 1/              # All vehicle snapshots for lane 1
        └── overspeed/      # Snapshots of vehicles over the limit
```

## Calibration

The default `km_per_pix = 0.0035 / lane_height` ratio is calibrated for standard highway footage. Adjust this constant in `src/speed_detect.py` for different camera heights or zoom levels.

## License

[MIT](LICENSE)
