# Computer Vision-Assisted Drone Inspection and Path Planning

An academic prototype exploring two components of a possible drone inspection
workflow: YOLO-based image/video analysis and A* route planning on a manually
defined 2D occupancy grid.

## Project status and scope

This was a school project. The team defined the problem and implemented selected
prototype components, but time and resource limits prevented real-world tests.
It has not been validated on a real drone, a live inspection site, or operational
safety data. The vision output is not automatically converted into a flight
map; the path-planning demo uses manually specified obstacle coordinates.

The repository does not include the original dataset or trained YOLO weights.
The PPE association and worker tracking methods are simple heuristics and need
validation against labeled data before they could support safety decisions.

## Components

### Computer vision: `ppe_and_helmet_detection.py`

Includes helpers for single-image, batch-image, and video inference, annotated
outputs, class filtering, simple PPE checks, rectangular hazard-zone checks, and
nearest-centroid worker tracking. The model must be compatible with
[Ultralytics YOLO](https://docs.ultralytics.com/).

### Path planning: `path_finding.py`

Creates a 2D binary occupancy grid and runs A* with eight-directional movement,
straight cost 1, diagonal cost `sqrt(2)`, an octile-distance heuristic, and
diagonal corner-cutting prevention. Running the file prints a sample route on a
10 x 10 grid.

## Requirements

- Python 3.9 or newer
- NumPy
- OpenCV
- Ultralytics

Create an environment and install dependencies:

```bash
python -m venv .venv
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
```

## Usage

### Run the path-planning example

```bash
python path_finding.py
```

The script prints the route and an ASCII grid. Legend: `S` = start, `G` = goal,
`#` = obstacle, `*` = route, and `.` = free cell.

### Run image detection

Provide your own compatible weights and input image:

```python
from ppe_and_helmet_detection import detect_single_image

result = detect_single_image(
    image_path="path/to/image.jpg",
    model_path="path/to/model.pt",
    conf_threshold=0.5,
)
print(result)
```

The model's class names must match the labels expected by helpers such as
`check_ppe_compliance` (which currently looks for the class name `worker`).
Function examples are not a substitute for the missing project weights/data.

## Algorithm

A* prioritizes nodes using `f(n) = g(n) + h(n)`, where `g(n)` is the cost from
the start and `h(n)` estimates the remaining cost. Octile distance is suitable
for the eight-direction movement and the straight/diagonal costs used here.

## Limitations and next steps

- Static 2D grid only; no 3D flight dynamics, localization, or flight control.
- No dynamic obstacle updates or real-time replanning.
- No tested connection from detections to occupancy-grid cells.
- No real drone or live-site evaluation was completed.
- PPE checks and tracking are heuristic prototypes, not safety-certified tools.

## Project materials

Project-specific datasets, trained weights, and restricted academic materials
are not included.
