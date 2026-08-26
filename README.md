# drone-inspection-path-planning
Computer vision and path-planning components developed for an autonomous drone inspection project.
# Computer Vision-Assisted Drone Inspection & Path Planning

A Python-based prototype combining computer vision processing and A* path planning for autonomous drone inspection.

## Overview

This project explores the integration of computer vision and path planning for automated drone inspection.

The repository contains selected implementation components from an academic project, including:

- YOLO-based computer vision processing
- Object detection and analysis
- Worker tracking
- PPE compliance analysis
- Hazard-zone violation detection
- 2D occupancy-grid representation
- A* path planning
- Route visualization

Project-specific datasets, trained model weights, and other restricted materials are not included.

## System Overview

The project consists of two main components:

1. Computer Vision Pipeline
2. A* Path Planning

The general workflow is:

Computer Vision → Object Detection → Object Analysis / Obstacle Representation → Occupancy Grid → A* → Planned Route

## Computer Vision Pipeline

The computer vision component uses YOLO for object detection and provides functions for processing images and video.

Key functionality includes:

- Single-image object detection
- Batch image processing
- Video inference
- Bounding-box extraction
- Detection filtering
- Worker tracking
- PPE compliance analysis
- Hazard-zone violation detection

The detection results are converted into structured information that can be used for downstream analysis.

The original trained model weights and project-specific data are not included in this repository.

## Path Planning

The path-planning component uses the A* search algorithm to calculate a route through a 2D occupancy grid.

The implementation supports:

- 8-directional movement
- Straight and diagonal movement costs
- Octile-distance heuristic
- Obstacle avoidance
- Diagonal corner-cutting prevention
- Path reconstruction
- ASCII-based route visualization

The example uses manually defined obstacle coordinates to demonstrate the path-planning process without requiring project-specific data.

## Requirements

- Python 3.9+
- NumPy
- OpenCV
- Ultralytics

Install the dependencies with:

    pip install -r requirements.txt

## Usage

### Path Planning

Run the path-planning example:

    python path_planning.py

The program creates a sample 10 x 10 navigation grid and calculates a route from the defined starting position to the destination.

The output includes the calculated route and an ASCII visualization.

### Visualization Legend

- S = Start
- G = Goal
- # = Obstacle
- * = Calculated Path
- . = Free Space

### Computer Vision

The computer vision pipeline requires a compatible YOLO model.

Example:

    from vision_pipeline import detect_single_image

    results = detect_single_image(
        image_path="example.jpg",
        model_path="path/to/model.pt",
        conf_threshold=0.5
    )

Model weights and input data used in the original academic project are not included in this repository.

## Algorithm

A* selects the next position based on:

    f(n) = g(n) + h(n)

where:

- g(n) represents the cost from the starting position
- h(n) estimates the cost from the current position to the destination
- f(n) is the total estimated cost

Because the implementation allows movement in eight directions, an octile-distance heuristic is used.

## Limitations

This implementation represents a prototype of selected components rather than a complete autonomous drone system.

The current path-planning implementation uses a static 2D environment and does not address:

- Dynamic obstacle avoidance
- Real-time path replanning
- 3D flight dynamics
- Flight-controller integration
- Sensor uncertainty

The computer vision component also depends on compatible YOLO model weights, which are not included in this repository.

## Project Materials

This repository contains selected source code from an academic project.

Original datasets, trained model weights, and other project-specific materials are not included due to confidentiality and usage restrictions.
