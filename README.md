
# Blind Criminal

Welcome to **Blind Criminal App**

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
  - [Run Gun Detection and Tracking](#run-gun-detection-and-tracking)
  - [Use Light Function with Pipeline](#use-light-function-with-pipeline)
  - [Camera Tracking Only](#camera-tracking-only)
  - [Camera Tracking with Light Function](#camera-tracking-with-light-function)
  - [Manual Camera Control](#manual-camera-control)
- [Camera Navigation Controls](#camera-navigation-controls)

## Requirements

- Jetson Nano or Orin Nano
- Python 3.x
- Jetpack 6.0

## Installation

To get started, you need to install the required dependencies and clone this repository.

1. **Install Jetson Inference**:  
   Follow the instructions in the docs/jetson-inference-installation.md to set up Jetson Inference.

2. **Install Ultralytics**:  
   Run the following command to install Ultralytics:
   ```bash
   pip3 install ultralytics
   ```

3. **Clone this repository**:
   ```bash
   git clone https://gitlab.com/nexpredsolutionscode/blind-criminal-application.git
   cd blind-criminal-application
   git checkout dev
   ```
4. **Allow permissions to run**:
   ```bash
   chmod +x permissions.sh
   ./permissions.sh
   ```


## Setup

Ensure that `USB0` is connected properly for the PTZ camera to function. Follow the steps in the [Installation](#installation) section to set up your environment.

## Usage

### Run Gun Detection and Tracking
#### Note: Running this for the first time will take time
To run the app for gun detection and following a person, use the following command:

```bash
./main.sh
```
To quit Press q
### Use Light Function with Pipeline
To run the gun detection, track a person, and use the light function, use:

```bash
./mainlight.sh
```
To quit Press q
### Camera Tracking Only
To run the camera and track a person without gun detection, use:

```bash
./ptztracking.sh
```
To quit Press q
### Camera Tracking with Light Function
To track a person and use the light function, run:

```bash
./ptztrackinglight.sh
```
To quit Press q
### Manual Camera Control
To manually control or navigate the camera, use:

```bash
./cameracontrol.sh
```
To quit Press q
## Camera Navigation Controls

- **W**: Move Up
- **A**: Move Left
- **S**: Move Down
- **D**: Move Right
- **O**: Control the Light (First press to turn on, second press to flash, third press to turn off)
- **Spacebar**: Stop

Ensure that `USB0` is properly connected for the PTZ camera to move.


