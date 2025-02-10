# RC Car Controller with Computer Vision

An RC Car Controller application that combines real-time video streaming with advanced computer vision capabilities. The application provides both manual and autonomous control modes, featuring face detection, YOLO-based object detection, and an autopilot mode.

## Features
- Real-time video streaming from RC car camera
- Manual control through keyboard and GUI buttons
- Face detection using Haar Cascade classifier
- YOLO-based object detection (stop signs, speed signs)
- Autopilot mode with lane detection and tracking
- Multiple speed control levels (40%, 60%, 80%, 100%)
- Dual control interfaces (GUI buttons and keyboard controls)
- Object detection visualization window
- Adaptive thresholding for lane detection
- Real-time control response system

## Demonstrations

### Autopilot Driving Mode
Watch the RC car navigate autonomously using lane detection and tracking:

https://github.com/junouno7/selfdriving-car/raw/main/videos/selfdrive.gif

### Face Detection
Demonstration of the real-time face detection:

https://github.com/junouno7/selfdriving-car/raw/main/videos/facedetect.gif

### Traffic Sign Detection
See the YOLO-based traffic sign detection in action:

https://github.com/junouno7/selfdriving-car/raw/main/videos/signdetect.gif

Closer look at YOLO sign detection:

https://github.com/junouno7/selfdriving-car/raw/main/videos/yolo.gif

## Requirements

### Hardware
- RC Car with camera module
- ESP32-CAM or similar compatible camera module
- Computer with network connectivity
- Local network connection between RC car and computer

### Software
- Python 3.8 or higher
- Required Python packages:
  ```
  PyQt5
  opencv-python
  numpy
  torch
  torchvision
  ultralytics
  ```

## Installation

1. Clone the repository:
```bash
git clone [https://github.com/junouno7/selfdriving-car]
cd [selfdriving-car]
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Download the YOLO model weights:
- Ensure the `best.pt` file (YOLO weights) is in the root directory
- If not present, train your own YOLO model or use a pre-trained model

4. Configure network settings:
- Update the `ip` variable in `main.py` to match your RC car's IP address
- Default IP is set to: '192.168.137.129'

## Usage

1. Start the application:
```bash
python main.py
```

2. The application window will open with:
- Live video feed from the RC car
- Control buttons for movement
- Speed control buttons
- Status indicators for different modes

### Control Modes

#### Manual Control
- **Keyboard Controls:**
  - W: Move Forward
  - S: Move Backward
  - A: Move Left
  - D: Move Right
  - Q: Turn Left
  - E: Turn Right
 

- **GUI Buttons:**
  - Forward/Backward movement
  - Left/Right movement
  - Turn Left/Right
  - Speed control (40%, 60%, 80%, 100%)

#### Special Features

1. **Face Detection (H key)**
- Toggle face detection mode
- Detects and highlights faces in the video feed
- Uses Haar Cascade classifier

2. **Autopilot Mode (P key)**
- Toggle autonomous driving mode
- Performs lane detection and tracking
- Automatically controls steering
- Disables manual controls when active

3. **YOLO Detection (Y key)**
- Toggle YOLO object detection
- Detects traffic signs (stop, speed)
- Automatically responds to detected signs
- Shows detection confidence levels
