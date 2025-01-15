import os
import sys
import cv2
import numpy as np
from urllib.request import urlopen
import sys
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QMainWindow, QGridLayout, QLabel, QShortcut)
import torch
import time

# Add YOLOv7 directory to system path
yolov7_path = os.path.join(os.path.dirname(__file__), 'yolov7')
if not os.path.exists(yolov7_path):
    raise ImportError("YOLOv7 directory not found. Please clone the YOLOv7 repository to the same directory as this script.")
sys.path.append(yolov7_path)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# Setup IP address and stream connection
ip = '192.168.137.227'
stream = urlopen(f'http://{ip}:81/stream')
buffer = b''

# Initialize car state
car_state = 'stop'

# Function to send commands to the RC car
def send_car_command(action):
    urlopen(f'http://{ip}/action?go={action}')
    print(f"Car action: {action}")

# Functions to control the car (will be triggered on mouse press/release)
def move_forward():
    global car_state
    car_state = 'go'
    send_car_command('forward')

def stop_car():
    global car_state
    car_state = 'stop'
    send_car_command('stop')

def turn_left():
    global car_state
    car_state = 'left'
    send_car_command('left')

def turn_left2():
    global car_state
    car_state = 'Turn left'
    send_car_command('turn_left')

def turn_right():
    global car_state
    car_state = 'right'
    send_car_command('right')

def turn_right2():
    global car_state
    car_state = 'Turn right'
    send_car_command('turn_right')

def move_backward():
    global car_state
    car_state = 'backward'
    send_car_command('backward')

def speed40():
    global car_state
    send_car_command('speed40')

def speed60():
    global car_state
    send_car_command('speed60')

def speed80():
    global car_state
    send_car_command('speed80')

def speed100():
    global car_state
    send_car_command('speed100')

# Function to exit the program gracefully
def exit_program():
    print("Exiting program...")
    # Stop the car before exiting
    urlopen(f'http://{ip}/action?go=stop')
    # Close the OpenCV window
    cv2.destroyAllWindows()
    app.quit()

# Function to handle the video stream in a separate thread
class VideoStreamThread(QThread):
    new_frame_signal = pyqtSignal(np.ndarray)  # Signal to pass the frame to the GUI

    def run(self):
        global buffer
        try:
            while True:
                # Read stream data in chunks
                buffer += stream.read(4096)  # Increase chunk size if necessary
                head = buffer.find(b'\xff\xd8')
                end = buffer.find(b'\xff\xd9')

                if head > -1 and end > -1:
                    jpg = buffer[head:end + 2]
                    buffer = buffer[end + 2:]

                    # Decode the JPEG image
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    img = cv2.flip(img, 0)  # Flip image vertically (if needed)

                    # Emit the frame to be displayed in the GUI
                    self.new_frame_signal.emit(img)

        except Exception as e:
            print(f"Error in video stream: {e}")

class ObjectDetectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection")
        self.setGeometry(100, 100, 400, 400)
        self.detection_label = QLabel(self)
        self.detection_label.setGeometry(0, 0, 400, 400)
        
    def update_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_frame.shape
        q_image = QImage(rgb_frame.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio)
        self.detection_label.setPixmap(scaled_pixmap)

class CarControllerApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("RC Car Controller")
        
        # Flags for the functionality
        self.autopilot_enabled = False
        self.haar_detection_enabled = False  # Flag to track Haar detection state
        self.manual_control_enabled = True  # Flag to track if manual control is enabled
        self.window_resized = False  # New variable to track resizing
        
        # Load the Haar cascade for object detection
        self.haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize YOLOv7 model loading with direct method
        yolo_repo = 'yolov7'
        weights_path = 'yolov7/best.pt'  # Update this path to where your weights file is located

        try:
            device = select_device('')
            self.model = attempt_load(weights_path, map_location=device)
            self.model.eval()
            print("YOLOv7 model loaded successfully.")
            print(f"Number of classes in model.names: {len(self.model.names)}")
            print(f"Sample class names: {self.model.names[:10]}")
        except Exception as e:
            print(f"Error loading YOLOv7 model: {e}")
            self.model = None  # Disable YOLO functionality if loading fails
        
        # Add object detection window
        self.detection_window = ObjectDetectionWindow()
        
        # Add timer for speed control
        self.speed_timer = QTimer()
        self.speed_timer.timeout.connect(self.reset_speed)
        
        # Add detection state tracking
        self.last_detection_time = 0
        self.detection_cooldown = 2  # seconds
        # Create a main layout for the application
        main_layout = QVBoxLayout()

        # Create label "AI CAR" at the top
        self.title_label = QLabel('AI CAR live video', self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 32px; font-weight: bold;")
        main_layout.addWidget(self.title_label)

        # Create the layout for the video stream
        self.video_label = QLabel(self)
        main_layout.addWidget(self.video_label)

        # Create the layout for the car control buttons
        button_layout = QGridLayout()

        # Create buttons for car control (Forward, Left, Right, Backward)
        forward_button = QPushButton("Forward")
        forward_button.setFixedSize(120, 45)
        forward_button.pressed.connect(move_forward)  # When button is pressed
        forward_button.released.connect(stop_car)    # When button is released
        button_layout.addWidget(forward_button, 0, 1)

        left_button = QPushButton("Move Left")
        left_button.setFixedSize(120, 45)
        left_button.pressed.connect(turn_left)        # When button is pressed
        left_button.released.connect(stop_car)        # When button is released
        button_layout.addWidget(left_button, 1, 0)

        left_button2 = QPushButton("Turn Left")
        left_button2.setFixedSize(120, 45)
        left_button2.pressed.connect(turn_left2)        # When button is pressed
        left_button2.released.connect(stop_car)        # When button is released
        button_layout.addWidget(left_button2, 0, 0)

        right_button = QPushButton("Move Right")
        right_button.setFixedSize(120, 45)
        right_button.pressed.connect(turn_right)      # When button is pressed
        right_button.released.connect(stop_car)       # When button is released
        button_layout.addWidget(right_button, 1, 2)

        right_button2 = QPushButton("Turn Right")
        right_button2.setFixedSize(120, 45)
        right_button2.pressed.connect(turn_right2)      # When button is pressed
        right_button2.released.connect(stop_car)       # When button is released
        button_layout.addWidget(right_button2, 0, 2)

        backward_button = QPushButton("Backward")
        backward_button.setFixedSize(120, 45)
        backward_button.pressed.connect(move_backward) # When button is pressed
        backward_button.released.connect(stop_car)    # When button is released
        button_layout.addWidget(backward_button, 2, 1)

        # Create a horizontal layout for the speed buttons (Speed 40, Speed 60, Speed 80, Speed 100)
        speed_layout = QHBoxLayout()
        speed_layout.setSpacing(0)  # No space between buttons

        # Create speed buttons
        speed40_button = QPushButton("Speed 40")
        speed40_button.setFixedSize(80, 30)
        speed40_button.pressed.connect(speed40)
        speed40_button.released.connect(stop_car)
        speed_layout.addWidget(speed40_button)

        speed60_button = QPushButton("Speed 60")
        speed60_button.setFixedSize(80, 30)
        speed60_button.pressed.connect(speed60)
        speed60_button.released.connect(stop_car)
        speed_layout.addWidget(speed60_button)

        speed80_button = QPushButton("Speed 80")
        speed80_button.setFixedSize(80, 30) 
        speed80_button.pressed.connect(speed80)
        speed80_button.released.connect(stop_car)
        speed_layout.addWidget(speed80_button)

        speed100_button = QPushButton("Speed 100")
        speed100_button.setFixedSize(80, 30)
        speed100_button.pressed.connect(speed100)
        speed100_button.released.connect(stop_car)
        speed_layout.addWidget(speed100_button)

        # Add the speed layout to the grid (row 3, columns 0-3)
        button_layout.addLayout(speed_layout, 3, 0, 1, 3)

        # Add the button layout to the main layout
        main_layout.addLayout(button_layout)

        # Status text box
        self.status_label = QLabel('H : Face Detection | P : Autopilot', self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 16px; padding: 10px; border-top: 1px solid #ddd;")
        main_layout.addWidget(self.status_label)

        # Set the layout for the window
        self.setLayout(main_layout)
        self.resize(300, 500)
        # Shortcut keys for face detection and autopilot toggle
        QShortcut(Qt.Key_H, self, self.toggle_face_detection)
        QShortcut(Qt.Key_P, self, self.toggle_autopilot)

        # Start the video stream in a separate thread
        self.video_thread = VideoStreamThread()
        self.video_thread.new_frame_signal.connect(self.update_video_frame)
        self.video_thread.start()

        self.frame_counter = 0  # To track frame counts for downsampling

    def update_video_frame(self, frame):
        """Update the video frame in the GUI."""
        if self.autopilot_enabled:
            # Resize window based on first frame size if not resized manually
            frame = cv2.flip(frame, 0)
            if not self.window_resized:
                h, w, _ = frame.shape
                self.resize(w, h + 100)  # Resize to fit the frame size + status label space
                self.window_resized = True  # Set the window as resized
            
            # Process frame with autopilot (No sign detection)
            processed_frame = frame.copy()
            
            # Update both windows
            if self.detection_window:
                self.detection_window.update_frame(processed_frame)  # Update detection window

            # Process the frame with autopilot
            if self.frame_counter % 5 == 0:  # Skip every 5th frame for autopilot
                self.autopilot_move(frame)
            self.frame_counter += 1
            return

        # Normal frame processing (no autopilot)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_frame.shape
        q_image = QImage(rgb_frame.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap)

        # Process the frame for face detection if enabled
        if self.haar_detection_enabled:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 150), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                color = (155, 50, 190) 
                thickness = 2
                text = "Face"
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = x + (w - text_size[0]) // 2
                text_y = y - 12
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

        # Convert the frame to RGB and update the GUI
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_frame.shape
        q_image = QImage(rgb_frame.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap)

        self.resize(w, h + 100)

    def autopilot_move(self, frame):
        """Autopilot logic (without sign detection)"""
        if self.autopilot_enabled:
            # Continue with existing line detection code only (no sign detection)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces using the Haar cascade classifier (if enabled)
            faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                # If faces are detected, find the center of the first face
                x, y, w, h = faces[0]
                # Calculate midpoint of the detected face
                midpoint_x = x + w // 2
                midpoint_y = y + h // 2
            else:
                # No face detected, use the center of the frame
                height, width = frame.shape[:2]
                midpoint_x = width // 2
                midpoint_y = height // 2

            # Crop the image to only process the bottom half
            height, width = frame.shape[:2]
            frame_cropped = frame[height // 2:, :]  # Only the bottom half of the image

            # Resize the cropped frame to 300x300 for faster processing
            resized_frame = cv2.resize(frame_cropped, (300, 300))

            # Convert resized frame to grayscale
            gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply binary thresholding to highlight the white spaces
            _, thresholded = cv2.threshold(blurred, 65, 255, cv2.THRESH_BINARY_INV)

            # Perform morphological operations (erosion and dilation) to clean the noise
            kernel = np.ones((5, 5), np.uint8)
            cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

            # Find contours (white areas between black lines should form contours)
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

                # If there are more than one contour, process the two largest contours (black lines)
                if len(sorted_contours) >= 2:
                    # Get the bounding boxes for the two largest contours (the black lines)
                    rect1 = cv2.boundingRect(sorted_contours[0])  # First black rectangle
                    rect2 = cv2.boundingRect(sorted_contours[1])  # Second black rectangle

                    # Get the coordinates and size of the two rectangles
                    x1, y1, w1, h1 = rect1
                    x2, y2, w2, h2 = rect2

                    # Calculate the midpoint of the white space between the two black rectangles
                    midpoint_x = (x1 + w1 + x2) // 2  # Midpoint horizontally between the two black lines
                    midpoint_y = (y1 + y2) // 2  # Midpoint vertically, use average of both rectangles' y-positions

                    # Draw a green dot at the calculated midpoint for visualization
                    cv2.circle(frame, (midpoint_x, midpoint_y), 10, (0, 255, 0), -1)

                # If only one black line is detected
                elif len(sorted_contours) == 1:
                    # Get the bounding box for the single contour (black line)
                    rect1 = cv2.boundingRect(sorted_contours[0])
                    x1, y1, w1, h1 = rect1

                    # Calculate the midpoint of the detected black line
                    midpoint_x = x1 + w1 // 2  # Midpoint horizontally of the black line
                    midpoint_y = y1 + h1 // 2  # Midpoint vertically of the black line

                # Draw the midpoint as a green circle for visualization
                cv2.circle(resized_frame, (midpoint_x, midpoint_y), 10, (0, 255, 0), -1)

                # Check if the midpoint is near the center of the track
                track_center = resized_frame.shape[1] // 2  # Center of the image (track center)

                # If the midpoint is within a certain threshold of the track center, move forward
                if abs(midpoint_x - track_center) < 10:  # Track is straight
                    move_forward()
                elif midpoint_x < track_center - 10:  # If the midpoint is far to the left
                    # Gradually turn left if midpoint is on the left side
                    turn_left()
                elif midpoint_x > track_center + 10:  # If the midpoint is far to the right
                    # Gradually turn right if midpoint is on the right side
                    turn_right()
                else:
                    # Default behavior if midpoint is somewhat centered but not perfectly aligned
                    pass
            else:
                # If no contours found, react by moving in the opposite direction
                move_backward()

            # Convert the processed frame to grayscale and apply thresholding to get the binary image
            _, binary_frame = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY)

            # Convert both frames to RGB for display (since OpenCV uses BGR by default)
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            binary_rgb_frame = cv2.cvtColor(binary_frame, cv2.COLOR_GRAY2RGB)

            # Stack both frames horizontally to display side-by-side
            side_by_side_frame = np.hstack((rgb_frame, binary_rgb_frame))  # Horizontally stack the frames

            # Convert the stacked frame to QImage for displaying in the GUI
            h, w, _ = side_by_side_frame.shape
            q_image = QImage(side_by_side_frame.data, w, h, w * 3, QImage.Format_RGB888)

            # Convert the QImage to QPixmap and update the label
            pixmap = QPixmap.fromImage(q_image)
            self.video_label.setPixmap(pixmap)
        
    def toggle_face_detection(self):
        """Toggle face detection on and off."""
        self.haar_detection_enabled = not self.haar_detection_enabled
        if self.haar_detection_enabled:
            self.status_label.setText('H : Face Detection (Enabled) | P : Autopilot')
        else:
            self.status_label.setText('H : Face Detection (Disabled) | P : Autopilot')

    def toggle_autopilot(self):
        """Toggle autopilot on and off."""
        self.autopilot_enabled = not self.autopilot_enabled
        self.manual_control_enabled = not self.autopilot_enabled

        if self.autopilot_enabled:
            self.status_label.setText('H : Face Detection | P : Autopilot (Enabled)')
            # Disable manual control when autopilot is enabled
            self.disable_manual_control()
            # Hide control buttons
            self.set_buttons_visible(False)
            # Resize window based on the video frame
            #self.resize(300, 600)  # Placeholder size, adjust based on your actual video frame size
            self.detection_window = ObjectDetectionWindow()
            self.detection_window.show()
        else:
            self.status_label.setText('H : Face Detection | P : Autopilot (Disabled)')
            # Re-enable manual control
            self.enable_manual_control()
            # Show control buttons again
            self.set_buttons_visible(True)
            stop_car()
            if self.detection_window:
                self.detection_window.close()
                self.detection_window = None

    def reset_speed(self):
        """Reset car speed after timeout"""
        speed100()
        self.speed_timer.stop()

    def disable_manual_control(self):
        """Disable all manual control buttons."""
        self.set_manual_control_enabled(False)

    def enable_manual_control(self):
        """Enable manual control buttons."""
        self.set_manual_control_enabled(True)

    def set_manual_control_enabled(self, enabled):
        """Enable or disable manual control."""
        # Disable or enable manual control buttons based on the status
        for button in self.findChildren(QPushButton):
            button.setEnabled(enabled)

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.isAutoRepeat():
            return

        if event.key() == Qt.Key_W and self.manual_control_enabled:
            move_forward()
        elif event.key() == Qt.Key_A and self.manual_control_enabled:
            turn_left()
        elif event.key() == Qt.Key_D and self.manual_control_enabled:
            turn_right()
        elif event.key() == Qt.Key_S and self.manual_control_enabled:
            move_backward()
        elif event.key() == Qt.Key_Q and self.manual_control_enabled:
            turn_left2()
        elif event.key() == Qt.Key_E and self.manual_control_enabled:
            turn_right2()

    def keyReleaseEvent(self, event):
        """Handle key release events."""
        if event.isAutoRepeat():
            return

        if event.key() in [Qt.Key_W, Qt.Key_S, Qt.Key_A, Qt.Key_D, Qt.Key_Q, Qt.Key_E]:
            stop_car()  # Stop the car when the key is released

# Run the application
app = QApplication(sys.argv)
window = CarControllerApp()
window.show()

sys.exit(app.exec_())