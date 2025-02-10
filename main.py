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
import pathlib
from queue import Queue
from threading import Lock
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath  # Fix for Windows path issues

# Global setup and initialization
ip = '192.168.137.129'
stream = urlopen(f'http://{ip}:81/stream')
buffer = b''
car_state = 'stop'

def send_car_command(action):
    try:
        urlopen(f'http://{ip}/action?go={action}')
        print(f"Car action: {action}")
    except Exception as e:
        print(f"Network error: {e}")

# Car control functions
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

def exit_program():
    print("Exiting program...")
    urlopen(f'http://{ip}/action?go=stop')
    cv2.destroyAllWindows()
    app.quit()

# Video stream handling thread
class VideoStreamThread(QThread):
    new_frame_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.frame_counter = 0
        self.processing = True
        self.frame_skip = 3
        self.buffer_size = 8192
        
    def run(self):
        global buffer
        try:
            last_frame_time = time.time()
            min_frame_interval = 1.0 / 30
            
            while self.processing:
                current_time = time.time()
                if current_time - last_frame_time < min_frame_interval:
                    continue
                
                buffer += stream.read(self.buffer_size)
                head = buffer.find(b'\xff\xd8')
                end = buffer.find(b'\xff\xd9')

                if head > -1 and end > -1:
                    jpg = buffer[head:end + 2]
                    buffer = buffer[end + 2:]
                    
                    self.frame_counter += 1
                    if self.frame_counter % self.frame_skip != 0:
                        continue

                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), 
                                     cv2.IMREAD_REDUCED_COLOR_2)
                    if img is not None:
                        img = cv2.flip(img, 0)
                        self.new_frame_signal.emit(img)

        except Exception as e:
            print(f"Error in video stream: {e}")

    def stop(self):
        self.processing = False

# Object detection window for displaying processed frames
class ObjectDetectionWindow(QMainWindow):
    def __init__(self, title="Object Detection"):
        super().__init__()
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 400, 400)
        self.detection_label = QLabel(self)
        self.detection_label.setGeometry(0, 0, 400, 400)
        
    def update_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_frame.shape
        q_image = QImage(rgb_frame.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio)
        self.detection_label.setPixmap(scaled_pixmap)

class YOLODetectionThread(QThread):
    detection_signal = pyqtSignal(object)  # Signal to emit detection results
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.frame_queue = Queue(maxsize=2)  # Only keep latest 2 frames
        self.running = True
        self.detection_cooldown = 2
        self.last_detection_time = 0
        self.lock = Lock()
    
    def add_frame(self, frame):
        # Non-blocking frame addition - drop frame if queue is full
        if not self.frame_queue.full():
            self.frame_queue.put(frame)
    
    def run(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                current_time = time.time()
                
                if current_time - self.last_detection_time >= self.detection_cooldown:
                    try:
                        with self.lock:
                            results = self.model(frame)
                            detections = results.pandas().xyxy[0]
                            
                            if not detections.empty:
                                detection_results = []
                                for _, detection in detections.iterrows():
                                    if detection['confidence'] > 0.3:
                                        detection_results.append({
                                            'bbox': detection[['xmin', 'ymin', 'xmax', 'ymax']].astype(int).values,
                                            'label': detection['name'],
                                            'confidence': detection['confidence']
                                        })
                                
                                if detection_results:
                                    self.detection_signal.emit({
                                        'frame': frame,
                                        'detections': detection_results,
                                        'timestamp': current_time
                                    })
                                    self.last_detection_time = current_time
                    
                    except Exception as e:
                        print(f"YOLO detection error: {e}")
            
            # Small sleep to prevent CPU overuse
            time.sleep(0.01)
    
    def stop(self):
        self.running = False
        self.wait()

# Main application window
class CarControllerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RC Car Controller")
        
        self.autopilot_enabled = False
        self.haar_detection_enabled = False
        self.manual_control_enabled = True
        self.window_resized = False
        self.yolo_detection_enabled = False
        
        self.haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
            self.model.conf = 0.3
            self.model.iou = 0.45
            self.model.classes = None
            self.model.max_det = 5
            self.model.agnostic = True
            self.model.half()
            print("YOLO model loaded successfully")
            
            # Initialize YOLO detection thread
            self.yolo_thread = YOLODetectionThread(self.model)
            self.yolo_thread.detection_signal.connect(self.handle_yolo_detection)
            self.yolo_thread.start()
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None
            self.yolo_thread = None
        
        self.autopilot_window = None
        self.yolo_window = None
        self.speed_timer = QTimer()
        self.speed_timer.timeout.connect(self.reset_speed)
        self.last_detection_time = 0
        self.detection_cooldown = 2

        self.setup_ui()
        self.setup_video_thread()

    def setup_ui(self):
        main_layout = QVBoxLayout()
        self.title_label = QLabel('AI CAR live video', self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 32px; font-weight: bold;")
        main_layout.addWidget(self.title_label)

        self.video_label = QLabel(self)
        main_layout.addWidget(self.video_label)

        button_layout = QGridLayout()

        forward_button = QPushButton("Forward")
        forward_button.setFixedSize(120, 45)
        forward_button.pressed.connect(move_forward)
        forward_button.released.connect(stop_car)
        button_layout.addWidget(forward_button, 0, 1)

        left_button = QPushButton("Move Left")
        left_button.setFixedSize(120, 45)
        left_button.pressed.connect(turn_left)
        left_button.released.connect(stop_car)
        button_layout.addWidget(left_button, 1, 0)

        left_button2 = QPushButton("Turn Left")
        left_button2.setFixedSize(120, 45)
        left_button2.pressed.connect(turn_left2)
        left_button2.released.connect(stop_car)
        button_layout.addWidget(left_button2, 0, 0)

        right_button = QPushButton("Move Right")
        right_button.setFixedSize(120, 45)
        right_button.pressed.connect(turn_right)
        right_button.released.connect(stop_car)
        button_layout.addWidget(right_button, 1, 2)

        right_button2 = QPushButton("Turn Right")
        right_button2.setFixedSize(120, 45)
        right_button2.pressed.connect(turn_right2)
        right_button2.released.connect(stop_car)
        button_layout.addWidget(right_button2, 0, 2)

        backward_button = QPushButton("Backward")
        backward_button.setFixedSize(120, 45)
        backward_button.pressed.connect(move_backward)
        backward_button.released.connect(stop_car)
        button_layout.addWidget(backward_button, 2, 1)

        speed_layout = QHBoxLayout()
        speed_layout.setSpacing(0)

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

        button_layout.addLayout(speed_layout, 3, 0, 1, 3)
        main_layout.addLayout(button_layout)

        self.status_label = QLabel('H : Face Detection | P : Autopilot | Y : YOLO Detection', self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 16px; padding: 10px; border-top: 1px solid #ddd;")
        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)
        self.resize(300, 500)

        QShortcut(Qt.Key_H, self, self.toggle_face_detection)
        QShortcut(Qt.Key_P, self, self.toggle_autopilot)
        QShortcut(Qt.Key_Y, self, self.toggle_yolo_detection)

    def setup_video_thread(self):
        self.video_thread = VideoStreamThread()
        self.video_thread.new_frame_signal.connect(self.update_video_frame)
        self.video_thread.start()
        self.frame_counter = 0
        self.last_processed_time = time.time()
        self.processing_queue = []
        self.max_queue_size = 2

    def update_video_frame(self, frame):
        try:
            if frame is None or frame.size == 0:
                print("Invalid frame received, skipping...")
                return
            
            if not hasattr(self, 'consecutive_errors'):
                self.consecutive_errors = 0
            
            if self.consecutive_errors > 5:
                print("Too many errors, resetting video stream...")
                self.video_thread.stop()
                time.sleep(1)
                self.video_thread.start()
                self.consecutive_errors = 0
        except Exception as e:
            self.consecutive_errors += 1
            print(f"Error processing frame: {e}")

        if self.autopilot_enabled:
            height, width, _ = frame.shape
            roi_height = height // 2
            roi_y = height - roi_height
            roi = frame[roi_y:height, 0:width].copy()
            processed_frame = cv2.resize(roi, (320, 240))
            processed_frame = processed_frame.copy()
            height, width, _ = processed_frame.shape
            bottom_half = processed_frame[height//2:, :]
            gray = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
            mask = np.zeros_like(thresh)
            height, width = thresh.shape
            roi_points = np.array([
                [(0, height), (width//4, height//2), 
                 (3*width//4, height//2), (width, height)]], dtype=np.int32)
            cv2.fillPoly(mask, roi_points, 255)
            combined = cv2.bitwise_and(thresh, edges, mask=mask)
            
            # Send frame to YOLO thread if enabled
            if self.yolo_detection_enabled and self.yolo_thread is not None:
                self.yolo_thread.add_frame(processed_frame)
            
            M = cv2.moments(combined)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center_offset = width // 2 - cX
                
                if time.time() - self.last_detection_time >= self.detection_cooldown:
                    if center_offset > 15:
                        turn_right()
                    elif center_offset < -15:
                        turn_left()
                    else:
                        move_forward()
                    
                cv2.circle(processed_frame, (cX, cY + height // 2), 10, (0, 255, 0), -1)
            
            if self.autopilot_window:
                self.autopilot_window.update_frame(processed_frame)
            
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb_frame.shape
            q_image = QImage(rgb_frame.data, w, h, w * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.video_label.setPixmap(pixmap)
            
        else:
            if self.haar_detection_enabled:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 150), 2)
                    cv2.putText(frame, "Face", (x + w//2 - 20, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155, 50, 190), 2)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb_frame.shape
            q_image = QImage(rgb_frame.data, w, h, w * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.video_label.setPixmap(pixmap)

    def toggle_face_detection(self):
        self.haar_detection_enabled = not self.haar_detection_enabled
        if self.haar_detection_enabled:
            self.status_label.setText('H : Face Detection (Enabled) | P : Autopilot | Y : YOLO Detection')
        else:
            self.status_label.setText('H : Face Detection (Disabled) | P : Autopilot | Y : YOLO Detection')

    def toggle_autopilot(self):
        self.autopilot_enabled = not self.autopilot_enabled
        self.manual_control_enabled = not self.autopilot_enabled

        if self.autopilot_enabled:
            self.status_label.setText('H : Face Detection | P : Autopilot (Enabled) | Y : YOLO Detection')
            self.disable_manual_control()
            self.set_buttons_visible(False)
            self.autopilot_window = ObjectDetectionWindow("Autopilot Lane Detection")
            self.autopilot_window.show()
            # Position the autopilot window
            self.autopilot_window.move(100, 100)
        else:
            self.status_label.setText('H : Face Detection | P : Autopilot (Disabled) | Y : YOLO Detection')
            self.enable_manual_control()
            self.set_buttons_visible(True)
            stop_car()
            if self.autopilot_window:
                self.autopilot_window.close()
                self.autopilot_window = None

    def toggle_yolo_detection(self):
        self.yolo_detection_enabled = not self.yolo_detection_enabled
        status = "Enabled" if self.yolo_detection_enabled else "Disabled"
        self.status_label.setText(f'H : Face Detection | P : Autopilot | Y : YOLO Detection ({status})')
        
        if self.yolo_detection_enabled:
            self.yolo_window = ObjectDetectionWindow("YOLO Sign Detection")
            self.yolo_window.show()
            # Position the YOLO window to the right of autopilot window
            self.yolo_window.move(520, 100)
        else:
            if self.yolo_window:
                self.yolo_window.close()
                self.yolo_window = None

    def reset_speed(self):
        if self.autopilot_enabled:
            speed80()
        self.speed_timer.stop()

    def disable_manual_control(self):
        self.set_manual_control_enabled(False)

    def enable_manual_control(self):
        self.set_manual_control_enabled(True)

    def set_manual_control_enabled(self, enabled):
        for button in self.findChildren(QPushButton):
            button.setEnabled(enabled)

    def set_buttons_visible(self, visible):
        for button in self.findChildren(QPushButton):
            button.setVisible(visible)

    def keyPressEvent(self, event):
        if event.isAutoRepeat() or not self.manual_control_enabled:
            return

        if event.key() == Qt.Key_W:
            move_forward()
        elif event.key() == Qt.Key_A:
            turn_left()
        elif event.key() == Qt.Key_D:
            turn_right()
        elif event.key() == Qt.Key_S:
            move_backward()
        elif event.key() == Qt.Key_Q:
            turn_left2()
        elif event.key() == Qt.Key_E:
            turn_right2()
        elif event.key() == Qt.Key_Escape:
            self.close()

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat() or not self.manual_control_enabled:
            return

        if event.key() in [Qt.Key_W, Qt.Key_S, Qt.Key_A, Qt.Key_D, Qt.Key_Q, Qt.Key_E]:
            stop_car()

    def closeEvent(self, event):
        stop_car()
        if self.autopilot_window:
            self.autopilot_window.close()
        if self.yolo_window:
            self.yolo_window.close()
        if self.speed_timer.isActive():
            self.speed_timer.stop()
        if self.yolo_thread is not None:
            self.yolo_thread.stop()
        self.video_thread.stop()
        self.video_thread.wait()
        event.accept()

    def handle_yolo_detection(self, result):
        if not self.yolo_detection_enabled:
            return
            
        frame = result['frame']
        detections = result['detections']
        current_time = result['timestamp']
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            label = detection['label']
            conf = detection['confidence']
            
            if self.autopilot_enabled:
                if "stop" in label:
                    stop_car()
                    self.last_detection_time = current_time
                    self.speed_timer.start(3000)
                elif "slow" in label:
                    speed40()
                    self.last_detection_time = current_time
                    self.speed_timer.start(5000)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', 
                      (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.5, (0, 255, 0), 2)
        
        if self.yolo_window:
            self.yolo_window.update_frame(frame)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CarControllerApp()
    window.show()
    sys.exit(app.exec_())