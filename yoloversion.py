import sys, torch, random, threading, time, os
from urllib import request
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QGridLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QKeyEvent, QFont
from PyQt5.QtCore import Qt, QTimer
import cv2
import numpy as np
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

class App(QWidget):
    ip = "192.168.137.129"
    
    def __init__(self):
        super().__init__()
        self.stream = request.urlopen('http://' + App.ip +':81/stream')
        self.buffer = b''
        request.urlopen('http://' + App.ip + "/action?go=speed80")
        self.initUI()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_detection_enabled = False
        self.autodrive = False
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

    def initUI(self):  
        main_layout = QVBoxLayout()
        self.title_label = QLabel('AI CAR live video', self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 32px; font-weight: bold;")
        main_layout.addWidget(self.title_label)

        self.label = QLabel(self)
        self.label.setFixedSize(400, 400)
        main_layout.addWidget(self.label)

        button_layout = QGridLayout()

        forward_button = QPushButton('Forward', self)
        forward_button.setFixedSize(120, 45)
        forward_button.pressed.connect(self.forward)
        forward_button.released.connect(self.stop)
        button_layout.addWidget(forward_button, 0, 1)

        left_button = QPushButton('Move Left', self)
        left_button.setFixedSize(120, 45)
        left_button.pressed.connect(self.left)
        left_button.released.connect(self.stop)
        button_layout.addWidget(left_button, 1, 0)

        turn_left_button = QPushButton('Turn Left', self)
        turn_left_button.setFixedSize(120, 45)
        turn_left_button.pressed.connect(self.turnleft)
        turn_left_button.released.connect(self.stop)
        button_layout.addWidget(turn_left_button, 0, 0)

        right_button = QPushButton('Move Right', self)
        right_button.setFixedSize(120, 45)
        right_button.pressed.connect(self.right)
        right_button.released.connect(self.stop)
        button_layout.addWidget(right_button, 1, 2)

        turn_right_button = QPushButton('Turn Right', self)
        turn_right_button.setFixedSize(120, 45)
        turn_right_button.pressed.connect(self.turnright)
        turn_right_button.released.connect(self.stop)
        button_layout.addWidget(turn_right_button, 0, 2)

        backward_button = QPushButton('Backward', self)
        backward_button.setFixedSize(120, 45)
        backward_button.pressed.connect(self.backward)
        backward_button.released.connect(self.stop)
        button_layout.addWidget(backward_button, 2, 1)

        speed_layout = QHBoxLayout()
        speed_layout.setSpacing(0)

        speed40_button = QPushButton('Speed 40', self)
        speed40_button.setFixedSize(80, 30)
        speed40_button.pressed.connect(self.speed40)
        speed_layout.addWidget(speed40_button)

        speed60_button = QPushButton('Speed 60', self)
        speed60_button.setFixedSize(80, 30)
        speed60_button.pressed.connect(self.speed60)
        speed_layout.addWidget(speed60_button)

        speed80_button = QPushButton('Speed 80', self)
        speed80_button.setFixedSize(80, 30)
        speed80_button.pressed.connect(self.speed80)
        speed_layout.addWidget(speed80_button)

        speed100_button = QPushButton('Speed 100', self)
        speed100_button.setFixedSize(80, 30)
        speed100_button.pressed.connect(self.speed100)
        speed_layout.addWidget(speed100_button)

        button_layout.addLayout(speed_layout, 3, 0, 1, 3)
        main_layout.addLayout(button_layout)

        self.status_label = QLabel('H : Face Detection | P : Autopilot | Y : YOLO Detection', self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 16px; padding: 10px; border-top: 1px solid #ddd;")
        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)
        self.setWindowTitle('RC Car Controller')
        self.resize(300, 500)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def speed40(self):
        request.urlopen('http://' + App.ip + "/action?go=speed40")
        
    def speed60(self):
        request.urlopen('http://' + App.ip + "/action?go=speed60")
       
    def speed80(self):
        request.urlopen('http://' + App.ip + "/action?go=speed80")
       
    def speed100(self):
        request.urlopen('http://' + App.ip + "/action?go=speed100")
        
    def forward(self):
        request.urlopen('http://' + App.ip + "/action?go=forward")
        
    def backward(self):
        request.urlopen('http://' + App.ip + "/action?go=backward")
        
    def left(self):
        request.urlopen('http://' + App.ip + "/action?go=left")
        
    def right(self):
        request.urlopen('http://' + App.ip + "/action?go=right")
        
    def stop(self):
        request.urlopen('http://' + App.ip + "/action?go=stop")

    def turnleft(self):
        request.urlopen('http://' + App.ip + "/action?go=turn_left")
        
    def turnright(self):
        request.urlopen('http://' + App.ip + "/action?go=turn_right")

    def haaron(self):
        self.face_detection_enabled = not self.face_detection_enabled

    def autoDrive(self):
        self.autodrive = not self.autodrive
        if not self.autodrive:
            self.stop()
            
    def update_frame(self):
        self.buffer += self.stream.read(4096)
        head = self.buffer.find(b'\xff\xd8')
        end = self.buffer.find(b'\xff\xd9')
        try:
            if head > -1 and end > -1:
                jpg = self.buffer[head:end+1]
                self.buffer = self.buffer[end+2:]
                img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                img = cv2.flip(img, 0)
                img = cv2.flip(img, 1)
                
                if self.autodrive:
                    car_state = "go"
                    yolo_state = "go"
                    height, width, _ = img.shape
                    frame = img[height // 2:, :]
                    lower_bound = np.array([0, 0, 0])
                    upper_bound = np.array([255, 255, 80])
                    mask = cv2.inRange(frame, lower_bound, upper_bound)
                    M = cv2.moments(mask)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        cX, cY = 0, 0
                    center_offset = width // 2 - cX
                    cv2.circle(frame, (cX, cY + height // 3), 10, (0, 255, 0), -1)

                    if center_offset > 15:
                        car_state = "right"
                    elif center_offset < -15:
                        car_state = "left"
                    else:
                        car_state = "go"

                    thread_frame = img
                    results = self.model(thread_frame)
                    detections = results.pandas().xyxy[0]

                    if not detections.empty:  
                        for _, detection in detections.iterrows():
                            x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']].astype(int).values           
                            label = detection['name']
                            conf = detection['confidence']

                            if "stop" in label and conf > 0.2:
                                yolo_state = "stop"
                            elif "slow" in label and conf > 0.3:
                                request.urlopen('http://' + App.ip + "/action?go=speed40")
                                yolo_state = "go"

                            color = [0, 0, 0]
                            cv2.rectangle(thread_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(thread_frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    if car_state == "go" and yolo_state =="go":
                        request.urlopen('http://' + App.ip + "/action?go=forward")
                    elif car_state == "right" and yolo_state =="go":
                        request.urlopen('http://' + App.ip + "/action?go=right")
                    elif car_state == "left" and yolo_state =="go":
                        request.urlopen('http://' + App.ip + "/action?go=left")
                    elif yolo_state =="stop":
                        request.urlopen('http://' + App.ip + "/action?go=stop")

                if self.face_detection_enabled:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(50, 50))
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img, "FACE", ((2*x + w - 84) // 2, y-10), cv2.FONT_HERSHEY_PLAIN, 2, 5)

                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                height, width, channels = frame.shape
                bytes_per_line = 3 * width
                q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.label.setPixmap(pixmap)

        except Exception as e:
            print(e)

    def closeEvent(self, event):
        event.accept()

    def keyPressEvent(self, event:QKeyEvent):
        key = event.key()
        if event.isAutoRepeat():
            return
    
        if key == Qt.Key_W:
            self.forward()
        elif key == Qt.Key_S:
            self.backward()
        elif key == Qt.Key_A:
            self.left()
        elif key == Qt.Key_D:
            self.right()
        elif key == Qt.Key_P:
            self.autoDrive()
        elif key == Qt.Key_H:
            self.haaron()
        elif key == Qt.Key_Escape:
            self.close()

    def keyReleaseEvent(self, event: QKeyEvent):
        key = event.key()
        if event.isAutoRepeat():
            return

        if key in [Qt.Key_W, Qt.Key_S, Qt.Key_A, Qt.Key_D]:
            self.stop()

if __name__ == '__main__':
   print(sys.argv)
   app = QApplication(sys.argv)
   view = App()
   view.show()
   sys.exit(app.exec_())