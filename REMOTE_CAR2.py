
import sys, torch, random, threading, time, os
from urllib import request
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QGridLayout
from PyQt5.QtGui import QPixmap, QImage, QKeyEvent, QFont
from PyQt5.QtCore import Qt, QTimer
import cv2
import numpy as np
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

class App(QWidget):
    ip = "192.168.137.227"
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
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setGeometry(0, 0, 640, 480)
        self.setWindowTitle('OpenCV x PyQt')

        self.text_label = QLabel(self)
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setGeometry(0, 0, 640, 480)
        self.text_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.text_label.setText("DADUINO AI CAR")

        btn1 = QPushButton('Speed 40', self)
        btn1.resize(60, 30)
        btn1.pressed.connect(self.speed40)
        
        btn2 = QPushButton('Speed 60', self)
        btn2.resize(60, 30)
        btn2.pressed.connect(self.speed60)

        btn3 = QPushButton('Speed 80', self)
        btn3.resize(60, 30)
        btn3.pressed.connect(self.speed80)

        btn4 = QPushButton('Speed 100', self)
        btn4.resize(60, 30)
        btn4.pressed.connect(self.speed100)

        btn5 = QPushButton('Forward(press W)', self)
        btn5.resize(60, 30)
        btn5.pressed.connect(self.forward)
        btn5.released.connect(self.stop)

        btn6 = QPushButton('Backward(press S)', self)
        btn6.resize(60, 30)
        btn6.pressed.connect(self.backward)
        btn6.released.connect(self.stop)

        btn7 = QPushButton('Left(Press A)', self)
        btn7.resize(60, 30)
        btn7.pressed.connect(self.left)
        btn7.released.connect(self.stop)

        btn8 = QPushButton('Right(Press D)', self)
        btn8.resize(60, 30)
        btn8.pressed.connect(self.right)
        btn8.released.connect(self.stop)

        btn9 = QPushButton('Stop', self)
        btn9.resize(60, 30)
        btn9.pressed.connect(self.stop)      

        btn10 = QPushButton('Turn Left', self)
        btn10.resize(60, 30)
        btn10.pressed.connect(self.turnleft)
        btn10.released.connect(self.stop)

        btn11 = QPushButton('Turn Right', self)
        btn11.resize(60, 30)
        btn11.pressed.connect(self.turnright)
        btn11.released.connect(self.stop)

        btn12 = QPushButton("Face", self)
        btn12.resize(60, 30)
        btn12.clicked.connect(self.haaron)

        btn13 = QPushButton("Auto drive", self)
        btn13.resize(60, 30)
        btn13.clicked.connect(self.autoDrive)
       
        grid = QGridLayout()
        grid.addWidget(btn1, 0, 0)
        grid.addWidget(btn2, 1, 0)
        grid.addWidget(btn3, 2, 0)
        grid.addWidget(btn4, 3, 0)
        grid.addWidget(btn5, 0, 2)
        grid.addWidget(btn6, 2, 2) 
        grid.addWidget(btn7, 1, 1) 
        grid.addWidget(btn8, 1, 3) 
        grid.addWidget(btn9, 1, 2) 
        grid.addWidget(btn10, 0, 4) 
        grid.addWidget(btn11, 1, 4)
        grid.addWidget(btn12, 2, 4) 
        grid.addWidget(btn13, 3, 4) 

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.text_label)
        self.layout.addWidget(self.label)       
        self.layout.addLayout(grid)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)


    def speed40(self) :
        request.urlopen('http://' + App.ip + "/action?go=speed40")
        
    def speed60(self) :
        request.urlopen('http://' + App.ip + "/action?go=speed60")
       
    def speed80(self) :
        request.urlopen('http://' + App.ip + "/action?go=speed80")
       
    def speed100(self) :
        request.urlopen('http://' + App.ip + "/action?go=speed100")
        
    def forward(self) :
        request.urlopen('http://' + App.ip + "/action?go=forward")
        
    def backward(self) :
        request.urlopen('http://' + App.ip + "/action?go=backward")
        
    def left(self) :
        request.urlopen('http://' + App.ip + "/action?go=left")
        
    def right(self) :
        request.urlopen('http://' + App.ip + "/action?go=right")
        
    def stop(self) :
        request.urlopen('http://' + App.ip + "/action?go=stop")

    def turnleft(self) :
        request.urlopen('http://' + App.ip + "/action?go=turn_left")
        
    def turnright(self) :
        request.urlopen('http://' + App.ip + "/action?go=turn_right")

    def haaron(self) :
        self.face_detection_enabled = not self.face_detection_enabled

    def autoDrive(self) :
        self.autodrive = not self.autodrive
        if not self.autodrive :
            self.stop()
            
    def update_frame(self) :
        self.buffer += self.stream.read(4096)
        head = self.buffer.find(b'\xff\xd8')
        end = self.buffer.find(b'\xff\xd9')
        try :
            if head > -1 and end > -1 :
                jpg = self.buffer[head : end+1]
                self.buffer = self.buffer[end+2 :]
                img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                img = cv2.flip(img, 0)
                img = cv2.flip(img, -1)
                
                if self.autodrive :
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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
                    # cv2.imshow("title", mask) 트랙킹 확인

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

                            if "stop" in label and conf > 0.3:
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
                


                if self.face_detection_enabled :
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(50, 50))
                    for (x, y, w, h) in faces :
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img, "FACE", ((2*x + w - 84) // 2, y-10), cv2.FONT_HERSHEY_PLAIN, 2, 5)


                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                height, width, channels = frame.shape
                bytes_per_line = 3 * width
                q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                

                pixmap = QPixmap.fromImage(q_img)
                self.label.setPixmap(pixmap)

        except Exception as e :
            print(e)
            
        


    def closeEvent(self, event) :
        event.accept()

    def keyPressEvent(self, event:QKeyEvent) :
        key = event.key()
        if event.isAutoRepeat() :
            return
    
        if key == Qt.Key_W :
            self.forward()
        elif key == Qt.Key_S :
            self.backward()
        elif key == Qt.Key_A :
            self.left()
        elif key == Qt.Key_D :
            self.right()
        elif key == Qt.Key_Escape :
            self.close()

    def keyReleaseEvent(self, event: QKeyEvent) :
        key = event.key()
        if event.isAutoRepeat() :
            return

        if key in [Qt.Key_W, Qt.Key_S, Qt.Key_A, Qt.Key_D] :
            self.stop()

if __name__ == '__main__':
   print(sys.argv)
   app = QApplication(sys.argv)
   view = App()
   view.show()
   sys.exit(app.exec_())