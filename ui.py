import sys

from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QFont, QPainter, QColor
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, pyqtSignal
import sys
sys.path.append('src/py/pipeline')
from pipeline import getResponseFromInput
import numpy as np
import cv2
import mediapipe as mp
import tkinter as tk
import sounddevice as sd
import soundfile as sf
import pygame

BACKGROUND_COLOR = "#71B48D"
PRIMARY_COLOR = "#86CB92"
SECONDARY_COLOR = "#F2F2F2"

root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        #self.detector = mp.pose.Pose()
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.results = None
        self.img = None
        self.imgRGB = None
        self.lmList = None
        self.pose = None
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(False, 1, True,
                                     False, True,
                                     0.5, 0.5)

    def run(self):
        # capture from web cam
        while self.cap.isOpened():
            ret, self.img = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            self.imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

            self.img = self.find_Pose(self.img)

            if ret:
                self.change_pixmap_signal.emit(self.img)

        self.cap.release()
        cv2.destroyAllWindows()
    
    def find_Pose(self, img, draw=True):
        self.results = self.pose.process(self.imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
                
        return img
    
    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle

    def handle_lunge(self):
        self.l_s = self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_SHOULDER]
        self.l_hi = self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_HIP]
        self.l_k = self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_KNEE]
        self.l_a = self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_ANKLE]
        self.r_s = self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_SHOULDER]
        self.r_hi = self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_HIP]
        self.r_k = self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_KNEE]
        self.r_a = self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_ANKLE]
        
class MicrophoneWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)  # Set minimum size for the widget
        self.image=QPixmap("assets/microphone.png")
        self.recordingThread=None
        self.recording=False

        self.screen = QDesktopWidget().screenGeometry()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)  # Enable antialiasing

        # Get the dimensions of the widget
        widget_width = float(self.screen.width()/10)
        widget_height = float(self.screen.height()/10)

        # Set the circle's properties
        circle_diameter = min(widget_width, widget_height) - 20  # Adjust for padding
        circle_x = (widget_width - circle_diameter) / 2
        circle_y = (widget_height - circle_diameter) / 2

        image_width = int(min(widget_width, widget_height) / 2)
        image_height = int(min(widget_width, widget_height) / 2)
        image_x = int((widget_width - image_width) / 2)
        image_y = int((widget_height - image_height) / 2)
        painter.drawPixmap(image_x, image_y, image_width, image_height, self.image)
        
        color = QColor(0, 255, 0) if self.recording else QColor(255, 0, 0)
        painter.setBrush(color)  # Green if recording, red otherwise
        painter.drawEllipse(int(circle_x), int(circle_y), int(circle_diameter), int(circle_diameter))

        self.setCursor(Qt.PointingHandCursor)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.recording = not self.recording  # Toggle recording status
            if self.recording:
                self.recordingThread = RecordingThread(self)
                self.recordingThread.finished.connect(self.onThreadEnded)
                self.recordingThread.start()
            else:
                if self.recordingThread:
                    self.recordingThread.stop()

        self.update()
        
    def onThreadEnded(self):
        print("test")
        
class RecordingThread(QThread):
    
    finished=pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.recording=True
        
    def run(self):
        input_audio = 'audio/input/input.wav'

        duration = 6 

        audio_data = sd.rec(int(duration * 44100), samplerate=44100, channels=2, dtype='int16')
        sd.wait() 

        sf.write(input_audio, audio_data, samplerate=44100)

        getResponseFromInput("input.wav")
        
        output_audio = 'audio/output/prompt-output.mp3'
        
        self.play_audio(output_audio)
        self.finished.emit()
    
    def stop(self):
        print("recording stopped")
        self.recording=False 

    def play_audio(self,filename):
        pygame.init()

        pygame.mixer.init()

        pygame.mixer.music.load(filename)

        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  

        pygame.mixer.quit()
        pygame.quit()
        
class ExerciseButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.setStyleSheet("""
            QPushButton {
                border-radius: 5px; 
                border: 6px solid #FEFEFE;
                background: #86CB92;
                min-height:100px;
            }
        """)
        self.setCursor(Qt.PointingHandCursor)
        self.exerciseName=text
    def mousePressEvent(self, event):
        if event.button() == 1:  # Left mouse button
            print(self.exerciseName)
            
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Coach.me")
        self.disply_width = screen_width
        self.display_height = screen_height
        self.showMaximized()
        self.setStyleSheet("background-color: {};".format(BACKGROUND_COLOR))

        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        self.image_label.setStyleSheet("""
            border: 6px solid white;
            border-radius: 16px;
        """)

        # create a text label
        self.textLabel = QLabel('Coach.me')
        font = QFont('Inter', 48)
        self.textLabel.setFont(font)
        self.textLabel.setStyleSheet("""
            color: white;
            font-weight: bold;
        """)

        self.button = QPushButton('Start', self)
        self.microphoneWidget=MicrophoneWidget()

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.textLabel)

        hbox = QHBoxLayout()
        hbox.addWidget(self.image_label)
        buttonvbox= QVBoxLayout()
        hbox.addLayout(buttonvbox)
        
        lungeButton=ExerciseButton("Lunges")
        buttonvbox.addWidget(lungeButton)
        squatButton=ExerciseButton("Squats")
        buttonvbox.addWidget(squatButton)
        pullupButton=ExerciseButton("Pull-Ups")
        buttonvbox.addWidget(pullupButton)
        pushupButton=ExerciseButton("Push-Ups")
        buttonvbox.addWidget(pushupButton)
        situpButton=ExerciseButton("Sit-Ups")
        buttonvbox.addWidget(situpButton)
        
        microphonewidgethbox=QHBoxLayout()
        buttonvbox.addLayout(microphonewidgethbox)
        microphonewidgethbox.addWidget(self.microphoneWidget, alignment=Qt.AlignCenter)
            
        vbox.addLayout(hbox)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

        
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())