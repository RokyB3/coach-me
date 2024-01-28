import sys

from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QFont, QPainter, QColor, QFontDatabase
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, pyqtSignal
import sys
sys.path.append('src/py/pipeline')
sys.path.append('src/py/exercises')
sys.path.append('src/py/opencv')
from pipeline import getResponseFromInput
from openai import OpenAI
import numpy as np
import cv2
import mediapipe as mp
import sounddevice as sd
import soundfile as sf
import pygame
import math

gpt_exercises = []

BACKGROUND_COLOR = "#71B48D"
PRIMARY_COLOR = "#86CB92"
SECONDARY_COLOR = "#F2F2F2"

a = None
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.started = True
        self.done = False
        self.feedback = None
        self.counter = 0
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
                                     0.7, 0.7)
        self.exercise = None
        self.down = True
        self.up = False
        self.knees_over = False
        self.back_forward = False
        self.low_knee_threshold = 90
        self.high_knee_threshold = 120
    

    def run(self): # displays the camera
        while self.cap.isOpened():
            ret, self.img = self.cap.read()
            pygame.mixer.init()
            if not ret:
                print("Failed to grab frame")
                break

            if gpt_exercises != [] and (self.exercise == None or self.counter >= 2):
                self.exercise = gpt_exercises.pop(0)
                self.counter = 0
                self.feedback = None
                if self.exercise == "lunge":
                    a.startLunge()
                    pygame.mixer.music.load('audio/output/lunges.mp3')
                    pygame.mixer.music.play()
                elif self.exercise == "squat":
                    a.startSquat()
                    pygame.mixer.music.load('audio/output/squats.mp3')
                    pygame.mixer.music.play()
                elif self.exercise == "pullup" or self.exercise == "pull-up":
                    a.startPullup()
                    pygame.mixer.music.load('audio/output/pullups.mp3')
                    pygame.mixer.music.play()
                    self.exercise = "pullup"
                # elif self.exercise == "situp":
                #     pygame.mixer.music.load('audio/output/situp.mp3')
                #     pygame.mixer.music.play()
                else:
                    self.exercise = None
                    self.feedback = None

            if self.exercise == None:
                font = cv2.FONT_HERSHEY_COMPLEX
                # Position (bottom right corner)
                widthHeight=self.img.shape
                bottomRightCornerOfText = (int((1/10)*widthHeight[1]), int((9/10)*widthHeight[0]))
                fontScale = 1
                fontColor = (255, 255, 255)  # White color
                lineType = 4

                cv2.putText(self.img, 'Pick an exercise', 
                            bottomRightCornerOfText, 
                            font, 
                            fontScale,
                            fontColor,
                            lineType) 
            
            self.imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            # self.results = self.detector.process(self.img)
            self.results = self.pose.process(self.imgRGB)
            self.lms = self.get_LM_positions(self.img)
            self.img = self.draw_Pose(self.img)
            self.display_count(self.img) 
            if ret:
                self.change_pixmap_signal.emit(self.img)
            if self.results.pose_landmarks:
                if self.exercise == "lunge": self.handle_lunge()
                elif self.exercise == "squat": self.handle_squat()
                elif self.exercise == "pullup": self.handle_pullup()

            # cv2.imshow("Image", self.img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        self.cap.release()
        cv2.destroyAllWindows()
        return


    def get_LM_positions(self, img):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                #finding height, width of the image printed
                h, w, _ = img.shape
                #Determining the pixels of the landmarks
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
        
        # self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,
        #                                    self.mpPose.POSE_CONNECTIONS)
        return lmList

    def draw_Pose(self, img, draw=True):
        if self.results.pose_landmarks:
            if draw:
                # Landmarks excluding the face
                selected_landmarks = set(range(11, 33))  # This includes shoulders to feet

                # Draw the landmarks
                for id, lm in enumerate(self.results.pose_landmarks.landmark):
                    if id in selected_landmarks:
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 5, (0, 0, 0), cv2.FILLED)

                # Draw the connections
                for connection in self.mpPose.POSE_CONNECTIONS:
                    if connection[0] in selected_landmarks and connection[1] in selected_landmarks:
                        start_landmark = self.results.pose_landmarks.landmark[connection[0]]
                        end_landmark = self.results.pose_landmarks.landmark[connection[1]]
                        start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
                        end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
                        cv2.line(img, start_point, end_point, (255, 0, 0), 2)
        return img

    def calculate_2d_angle(self, p1, p2, p3):

        # Get ladmark positions
        x1, y1 = self.lms[p1][1:]
        x2, y2 = self.lms[p2][1:]
        x3, y3 = self.lms[p3][1:]
        
        # Calculate the angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - 
                             math.atan2(y1-y2, x1-x2))
        
        # Convert to positive angle (Mirror to 180 degrees)
        if angle < 0:
            angle += 360
            if angle > 180:
                angle = 360 - angle
        elif angle > 180:
            angle = 360 - angle
        
        return angle

    def calculate_3d_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        # Create vectors
        vector_ab = b - a
        vector_cb = b - c
        # Dot product
        dot_product = np.dot(vector_ab, vector_cb)
        # Magnitude
        magnitude_ab = np.linalg.norm(vector_ab)
        magnitude_cb = np.linalg.norm(vector_cb)
        # Cos angle
        cos_angle = dot_product / (magnitude_ab * magnitude_cb)

        # Avoiding possible numerical issues with arccos
        cos_angle = np.clip(cos_angle, -1, 1)
# Angle in radians
        angle_radians = np.arccos(cos_angle)

        # Convert to degrees
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees

    def get_angles(self): # get the angles of the joints (knees, hips, shoulders)
        # get the angles of the knees
        self.l_knee_angle = self.calculate_3d_angle(self.l_hi, self.l_k, self.l_a)
        self.r_knee_angle = self.calculate_3d_angle(self.r_hi, self.r_k, self.r_a) 
        self.l_hip_angle = self.calculate_3d_angle(self.l_s, self.l_hi, self.l_k)
        self.r_hip_angle = self.calculate_3d_angle(self.r_s, self.r_hi, self.r_k)
        return
    
    def check_knees(self): # check if the knees are bent
        if self.down and self.l_knee_angle < self.low_knee_threshold and self.r_knee_angle < self.low_knee_threshold:
            if 85 <= self.l_hip_angle <= 100 and 85 <= self.r_hip_angle <= 100:
                print("Correct form")
                self.knees_over = False
                self.back_forward = False

                self.up = True
                self.down = False
            elif self.l_hip_angle < 85 or self.r_hip_angle < 85:
                self.back_forward = True
            elif self.check_toes():
                self.knees_over = True

            self.up = True
            self.down = False

        elif self.up and self.l_knee_angle > 120 and self.r_knee_angle > 120:
            if self.l_hip_angle > 135 and self.r_hip_angle > 135:
                print("Suffienctly straight hips and knees, go down")
                self.up = False
                self.down = True
                if not self.knees_over and not self.back_forward:
                    self.counter += 1
                    print("Squat counter: ", self.counter)

                    # play feedback
                    pygame.mixer.music.load('audio/output/squat_is_good.mp3')
                    pygame.mixer.music.play()
                if self.knees_over:
                    print("Your knees are over your toes")
                    
                    # play feedback
                    pygame.mixer.music.load('audio/output/squat_toes_over.mp3')
                    pygame.mixer.music.play()
                if self.back_forward:
                    print("Back is too forward enough") 

                    # play feedback
                    pygame.mixer.music.load('audio/output/squat_back_forward.mp3')
                    pygame.mixer.music.play()
            
                self.knees_over = False
                self.back_forward = False
        return

    def check_toes(self): # check if the toes are over the knees
        # check if z coordinate of the knees are greater than the z coordinate of the toes plus some threshold
        if  self.l_toe[2] - self.l_k[2] > 2.5 or self.r_toe[2] - self.r_k[2] > 2.5:
        # Angle in radians:
            return True
        else:
            return False
    
    def display_count(self, img):
        if self.exercise == None: return
        font = cv2.FONT_HERSHEY_COMPLEX
        # Position (bottom right corner)
        widthHeight=self.img.shape
        bottomRightCornerOfText = (int((1/10)*widthHeight[1]), int((9/10)*widthHeight[0]))
        fontScale = 1
        fontColor = (255, 255, 255)  # White color
        lineType = 4

        cv2.putText(self.img, 'Count: ' + str(self.counter), 
                    bottomRightCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType) 

    def get_landmarks(self): # gets the landmarks from the camera
        self.results = self.pose.process(self.imgRGB)

        # get the landmarks of the person in the camera
        if self.results.pose_landmarks:
            self.l_s = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_SHOULDER])
            self.l_hi = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_HIP])
            self.l_k = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_KNEE])
            self.l_a = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_ANKLE])
            self.l_e = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_ELBOW])
            self.l_w = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_WRIST])
            self.l_he = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_HEEL])
            self.l_toe = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_FOOT_INDEX])
            self.r_s = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_SHOULDER])
            self.r_hi = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_HIP])
            self.r_k = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_KNEE])
            self.r_a = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_ANKLE])
            self.r_e = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_ELBOW])
            self.r_w = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_WRIST])
            self.r_he = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_HEEL])
            self.r_toe = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_FOOT_INDEX])
            # create a list of all the landmarks
            self.landmark_list = [self.l_s, self.l_hi, self.l_k, self.l_a, self.l_e, self.l_w, self.l_he,
                          self.r_s, self.r_hi, self.r_k, self.r_a, self.r_e, self.r_w, self.r_he]
            return True

    def get_xyz(self, landmark): # gets the x, y, and z coordinates of a landmark
        return [landmark.x, landmark.y, landmark.z]
    
    def get_xy_distance(self, a, b): # gets the distance between two landmarks in the x and y directions
        a = np.array(a)
        b = np.array(b)
        # Create vectors
        vector_ab = b - a
        # Magnitude
        magnitude_ab = np.linalg.norm(vector_ab[0:3:2])
        return magnitude_ab

    def handle_lunge(self):
        # Constants
        min_i_angle, max_i_angle = 75, 90
        min_r_angle, max_r_angle = 80, 100
        min_s_angle = 160
        # max_b_distance = 0.1

        # Angles
        self.l_hip = self.calculate_2d_angle(11, 23, 25) # Hip angle (left)
        self.l_knee = self.calculate_2d_angle(23, 25, 27) # Knee angle (left)
        self.r_hip = self.calculate_2d_angle(12, 24, 26) # Hip angle (right)
        self.r_knee = self.calculate_2d_angle(24, 26, 28) # Knee angle (right)
        
        if self.l_hip == None or self.l_knee == None or self.r_hip == None or self.r_knee == None:
            return

        # Check start position
        if self.started == False or self.done == True:
            if self.l_hip >= min_s_angle and self.r_hip >= min_s_angle:
                print("Start position is correct")
                self.started = True
                self.done = False
                if self.feedback != None:
                    pygame.mixer.music.load(self.feedback)
                    pygame.mixer.music.play()
                    self.feedback = None
                return

        if self.done == False:
            # Check left lunge
            if self.l_hip <= max_r_angle and self.l_hip >= min_r_angle and self.l_knee >= min_i_angle and self.l_knee <= max_i_angle:
                if self.r_hip >= min_s_angle and self.r_knee >= min_r_angle and self.r_knee <= max_r_angle:
                    print("Left lunge is correct")
                    self.feedback = 'audio/output/lunge_is_good.mp3'
                    self.done = True
                    self.counter += 1
                    return

            # Check right lunge
            if self.r_hip <= max_r_angle and self.r_hip >= min_r_angle and self.r_knee >= min_i_angle and self.r_knee <= max_i_angle:
                if self.l_hip >= min_s_angle and self.l_knee >= min_r_angle and self.l_knee <= max_r_angle:
                    print("Right lunge is correct")
                    self.feedback = 'audio/output/lunge_is_good.mp3'
                    self.done = True
                    self.counter += 1
                    return
            
            # Error if left foot too front
            if self.l_hip <= max_r_angle+30 and self.l_hip >= min_r_angle-30 and self.l_knee >= min_i_angle-30 and self.l_knee <= max_i_angle+30:
                if self.r_hip >= min_s_angle-30 and self.r_knee >= min_r_angle-30 and self.r_knee <= max_r_angle+30:
                    print("Left foot is too front")
                    if self.feedback != 'audio/output/lunge_is_good.mp3': self.feedback = 'audio/output/lunge_too_forward.mp3'
                    self.started = False
                    return
            
            # Error if right foot too front
            if self.r_hip <= max_r_angle+30 and self.r_hip >= min_r_angle-30 and self.r_knee >= min_i_angle-30 and self.r_knee <= max_i_angle+30:
                if self.l_hip >= min_s_angle-30 and self.l_knee >= min_r_angle-30 and self.l_knee <= max_r_angle+30:
                    print("Right foot is too front")
                    if self.feedback != 'audio/output/lunge_is_good.mp3': self.feedback = 'audio/output/lunge_too_forward.mp3'
                    self.started = False
                    return
            
    
    def handle_pullup(self):
        # Constants
        max_i_angle = 75
        min_s_angle = 140
        # max_b_distance = 0.1

        # Angles
        self.l_e = self.calculate_2d_angle(12, 14, 16) # Hip angle (left)
        self.l_s = self.calculate_2d_angle(14, 12, 24) # Knee angle (left)
        self.r_e = self.calculate_2d_angle(11, 13, 15) # Hip angle (right)
        self.r_s = self.calculate_2d_angle(13, 11, 23) # Knee angle (right)
        
        if self.l_e == None or self.l_s == None or self.r_e == None or self.r_s == None:
            return

        # Check start position
        if self.started == False or self.done == True:
            if self.l_e >= min_s_angle and self.l_s >= min_s_angle:
                if self.r_e >= min_s_angle and self.r_s >= min_s_angle:
                    print("Start position is correct")
                    self.started = True
                    self.done = False
                    if self.feedback != None:
                        pygame.mixer.music.load(self.feedback)
                        pygame.mixer.music.play()
                        self.feedback = None
                        return

        if self.done == False:
            if self.l_e <= max_i_angle and self.l_s <= max_i_angle:
                if self.r_e <= max_i_angle and self.r_s <= max_i_angle:
                    print("Pullup is correct")
                    self.feedback = 'audio/output/pullup_is_good.mp3'
                    self.done = True
                    self.counter += 1
                    return
            
            # Not high enough
            if self.l_e <= max_i_angle+40 or self.l_s <= max_i_angle+40 or self.r_e <= max_i_angle+40 or self.r_s <= max_i_angle+40:
                self.started = False
                print("Pullup is not high enough")
                if self.feedback != 'audio/output/pullup_is_good.mp3': # Don't overwrite good feedback
                    self.feedback = 'audio/output/pullup_not_high_enough.mp3'
                return
    
    def handle_squat(self): # displays the camera
        self.get_landmarks()
        self.get_angles() # get the angles of the joints (knees, hips, shoulders)
        self.check_knees() # check if the knees are bent
        return
    
class MicrophoneWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.screen = QDesktopWidget().screenGeometry()
        self.setMinimumSize(int(0.7*self.screen.width()/10), int(0.7*self.screen.width()/10))
        self.setMaximumSize(int(0.7*self.screen.width()/10), int(0.7*self.screen.width()/10))
        self.image=QPixmap("assets/microphone.png")
        self.recordingThread=None
        self.recording=False
        self.generating=False
        self.setCursor(Qt.PointingHandCursor)
        self.hovering=False
        self.setStyleSheet("""
                border-radius: 5px; 
                border: 6px solid #FEFEFE;
                background: #86CB92;
        """)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)  # Enable antialiasing

        # Get the dimensions of the widget
        widget_width = float(0.7*self.screen.width()/10)
        widget_height = float(0.7*self.screen.width()/10)

        # Set the circle's properties
        circle_diameter = min(widget_width, widget_height) - 20  # Adjust for padding
        circle_x = (widget_width - circle_diameter) / 2
        circle_y = (widget_height - circle_diameter) / 2

        color = None
        if self.generating:
            color=QColor(255,255, 0)
        elif (not self.recording) and not self.hovering:
            color=QColor(255,0,0)
        elif (not self.recording) and self.hovering:
            color=QColor(230,0,0)
        else:
            color=QColor(0, 255, 0)
        painter.setBrush(color)  # Green if recording, red otherwise
        painter.drawEllipse(int(circle_x), int(circle_y), int(circle_diameter), int(circle_diameter)) 
        
        image_width = int(min(widget_width, widget_height) / 2)
        image_height = int(min(widget_width, widget_height) / 2)
        image_x = int((widget_width - image_width) / 2)
        image_y = int((widget_height - image_height) / 2)
        painter.drawPixmap(image_x, image_y, int(image_width), int(image_width), self.image)
        
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if not(self.generating or self.recording):
                self.recording = not self.recording  # Toggle recording status
                if self.recording:
                    self.recordingThread = RecordingThread(parent=self)
                    self.recordingThread.finished.connect(self.onThreadEnded)
                    self.recordingThread.start()
                else:
                    if self.recordingThread:
                        self.recordingThread.stop()

            self.update()

    def enterEvent(self,event):
        self.hovering=True
        self.update()
        
    def leaveEvent(self,event):
        self.hovering=False
        self.update()
        
    def onThreadEnded(self):
        self.update()
        
class RecordingThread(QThread):
    
    finished=pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.recording=True
        self.microphoneWidget=parent
    def run(self):
        input_audio = 'audio/input/input.wav'

        duration = 5

        audio_data = sd.rec(int(duration * 44100), samplerate=44100, channels=2, dtype='int16')
        sd.wait() 

        sf.write(input_audio, audio_data, samplerate=44100)

        self.microphoneWidget.recording=False
        self.microphoneWidget.generating=True
        self.microphoneWidget.update()
        text = getResponseFromInput("input.wav")
        
        output_audio = 'audio/output/prompt-output.mp3'

        self.play_audio(output_audio)
        self.microphoneWidget.generating=False
        self.microphoneWidget.update()

        print(text)
        for ex in ['lunge', 'squat', 'pullup', 'pull-up']:
            if ex in text.lower():
                gpt_exercises.append(ex)
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
                min-height:100%;
            }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton:pressed { background-color: #1f618d; }
        """)
        font=QFont()
        font.setFamily("Helvetica")
        font.setPointSize(24)
        font.setBold(True) 
        self.setFont(font)
        self.setCursor(Qt.PointingHandCursor)
            
class App(QWidget):
    def __init__(self):
        super().__init__() 
        self.setWindowTitle("Coach.me")
        self.display_width = int(1000)
        self.display_height = int(700)
        self.showMaximized()
        self.setStyleSheet("background-color: {};".format(BACKGROUND_COLOR))

        # create a text label
        self.textLabel = QLabel('- - - - C o a c h . m e - - - -')
        self.textLabel.setAlignment(Qt.AlignCenter)
        font = QFont('Yu Gothic', 32)
        self.textLabel.setFont(font)
        self.textLabel.setStyleSheet("""
            color: black;
            font-weight: bold; 
            text-transform:uppercase;
        """)

        self.button = QPushButton('Start', self)
        microphonewidgethbox=QHBoxLayout()
        self.microphoneWidget=MicrophoneWidget()

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.textLabel)

        hbox = QHBoxLayout()
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.setStyleSheet("""
            border: 6px solid white;
            border-radius: 16px;
        """)
        hbox.addWidget(self.image_label)
        buttonvbox= QVBoxLayout()
        hbox.addLayout(buttonvbox)
        
        lungeButton=ExerciseButton("- - - - Lunges - - - -")
        buttonvbox.addWidget(lungeButton)
        squatButton=ExerciseButton("- - - - Squats - - - -")
        buttonvbox.addWidget(squatButton)
        pullupButton=ExerciseButton("- - - - Pull-Ups - - - -")
        buttonvbox.addWidget(pullupButton)
        # situpButton=ExerciseButton("- - - - Planks - - - -")
        # buttonvbox.addWidget(situpButton)
        
        buttonvbox.addLayout(microphonewidgethbox)
        microphonewidgethbox.addWidget(self.microphoneWidget, alignment=Qt.AlignCenter)
        instructionLabel=QLabel("Press to record, speak while button is green.")
        instructionLabel.setWordWrap(True)
        instructionLabel.setStyleSheet("font-weight:bold; color:red; font-size:48px; text-align:center;")
        instructionLabel.setAlignment(Qt.AlignCenter)
        buttonvbox.addWidget(instructionLabel, alignment=Qt.AlignCenter)
        vbox.addLayout(hbox)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()
        # self.thread.display_camera()
        lungeButton.clicked.connect(self.startLunge)
        squatButton.clicked.connect(self.startSquat)
        pullupButton.clicked.connect(self.startPullup)
        self.buttons = [lungeButton, squatButton, pullupButton]
    
    def startLunge(self):
        self.thread.exerciseSelected=True
        self.thread.exercise = "lunge"
        self.thread.counter = 0
        for button in self.buttons:
            self.resetExerciseButtonColors(button)
        self.setExerciseButtonColor(self.buttons[0])
    
    def startSquat(self):
        self.thread.exerciseSelected=True
        self.thread.exercise = "squat"
        self.thread.counter = 0
        for button in self.buttons:
            self.resetExerciseButtonColors(button)
        self.setExerciseButtonColor(self.buttons[1])
    
    def startPullup(self):
        self.thread.exerciseSelected=True
        self.thread.exercise = "pullup"
        self.thread.counter = 0
        for button in self.buttons:
            self.resetExerciseButtonColors(button)
        self.setExerciseButtonColor(self.buttons[2])

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
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
    def resetExerciseButtonColors(self, button):
        button.setStyleSheet("""
            QPushButton {
                border-radius: 5px; 
                border: 6px solid #FEFEFE;
                background: #86CB92;
                min-height:100%;
            }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton:pressed { background-color: #1f618d; }
        """)
    
    def setExerciseButtonColor(self, button):
        button.setStyleSheet("""
            QPushButton {
                border-radius: 5px; 
                border: 6px solid #FEFEFE;
                background: #2980b9;
                min-height:100%;
            }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton:pressed { background-color: #1f618d; }
        """)

if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())