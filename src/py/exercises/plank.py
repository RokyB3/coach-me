import cv2
import mediapipe as mp
import numpy as np
import math
import pygame

class camera: # class for the camera, so that we can use it to display the camera and get all the joints from it
    def __init__(self):
        self.start = True
        self.done = False
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
        self.time_incorrect = 0
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(False, 1, True,
                                     False, True,
                                     0.7, 0.7)

        self.feedback = None

        # bool variables
        self.correct_shoulder = False
        self.correct_elbow = False
        self.time_incorrect_shoulder = 0
        self.time_incorrect_elbow = 0
        self.time_incorrect = 0

    def display_camera(self): # displays the camera
        while self.cap.isOpened():
            ret, self.img = self.cap.read()

            pygame.mixer.init()
            if not ret:
                print("Failed to grab frame")
                break
            self.imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            # self.results = self.detector.process(self.img)
            self.results = self.pose.process(self.img)
            self.lms = self.get_LM_positions(self.img)
            if self.results.pose_landmarks:
                self.handle_plank()

            cv2.imshow("Image", self.img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
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
        
        self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return lmList


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

    def handle_plank(self):
        # Constants
        min_i_angle, max_i_angle = 75, 90
        min_r_angle, max_r_angle = 80, 100
        min_s_angle = 160

        # Angles
        self.l_e_angle = self.calculate_2d_angle(12, 14, 16) # elbow angle (left)
        self.l_s_angle = self.calculate_2d_angle(14, 12, 24) # shoulder angle (left)
        self.r_e_angle = self.calculate_2d_angle(11, 13, 15) # elbow angle (right)
        self.r_s_angle = self.calculate_2d_angle(13, 11, 23) # shoulder angle (right)


        # Check if the user is in the initial position
        self.check_shoulder()
        self.check_elbows()
        if self.correct_shoulder and self.correct_elbow:
            print("correct")
            self.time_incorrect -= 1
            if self.time_incorrect < 0:
                self.time_incorrect = 0
        elif self.time_incorrect > 50:
            print("Incorrect")
            if self.time_incorrect_elbow > self.time_incorrect_shoulder:
                print(" Elbows")
            else:
                print(" Shoulders")

            self.time_incorrect = 0

        return 
    
    def check_shoulder(self): # check the angle of the shoulders
        # Constants
        min_angle = 85
        max_angle = 115
        if min_angle <= self.l_s_angle <= max_angle and min_angle <= self.r_s_angle <= max_angle:
            self.correct_shoulder = True
            self.time_incorrect_shoulder -= 1
            if self.time_incorrect_shoulder < 0:
                self.time_incorrect_shoulder = 0
        elif self.l_s_angle < min_angle or self.l_s_angle > max_angle:
            self.correct_shoulder = False
            self.time_incorrect += 1
            if self.time_incorrect > 100:
                self.time_incorrect = 100
            self.time_incorrect_shoulder += 1
            if self.time_incorrect_shoulder > 100:
                self.time_incorrect_shoulder = 100
        elif self.r_s_angle < min_angle or self.r_s_angle > max_angle:
            self.correct_shoulder = False
            self.time_incorrect += 1
            if self.time_incorrect > 100:
                self.time_incorrect = 100
            self.time_incorrect_shoulder += 1
            if self.time_incorrect_shoulder > 100:
                self.time_incorrect_shoulder = 100
        return
    
    def check_elbows(self): # check the angle of the elbows
        # Constants
        min_angle = 85
        max_angle = 115
        if min_angle <= self.l_e_angle <= max_angle and min_angle <= self.r_e_angle <= max_angle:
            self.correct_elbow = True
            self.time_incorrect_elbow -= 1
            if self.time_incorrect_elbow < 0:
                self.time_incorrect_elbow = 0
        elif self.l_e_angle < min_angle or self.l_e_angle > max_angle:
            self.correct_elbow = False
            self.time_incorrect += 1
            if self.time_incorrect > 100:
                self.time_incorrect = 100
            self.time_incorrect_elbow += 1
            if self.time_incorrect_elbow > 100:
                self.time_incorrect_elbow = 100
        elif self.r_e_angle < min_angle or self.r_e_angle > max_angle:
            self.correct_elbow = False
            self.time_incorrect += 1
            if self.time_incorrect > 100:
                self.time_incorrect = 100
            self.time_incorrect_elbow += 1
            if self.time_incorrect_elbow > 100:
                self.time_incorrect_elbow = 100

        return

if __name__ == "__main__":
    cam = camera()
    cam.display_camera()
