import cv2
import mediapipe as mp
import numpy as np
import math
import pygame

class camera: # class for the camera, so that we can use it to display the camera and get all the joints from it
    def __init__(self):
        self.start = True
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
                self.handle_pullup()

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
        if self.start == False or self.done == True:
            if self.l_e >= min_s_angle and self.l_s >= min_s_angle:
                if self.r_e >= min_s_angle and self.r_s >= min_s_angle:
                    print("Start position is correct")
                    self.start = True
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
                    self.feedback = '../../../audio/output/pullup_is_good.mp3'
                    self.done = True
                    self.counter += 1
                    return
            
            # Not high enough
            if self.l_e <= max_i_angle+40 or self.l_s <= max_i_angle+40 or self.r_e <= max_i_angle+40 or self.r_s <= max_i_angle+40:
                self.start = False
                print("Pullup is not high enough")
                if self.feedback != '../../../audio/output/pullup_is_good.mp3': # Don't overwrite good feedback
                    self.feedback = '../../../audio/output/pullup_not_high_enough.mp3'
                return

if __name__ == "__main__":
    cam = camera()
    cam.display_camera()
