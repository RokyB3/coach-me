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
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(False, 1, True,
                                     False, True,
                                     0.7, 0.7)

        self.feedback = None

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
                self.handle_lunge()

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
        if self.start == False or self.done == True:
            if self.l_hip >= min_s_angle and self.r_hip >= min_s_angle:
                print("Start position is correct")
                self.start = True
                self.done = False
                if self.feedback != None:
                    pygame.mixer.music.load(self.feedback)
                    pygame.mixer.music.play()
                    self.feedback = None
                return

            # # Error if not straight
            # if self.l_hip < min_s_angle or self.r_hip < min_s_angle:
            #     print("Stand straight")
            #     pygame.mixer.music.load('../../../audio/output/lunge_stand_straight.mp3')
            #     pygame.mixer.music.play()
            #     self.start = False
            #     return
        if self.done == False:
            # Check left lunge
            if self.l_hip <= max_r_angle and self.l_hip >= min_r_angle and self.l_knee >= min_i_angle and self.l_knee <= max_i_angle:
                if self.r_hip >= min_s_angle and self.r_knee >= min_r_angle and self.r_knee <= max_r_angle:
                    print("Left lunge is correct")
                    self.feedback = '../../../audio/output/lunge_is_good.mp3'
                    self.done = True
                    self.counter += 1
                    return

            # Check right lunge
            if self.r_hip <= max_r_angle and self.r_hip >= min_r_angle and self.r_knee >= min_i_angle and self.r_knee <= max_i_angle:
                if self.l_hip >= min_s_angle and self.l_knee >= min_r_angle and self.l_knee <= max_r_angle:
                    print("Right lunge is correct")
                    self.feedback = '../../../audio/output/lunge_is_good.mp3'
                    self.done = True
                    self.counter += 1
                    return
            
            # Error if left foot too front
            if self.l_hip <= max_r_angle+30 and self.l_hip >= min_r_angle-30 and self.l_knee >= min_i_angle-30 and self.l_knee <= max_i_angle+30:
                if self.r_hip >= min_s_angle-30 and self.r_knee >= min_r_angle-30 and self.r_knee <= max_r_angle+30:
                    print("Left foot is too front")
                    if self.feedback != '../../../audio/output/lunge_is_good.mp3': self.feedback = '../../../audio/output/lunge_too_forward.mp3'
                    self.start = False
                    return
            
            # Error if right foot too front
            if self.r_hip <= max_r_angle+30 and self.r_hip >= min_r_angle-30 and self.r_knee >= min_i_angle-30 and self.r_knee <= max_i_angle+30:
                if self.l_hip >= min_s_angle-30 and self.l_knee >= min_r_angle-30 and self.l_knee <= max_r_angle+30:
                    print("Right foot is too front")
                    if self.feedback != '../../../audio/output/lunge_is_good.mp3': self.feedback = '../../../audio/output/lunge_too_forward.mp3'
                    self.start = False
                    return

        # Check too high (Right Foot Front)
        # if self.l_knee > max_r_angle and self.r_hip > max_r_angle:
        #     print("Too high")
        #     if self.feedback != '../../../audio/output/lunge_is_good.mp3': self.feedback = '../../../audio/output/lunge_too_high.mp3'
        #     self.start = False
        #     return

        # # Check too high (Left Foot Front)
        # if self.r_knee > max_r_angle and self.l_hip > max_r_angle:
        #     print("Too high")
        #     if self.feedback != '../../../audio/output/lunge_is_good.mp3': self.feedback = '../../../audio/output/lunge_too_high.mp3'
        #     self.start = False
        #     return

if __name__ == "__main__":
    cam = camera()
    cam.display_camera()
