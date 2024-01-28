import cv2
import mediapipe as mp
import numpy as np
import math

class camera: # class for the camera, so that we can use it to display the camera and get all the joints from it
    def __init__(self):
        self.start = True
        self.counter = 0
        self.cap = cv2.VideoCapture(0)
        #self.detector = mp.pose.Pose()
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.results = None
        self.img = None
        self.imgRGB = None
        self.lmList = None
        self.down = True
        self.up = False
        self.pose = None
        self.squat_counter = 0
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(False, 1, True,
                                     False, True,
                                     0.7, 0.7)

    def display_camera(self): # displays the camera
        while self.cap.isOpened():
            ret, self.img = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            self.imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            # self.results = self.detector.process(self.img)
            self.results = self.pose.process(self.img)
            self.lms = self.get_LM_positions(self.img)
            if self.results.pose_landmarks:
                self.handle_squat()

            cv2.imshow("Image", self.img)
            self.display_squat_count(self.img)
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
                h, w, c = img.shape
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

    def handle_squat(self):
        # Constants
        low_knee_threshold =  90
        high_knee_threshold = 120
        # max_b_distance = 0.1

        # Angles
        self.l_hip_angle = self.calculate_2d_angle(11, 23, 25) # Hip angle (left)
        self.l_knee_angle = self.calculate_2d_angle(23, 25, 27) # Knee angle (left)
        self.r_hip_angle = self.calculate_2d_angle(12, 24, 26) # Hip angle (right)
        self.r_knee_angle = self.calculate_2d_angle(24, 26, 28) # Knee angle (right)
        
        if self.down and self.l_knee_angle < low_knee_threshold and self.r_knee_angle < low_knee_threshold:
            print("Suffienctly bent knees")
            if 70 < self.l_hip_angle < 90 and 70 < self.r_hip_angle < 90:
                print("Suffienctly bent hips and knees, go up")
                self.up = True
                self.down = False
            elif self.l_hip_angle < 70 or self.r_hip_angle < 70:
                print("Back is too forward enough")
            elif self.l_hip_angle > 90 or self.r_hip_angle > 90:
                print("Back is too backwards enough")

        elif self.up and self.l_knee_angle > high_knee_threshold and self.r_knee_angle > high_knee_threshold:
            print("Suffienctly straight knees")
            if self.l_hip_angle > 135 and self.r_hip_angle > 135:
                print("Suffienctly straight hips and knees, go down")
                self.up = False
                self.down = True
                self.squat_counter += 1
                print("Squat counter: ", self.squat_counter)
            else:
                print("Back is not straight enough")
        return
    
    def display_squat_count(self, img):
        # Choose a font
        font = cv2.FONT_HERSHEY_COMPLEX
        # Position (bottom right corner)
        bottomRightCornerOfText = (img.shape[1] - 1250, img.shape[0] - 50)
        fontScale = 2
        fontColor = (255, 255, 255)  # White color
        lineType = 4

        cv2.putText(img, f'Squat Count: {self.squat_counter}', 
                    bottomRightCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
    
    
if __name__ == "__main__":
    cam = camera()
    cam.display_camera()
