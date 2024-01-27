import cv2
import mediapipe as mp
import numpy as np

# I Hate it here, I hate it here
import sys 
sys.path.append('../')
from opencv.camDisplay import camera


class squat: # class for the camera, so that we can use it to display the camera and get all the joints from it
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        #self.detector = mp.pose.Pose()
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.results = None
        self.cam = camera()

    def display_camera(self): # displays the camera
        self.cam.display_camera()
        return
    
    def find_Pose (self, img, draw=True):
        self.results = self.pose.process(self.imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img
    
    def get_xyz(self, landmark):
        return [landmark.x, landmark.y, landmark.z]
        

if __name__ == "__main__":
    squat = squat()
    squat.display_camera()
