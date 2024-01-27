import cv2
import mediapipe as mp
import numpy as np

class camera: # class for the camera, so that we can use it to display the camera and get all the joints from it
    def __init__(self):
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

    def display_camera(self): # displays the camera
        while self.cap.isOpened():
            ret, self.img = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            self.imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            #self.results = self.detector.process(self.imgRGB)
            self.img = self.find_Pose(self.img)
            cv2.imshow("Image", self.img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
        return
    
    def find_Pose (self, img, draw=True):
        self.results = self.pose.process(self.imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
                
        return img
        

if __name__ == "__main__":
    cam = camera()
    cam.display_camera()
