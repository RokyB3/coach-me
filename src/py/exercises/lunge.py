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
        
        
        

if __name__ == "__main__":
    cam = camera()
    cam.display_camera()
