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
            if self.results.pose_landmarks:
                self.handle_lunge()
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
    
    def get_xyz(self, landmark):
        return [landmark.x, landmark.y, landmark.z]

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

    def get_xy_distance(self, a, b):
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
        max_b_distance = 0.1
        # Landmarks
        self.l_s = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_SHOULDER])
        self.l_hi = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_HIP])
        self.l_k = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_KNEE])
        self.l_a = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_ANKLE])
        self.r_s = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_SHOULDER])
        self.r_hi = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_HIP])
        self.r_k = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_KNEE])
        self.r_a = self.get_xyz(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_ANKLE])

        # Back translation
        self.l_back = self.get_xy_distance(self.l_s, self.l_hi)
        self.r_back = self.get_xy_distance(self.r_s, self.r_hi)

        # Angles
        self.la0 = self.calculate_3d_angle(self.l_s, self.l_hi, self.l_k) # Hip angle (left)
        self.la1 = self.calculate_3d_angle(self.l_hi, self.l_k, self.l_a) # Knee angle (left)
        self.ra0 = self.calculate_3d_angle(self.r_s, self.r_hi, self.r_k) # Hip angle (right)
        self.ra1 = self.calculate_3d_angle(self.r_hi, self.r_k, self.r_a) # Knee angle (right)
        
        # print(f"L_Back: {self.l_back}   |   R_Back: {self.r_back}")
        # print(f"L Hip angle: {self.la0}   |   R Hip angle: {self.ra0}")
        # print(f"L Knee angle: {self.la1}   |   R Knee angle: {self.ra1}")

        # Check if back is straight
        if self.l_back < max_b_distance and self.r_back < max_b_distance:
            print("Back is straight")
        else:
            print("Back is not straight")

        # Check start position
        if self.la0 >= min_s_angle and self.ra0 >= min_s_angle:
            print("Start position is correct")
        else:
            print("Start position is incorrect")
        # Check left lunge
        if self.la0 <= max_r_angle and self.la0 >= min_r_angle and self.la1 >= min_i_angle and self.la1 <= max_i_angle:
            if self.la0 >= min_s_angle and self.la1 >= min_r_angle and self.la1 <= max_r_angle:
                print("Left lunge is correct")
        else:
            print("Left lunge is incorrect")

        # Check right lunge
        if self.ra0 <= max_r_angle and self.ra0 >= min_r_angle and self.ra1 >= min_i_angle and self.ra1 <= max_i_angle:
            if self.ra0 >= min_s_angle and self.ra1 >= min_r_angle and self.ra1 <= max_r_angle:
                print("Right lunge is correct")
        else:
            print("Right lunge is incorrect")

        


        

if __name__ == "__main__":
    cam = camera()
    cam.display_camera()
