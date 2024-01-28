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
        
        # Landmarks
        self.l_s = (0,0,0)
        self.l_hi = (0,0,0)
        self.l_k = (0,0,0)
        self.l_a = (0,0,0)
        self.l_e = (0,0,0)
        self.l_w = (0,0,0)
        self.l_he = (0,0,0)
        self.r_s = (0,0,0)
        self.r_hi = (0,0,0)
        self.r_k = (0,0,0)
        self.r_a = (0,0,0)
        self.r_e = (0,0,0)
        self.r_w = (0,0,0)
        self.r_he = (0,0,0)

    def display_camera(self): # displays the camera
        while self.cap.isOpened():
            ret, self.img = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            self.imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB) # convert the image to RGB
            self.get_landmarks() # get the landmarks of the person in the camera

            self.img = self.draw_Pose(self.img) # draw the landmarks that we got from get_landmarks
            cv2.imshow("Image", self.img) # display the image

            if cv2.waitKey(1) & 0xFF == ord('q'): # if the user presses q, then 
                break

        self.cap.release()
        cv2.destroyAllWindows()
        return
        
    def draw_Pose (self, img, draw=True):
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

if __name__ == "__main__":
    cam = camera()
    cam.display_camera()
