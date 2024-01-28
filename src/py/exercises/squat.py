import cv2
import mediapipe as mp
import numpy as np
import pygame

# I Hate it here, I hate it here
import sys 
sys.path.append('../')
from opencv.camDisplay import camera

low_knee_threshold = 90
high_knee_threshold = 120

class squat: # class for the camera, so that we can use it to display the camera and get all the joints from it
    def __init__(self):
        self.cam = camera()
        self.down = True
        self.up = False
        self.squat_counter = 0
        self.knees_over = False
        self.back_forward = False

    def handle_squat(self): # displays the camera
        while self.cam.cap.isOpened():
            ret, self.cam.img = self.cam.cap.read()

            #initialize the mixer
            pygame.mixer.init()

            if not ret:
                print("Failed to grab frame")
                break

            self.cam.imgRGB = cv2.cvtColor(self.cam.img, cv2.COLOR_BGR2RGB) # convert the image to RGB
            self.cam.get_landmarks()
            self.get_angles() # get the angles of the joints (knees, hips, shoulders)
            if self.check_knees(): # check if the knees are bent
                    print("Squat")

            self.cam.img = self.cam.draw_Pose(self.cam.img) # draw the landmarks that we got from get_landmarks
            self.display_squat_count(self.cam.img) # display the squat count
            cv2.imshow("Image", self.cam.img) # display the image

            if cv2.waitKey(1) & 0xFF == ord('q'): # if the user presses q, then 
                break
                
        self.cam.cap.release()
        cv2.destroyAllWindows()
        return



    def get_angles(self): # get the angles of the joints (knees, hips, shoulders)
        # get the angles of the knees
        self.l_knee_angle = self.cam.calculate_3d_angle(self.cam.l_hi, self.cam.l_k, self.cam.l_a)
        self.r_knee_angle = self.cam.calculate_3d_angle(self.cam.r_hi, self.cam.r_k, self.cam.r_a) 

        #print("L_knee: ", self.l_knee_angle, " R_knee: ", self.r_knee_angle)

        # get the angles of the hips
        self.l_hip_angle = self.cam.calculate_3d_angle(self.cam.l_s, self.cam.l_hi, self.cam.l_k)
        self.r_hip_angle = self.cam.calculate_3d_angle(self.cam.r_s, self.cam.r_hi, self.cam.r_k)

       #print("L_hip: ", self.l_hip_angle, " R_hip: ", self.r_hip_angle)

        return        
    
    def check_knees(self): # check if the knees are bent
        if self.down and self.l_knee_angle < low_knee_threshold and self.r_knee_angle < low_knee_threshold:
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

        elif self.up and self.l_knee_angle > 130 and self.r_knee_angle > 130:
            if self.l_hip_angle > 135 and self.r_hip_angle > 135:
                print("Suffienctly straight hips and knees, go down")
                self.up = False
                self.down = True
                if not self.knees_over and not self.back_forward:
                    self.squat_counter += 1
                    print("Squat counter: ", self.squat_counter)

                    # play feedback
                    pygame.mixer.music.load('../../audio/output/squat_is_good.mp3')
                    pygame.mixer.music.play()
                if self.knees_over:
                    print("Your knees are over your toes")
                    
                    # play feedback
                    pygame.mixer.music.load('../../../audio/output/squat_toes_over.mp3')
                    pygame.mixer.music.play()
                if self.back_forward:
                    print("Back is too forward enough") 

                    # play feedback
                    pygame.mixer.music.load('../../audio/output/squat_back_forward.mp3')
                    pygame.mixer.music.play()
            
                self.knees_over = False
                self.back_forward = False
        return
    
    def check_toes(self): # check if the toes are over the knees
        # check if z coordinate of the knees are greater than the z coordinate of the toes plus some threshold
        if  self.cam.l_toe[2] - self.cam.l_k[2] > 0 or self.cam.r_toe[2] - self.cam.r_k[2] > 0:
            return True
        else:
            return False

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
    squat = squat()
    squat.handle_squat()
