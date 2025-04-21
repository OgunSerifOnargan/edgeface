from time import time
from cv2 import resize, absdiff, cvtColor, COLOR_BGR2GRAY, GaussianBlur, threshold, dilate, findContours, RETR_TREE, CHAIN_APPROX_SIMPLE, contourArea

class motion_detection():
    def __init__(self, min_area):
        self.min_area = min_area
        self.motion_detected = True
        self.previousFrame = None
        self.currentFrame = None
        self.last_motion_detection_time = time()
        self.is_first_frame = True
        
    def set_motion_detected(self, state):
        self.motion_detected = state
		
    def motion_checker(self):
        resize(self.previousFrame, (100,100))
        resize(self.currentFrame, (100,100))
        diff = absdiff(self.previousFrame, self.currentFrame)
        diff_gray = cvtColor(diff, COLOR_BGR2GRAY)
        blur = GaussianBlur(diff_gray, (21, 21), 0)
        _, thresh = threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = dilate(thresh, None, iterations=3)
        contours, _ = findContours( dilated, RETR_TREE, CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if contourArea(contour) < self.min_area:
                pass
            else:
                print(contourArea(contour))
                self.motion_detected = True
                self.last_motion_detection_time = time.time()
                break