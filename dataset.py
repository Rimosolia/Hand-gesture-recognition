import cv2
import imutils
#import numpy as np
#from sklearn.metrics import pairwise
#import time
bg = None
def run_avg(image, accumWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, accumWeight)
# Function - To segment the region of hand in the image
def segment(image, threshold=30):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)
# Main function
if __name__ == "__main__":
    accumWeight = 0.5
    camera = cv2.VideoCapture("http://192.168.1.35:4747/video")
    top, right, bottom, left = 44, 344, 300, 600
    num_frames = 0
    imageNumber =1
    calibrated = False
    while(True):
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
            	print (">>> Please wait!...")
            elif num_frames == 29:
                print (">>> Tiến Hành...")
        else:
            hand = segment(gray)
            if hand is not None:
                (thresholded, segmented) = hand
                # show the thresholded image
                cv2.imshow("Thesholded", thresholded)
                keypress = cv2.waitKey(1) & 0xFF
                if keypress == ord("a"):
                # Set the directory CORRECTLY
                    directory = "D:/Dataset/Hand_Fi/train/peace/"+str(imageNumber)+".jpg"
                    cv2.imwrite(directory,thresholded)  
                    imageNumber += 1
                    print(directory)
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        num_frames += 1        
        cv2.imshow("Video Feed", clone)
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break
camera.release()
cv2.destroyAllWindows()
