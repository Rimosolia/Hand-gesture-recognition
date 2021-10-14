#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import cv2

newmod=load_model('C:/Users/FPTSHOP/hand_gestures.h5')  
background = None

accumulated_weight = 0.4

pre_top = 20
pre_bot = 300
pre_right = 300
pre_left = 600

def segment(frame, threshold=20):
    global background
    diff = cv2.absdiff(background.astype("uint8"), frame)
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)


    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        hand_segment = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment)

def calc_accum_avg(frame, accumulated_weight):

    global background
    if background is None:
        background = frame.copy().astype("float")
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)
 
def predict_display(img):
    width=64
    height=64
    dim=(width,height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    test_img=image.img_to_array(resized)
    test_img=np.expand_dims(test_img,axis=0)
    result= newmod.predict(test_img)
    val=[index for index,value in enumerate(result[0]) if value ==1]
    return val
    
cam = cv2.VideoCapture(0)

num_frames = 0

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_copy = frame

    roi = frame[pre_top:pre_bot, pre_right:pre_left]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    if num_frames < 60:
        calc_accum_avg(gray, accumulated_weight)
        if num_frames <= 59:
            cv2.putText(frame_copy, "DANG XU LI BACKGROUND.", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
    else:
        cv2.putText(frame_copy, "Place your hand in side the box", (330, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        cv2.putText(frame_copy, "Buffalo = 0", (330, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        cv2.putText(frame_copy, "Fist = 1", (330, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        cv2.putText(frame_copy, "Five = 2", (330, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        cv2.putText(frame_copy, "Okay = 3", (330, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        cv2.putText(frame_copy, "Hi = 4", (330, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)


        hand = segment(gray)

        if hand is not None:
            thresholded, hand_segment = hand

            cv2.drawContours(frame_copy, [hand_segment + (pre_right, pre_top)], -1, (255, 0, 0),1)

            cv2.imshow("Thresholded Image", thresholded)
            res=predict_display(thresholded)
            
            if len(res)==0:
                cv2.putText(frame_copy, str('None'), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                x='index'+str(res[0])
                cv2.putText(frame_copy, str(x), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            

    cv2.rectangle(frame_copy, (pre_left, pre_top), (pre_right, pre_bot), (0,0,255), 2)

    num_frames += 1

    cv2.imshow("Hand Gestures", frame_copy)

    # Close windows with Esc
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
