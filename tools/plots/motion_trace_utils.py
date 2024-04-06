from __future__ import print_function
import cv2 as cv
import os
import numpy as np
from tqdm import tqdm
scr_size = [1920,1090]

max_value = 255
max_value_H = 360//2
low_H = 70 #0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capt_name = 'Video Capture'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

def custom_add(img1, img2,mask1, mask2):
    
    new_image = np.zeros_like(img1)
    # cv.imshow('mask1',mask1)
    # cv.imshow('mask2',mask2)
    # cv.waitKey(0)
    # exit()
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            
            if mask2[i,j] != 0:

                new_image[i,j,:] = img2[i,j,:]
            
            elif mask1[i,j] != 0:
                new_image[i,j,:] = img1[i,j,:]

    return new_image

def final_add(img1, img2,mask):
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if mask[i,j] == 0:
                img1[i,j,:] = img2[i,j,:]
            else:
                img1[i,j,:] = 0.4*img1[i,j,:] + 0.6*img2[i,j,:]
    return img1

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, 'trackbars', low_H)

def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, 'trackbars', high_H)

def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, 'trackbars', low_S)

def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, 'trackbars', high_S)

def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos(low_V_name, 'trackbars', low_V)

def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos(high_V_name, 'trackbars', high_V)

def extract_ooi(frames,op_path,frame_skip=1,frame_i_offset=0):

    # make the ncessary windows
    for win_name in ['feed','trackbars','mask','filtered']:
        cv.namedWindow(win_name)
        # cv.resizeWindow(
        #             win_name, 
        #             int(0.5*scr_size[0]),
        #             int(0.5*scr_size[1])
        #             )

    ## create trackbars
    cv.createTrackbar(low_H_name, 'trackbars' , low_H, max_value_H, on_low_H_thresh_trackbar)
    cv.createTrackbar(high_H_name, 'trackbars' , high_H, max_value_H, on_high_H_thresh_trackbar)
    cv.createTrackbar(low_S_name, 'trackbars' , low_S, max_value, on_low_S_thresh_trackbar)
    cv.createTrackbar(high_S_name, 'trackbars' , high_S, max_value, on_high_S_thresh_trackbar)
    cv.createTrackbar(low_V_name, 'trackbars' , low_V, max_value, on_low_V_thresh_trackbar)
    cv.createTrackbar(high_V_name, 'trackbars' , high_V, max_value, on_high_V_thresh_trackbar)

    # arrange wndows
    cv.moveWindow('trackbars', 0, 0)
    cv.moveWindow('filtered', 0, 0)

    cv.moveWindow('feed', int(0.5*scr_size[0]),0)
    cv.moveWindow('mask', int(0.5*scr_size[1]),0)


    os.makedirs(op_path,exist_ok=True)
    
    i = 0
    while True:
        frame = frames[i] #cv.resize(frame, ( int(0.5*scr_size[0]),int(0.5*scr_size[1])) )
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame_thr = cv.bitwise_not(cv.inRange(frame_hsv, (low_H, low_S, low_V), (high_H, high_S, high_V)))
        frame_fil = cv.bitwise_and(frame,frame,mask= frame_thr)

        cv.imshow('mask', frame_thr)
        cv.imshow('feed', frame)
        cv.imshow('filtered', frame_fil)
    

        c = chr(cv.waitKey(200) % 256)
        if c == 'q':
            break
        
        elif c == 'f':
            import matplotlib.pyplot as plt

            cv.destroyAllWindows()
            
            i = 0            
            prev_frame = None
            prev_mask = None
            final_image = None
            # uniformly spaced along time
            for i in tqdm( range(0,len(frames),frame_skip) ):
                
                frame = frames[i] #cv.resize(frame, ( int(0.5*scr_size[0]),int(0.5*scr_size[1])) )
                frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                frame_thr = cv.bitwise_not(cv.inRange(frame_hsv, (low_H, low_S, low_V), (high_H, high_S, high_V)))
                frame_fil = cv.bitwise_and(frame,frame,mask= frame_thr)


                if prev_frame is None:
                    prev_frame = np.copy(frame)
                    prev_mask = np.copy(frame_thr)
                    final_image = np.copy(frame_fil)


                prev_mask = cv.bitwise_or(prev_mask, frame_thr)
                    

                final_image = custom_add(
                                            final_image,
                                            frame_fil, 
                                            prev_mask,
                                            frame_thr
                                        )
                # custom hack, for ov fig 
                if i == 10: # at frame i*frame_skip
                    # at frame 51
                    frame = frames[17]
                    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                    frame_thr = cv.bitwise_not(cv.inRange(frame_hsv, (low_H, low_S, low_V), (high_H, high_S, high_V)))
                    frame_fil = cv.bitwise_and(frame,frame,mask= frame_thr)
                    if prev_frame is None:
                        prev_frame = np.copy(frame)
                        prev_mask = np.copy(frame_thr)
                        final_image = np.copy(frame_fil)
                    prev_mask = cv.bitwise_or(prev_mask, frame_thr)
                        

                    final_image = custom_add(
                                                final_image,
                                                frame_fil, 
                                                prev_mask,
                                                frame_thr
                                            )
                if i == 40: # at frame i*frame_skip
                    # at frame 51
                    frame = frames[46]
                    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                    frame_thr = cv.bitwise_not(cv.inRange(frame_hsv, (low_H, low_S, low_V), (high_H, high_S, high_V)))
                    frame_fil = cv.bitwise_and(frame,frame,mask= frame_thr)
                    if prev_frame is None:
                        prev_frame = np.copy(frame)
                        prev_mask = np.copy(frame_thr)
                        final_image = np.copy(frame_fil)
                    prev_mask = cv.bitwise_or(prev_mask, frame_thr)
                        

                    final_image = custom_add(
                                                final_image,
                                                frame_fil, 
                                                prev_mask,
                                                frame_thr
                                            )
                if i == 70: # at frame i*frame_skip
                    # at frame 51
                    frame = frames[76]
                    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                    frame_thr = cv.bitwise_not(cv.inRange(frame_hsv, (low_H, low_S, low_V), (high_H, high_S, high_V)))
                    frame_fil = cv.bitwise_and(frame,frame,mask= frame_thr)
                    if prev_frame is None:
                        prev_frame = np.copy(frame)
                        prev_mask = np.copy(frame_thr)
                        final_image = np.copy(frame_fil)
                    prev_mask = cv.bitwise_or(prev_mask, frame_thr)
                        

                    final_image = custom_add(
                                                final_image,
                                                frame_fil, 
                                                prev_mask,
                                                frame_thr
                                            )                    

                # cv.imshow("current merged frame",final_image)
                cv.imwrite(op_path+'/'+str(i)+'.png',final_image)  

                # wait till frame is closed
                # while True:
                #     c = chr(cv.waitKey(25) % 256)
                #     if c == 'q':
                #         break



                i += frame_skip
                # print(i,frame_skip)               
            

            merged_frames_if = cv.addWeighted(
                            frames[0], 
                            0.5, 
                            frames[len(frames)-1], 
                            0.5, 
                            0.0
                            )
            result = final_add(
                                final_image,
                                # frames[len(frames)-1],
                                merged_frames_if,
                                mask=prev_mask
                                )


            print(op_path)
            cv.imwrite(op_path+'/final_merge.png',result)            
            
            cv.destroyAllWindows()
            exit()
            
            # manually select the next frme on the for
            # i = 0            
            # prev_frame = None
            # prev_mask = None
            # final_image = None

            # while True:
            #     frame = frames[i] #cv.resize(frame, ( int(0.5*scr_size[0]),int(0.5*scr_size[1])) )
            #     frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            #     frame_thr = cv.bitwise_not(cv.inRange(frame_hsv, (low_H, low_S, low_V), (high_H, high_S, high_V)))
            #     frame_fil = cv.bitwise_and(frame,frame,mask= frame_thr)


            #     if prev_frame is None:
            #         prev_frame = np.copy(frame)
            #         prev_mask = np.copy(frame_thr)
            #         final_image = np.copy(frame_fil)


            #     prev_mask = cv.bitwise_or(prev_mask, frame_thr)
                                
            #     cv.imshow("current merged frame",final_image)

            #     c = chr(cv.waitKey(25) % 256)
            #     if c == 'q':
            #         break
            #     elif c == 'd':
            #         i = i+1 if i+1<len(frames) else i
            #         print("inframe:",i)
            #     elif c == 'a':
            #         i = i-1 if i-1>0 else i
            #         print("inframe:",i)

            #     elif c == 's':
            #         final_image = custom_add(
            #                 final_image,
            #                 frame_fil, 
            #                 prev_mask,
            #                 frame_thr
            #             )
            # exit()
        i +=1
        if i >= len(frames):
            i = 0
  


