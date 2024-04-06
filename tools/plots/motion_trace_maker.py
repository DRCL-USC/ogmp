import sys
sys.path.append("./")
from tools.plots.motion_trace_utils import *
import argparse
import matplotlib.pyplot as plt

# load images



parser = argparse.ArgumentParser()
parser.add_argument("--path2video", default="results/bg_b_pr3_2_obs3/0/mode_space_test_5_5/logs/0/video.mp4", type=str)  
parser.add_argument("--path2frames", default=None, type=str)  

parser.add_argument("--fps", default=33, type=int)
parser.add_argument("--frame_skip", default=1, type=int)
parser.add_argument("--export_folder_prefix", default="motion_trace", type=str)
args = parser.parse_args()

args.export_folder_prefix = args.export_folder_prefix + '_'+str(args.frame_skip)


if args.path2frames is not None:
    path = args.path2frames
    op_path = args.path2frames + '../' + args.export_folder_prefix
else:
    # convert video to frames using ffmeg
    folder_path = os.path.dirname(args.path2video)
    frames_folder_path = os.path.join(folder_path,'frames_fs'+str(args.frame_skip))

    os.makedirs(frames_folder_path,exist_ok=True)

    video2frames_cmnd = "ffmpeg -i "+args.path2video+" -qscale:v 2 -r "+str(args.fps)+" "+frames_folder_path+"/%d.png"

    os.system(video2frames_cmnd)


    path = frames_folder_path
    op_path =folder_path + '/' + args.export_folder_prefix

os.makedirs(op_path,exist_ok=True)
imgs = []

valid_images = [".jpeg",".gif",".png",".bmp"]


fnames = sorted(os.listdir(path), key=lambda e: int(e.replace('frame_','').replace('.png','')))


for f in fnames:
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(cv.imread(os.path.join(path,f)))

curr_x = 0

def mouse_callback(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN:
        global curr_x
        curr_x = x

def glance_through_frames(frames):
    
    cv.namedWindow('Frame',cv.WINDOW_NORMAL) # Can be resized
    # cv.resizeWindow('image',frames[0].shape[1], frames[0].shape[0] ) #Reasonable size window
    cv.setMouseCallback('Frame',mouse_callback) #Mouse callback

    i = 0
    while True:
        
        frame = cv.putText(
                            np.copy(frames[i]), 
                            "Frame:"+str(i), 
                            org=(0,50),
                            fontFace=cv.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(0,0,255),
                            thickness = 2
                            )
        cv.imshow("Frame",frame)

        c = chr(cv.waitKey(25) % 256)
        if c == 'q':
            break
        elif c == 'd':
            i = i+1 if i+1<len(frames) else i
        elif c == 'a':
            i = i-1 if i-1>0 else i
    cv.destroyAllWindows()    

def crop_frames(frames,roi):
    
    for i,frame in enumerate(frames):
        frames[i] = frame[
                            roi[1]:roi[1]+roi[3],
                            roi[0]:roi[0]+roi[2],
                            :
                          ]
    return frames

glance_through_frames(imgs)

# get range of frames
frame_i_id = int(input("initial frame index:"))
frame_f_id = int(input("  final frame index:"))
frame_i_offset = int(input("initial frame offset:"))

# get merge initial and final frmae and get a rough 
# estimate of the roi
merged_frames_if = cv.addWeighted(
                            imgs[frame_i_id], 
                            0.5, 
                            imgs[frame_f_id], 
                            0.5, 
                            0.0
                            )
imgs = imgs[frame_i_id:frame_f_id]

if input("select roi manuly? press y") == 'y':
    cv.namedWindow("select roi") 
    roi = cv.selectROI("select roi", merged_frames_if)
    cv.moveWindow("select roi", 0, 0)
    cv.destroyWindow("select roi")
else:
    # roi = (340, 0, 1100, 913) # for mst
    # roi = (3, 387, 1437, 236) # for lons strip
    # roi = (74, 0, 1044, 690) # for oracle length 253:501
    roi = (119, 118, 985, 737) # for lmst_x0
    roi = (0, 215, 1437, 761) # for lmst_tr
    roi = (374, 0, 694, 976) # for front flip
    roi = (0, 380, 1437, 506) # for fgb longshot
    roi = (447, 0, 612, 976) # for abl vary orac
    roi = (1, 153, 1439, 823) # for abl vary horiz, rho

print("selected roi for crop:",roi)

# crop the imgs
imgs = crop_frames(imgs,roi)
glance_through_frames(imgs)

extract_ooi(
            frames=imgs,
            op_path=op_path, 
            frame_skip= args.frame_skip,
            frame_i_offset=frame_i_offset,
            )






