import numpy as np
import cv2
import pandas as pd
import scipy
import os
from tracker_utilities import worms_track

def grab_vids(folder):
    vids = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith('.avi'):
                vids.append(os.path.join(root, f))
    return vids

def chunks(n, min, thresh):
    while n % min < thresh:
        min += 1
    return min

def play_vid_t(M, h, w, index, scale, path, title="Video"):
    vid = cv2.VideoCapture(path)
    df_l = []
    paused = False
    for i in range(M.shape[2]):
        frame = M[:,:, i]
        f1 = cv2.resize(np.uint8(cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)), (int(w / scale), int(h / scale)))
        frame = cv2.adaptiveThreshold(f1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, (M.shape[0]//16 if (M.shape[0]//16) % 2 ==1 else M.shape[0]//16 + 1), (-M.shape[0]//32 if (M.shape[0]//32) % 2 ==1 else -M.shape[0]//32 + 1))
        df, worms = worms_track(frame, min=150, max=1000, time=i, ratio = 1)
        df_l.append(df)
        f = cv2.resize(worms, (1024, 768))
        vid.set(cv2.CAP_PROP_POS_FRAMES, i + index)
        _, b = vid.read()
        b = cv2.resize(b, (1024, 768))
        f = cv2.merge((f, np.zeros_like(f), f))
        cv2.imshow(title, cv2.addWeighted(b, 0.5, f, 0.9, 0))
        key = cv2.waitKey(300) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        while paused:
            key = cv2.waitKey(100) & 0xFF
            if key == ord(' '):
                paused = False
                break
    cv2.destroyAllWindows()
    return pd.concat(df_l)