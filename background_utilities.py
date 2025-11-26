import numpy as np
import cv2
import pandas as pd
import os
import torch
from tracker_utilities import worms_track

# sth from https://www.nature.com/articles/s41598-025-18059-x
def shrink(X, tau):
    """
    :param X: input tensor
    :param tau: threshold
    :return: thresholded X, with zeros everywhere that X doesn't clear threshold tau
    """
    return torch.mul(torch.sign(X), torch.max(torch.abs(X) - tau, torch.zeros_like(X)))

def grab_vids(folder):
    """
    :param folder: input folder that you want to recursively track all .avi files in
    :return: all videos (with .avi extension) in a given folder
    """
    vids = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith('.avi'):
                vids.append(os.path.join(root, f))
    return vids

def chunks(n, min, thresh):
    """
    :param n: # of frames in vid
    :param min: minimum chunk size in frames
    :param thresh: how small can the final chunk be
    :return: min, updated as per condition below
    """
    while n % min < thresh:
        min += 1
    return min

def play_vid_t(M, h, w, index, scale, path, out, title="Video"):
    """
    :param M: input video
    :param h: unscaled height of video frame
    :param w: unscaled width
    :param index: index of first frame in video
    :param scale: scaling factor
    :param path: path to raw .avi video
    :param out: output .avi video (does not work yet)
    :param title: title of cv2 window
    :return: data frame with all worm tracking info
    """
    vid = cv2.VideoCapture(path)
    df_l = []
    paused = False
    for i in range(M.shape[2]):
        frame = M[:,:, i]
        f1 = cv2.resize(np.uint8(cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)), (int(w / scale), int(h / scale)))
        frame = cv2.adaptiveThreshold(f1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, (M.shape[0]//32 if (M.shape[0]//32) % 2 ==1 else M.shape[0]//32 + 1), (-M.shape[0]//64 if (M.shape[0]//64) % 2 ==1 else -M.shape[0]//64 + 1))
        out.write(frame)
        df, worms = worms_track(frame, min=150, max=1000, time=i, ratio = 1)
        df_l.append(df)
        f = cv2.resize(worms, (1024, 768))
        vid.set(cv2.CAP_PROP_POS_FRAMES, i + index)
        _, b = vid.read()
        b = cv2.resize(b, (1024, 768))
        f = cv2.merge((f, np.zeros_like(f), f))
        cv2.imshow(title, cv2.addWeighted(b, 0.5, f, 0, 0))
        key = cv2.waitKey(1000) & 0xFF
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
