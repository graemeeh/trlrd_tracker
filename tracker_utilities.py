import numpy as np
import cv2
import pandas as pd
import scipy
import os

from cv2 import COLOR_GRAY2BGR


def log_abs(img, sigma: float = 0.3):
    """
    see https://www.youtube.com/watch?v=uNP6ZwQ3r6A for how this works
    :param img:
    :param sigma:
    :param thresh:
    :return:
    """
    img = cv2.Laplacian(cv2.GaussianBlur(img, (0, 0), sigma), cv2.CV_64F) # note data type cv_64f
    return cv2.normalize(np.abs(-img), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # gets absolute value & scales the pixel values of array so that theyare between 0-255

def worms_track(edges, min: int, max: int, ratio: float, time: int):
    """

    :param edges: input image
    :param min: minimum area
    :param max: max area
    :param ratio: ratio between circumference and area
    :param time: frame index
    :return: pandas dataframe, highlighted image
    """
    labeled_image, num_labels = scipy.ndimage.label(edges)
    slices = scipy.ndimage.find_objects(labeled_image)  # chops it up into slices based on labels
    contour_data = []
    highlighted = np.zeros_like(edges, dtype=np.uint8)
    for i, slice in enumerate(slices, 1):  # start from 1 because 0 is background
        contour_mask = (labeled_image[slice] == i)
        # peri = np.sum(log_abs(contour_mask))
        area = np.sum(contour_mask)
        if min < area < max:
            cx, cy = np.mean(np.column_stack(np.where(contour_mask)), axis=0)
            highlighted[slice][contour_mask] = 255
            contour_data.append({
                'index': i,
                'frame': time,
                'centroid_x': int(cx + slice[1].start),
                'centroid_y': int(cy + slice[0].start),
                'area': area,
                'convex_hull_area': int(scipy.spatial.ConvexHull(np.column_stack(np.where(contour_mask))).volume),
                # 'circumference': peri,
                'height': slice[0].stop - slice[0].start,
                'width': slice[1].stop - slice[1].start})
    return pd.DataFrame(contour_data), highlighted

def median_frame(vid):
    # This is from https://github.com/samwestby/OpenCV-Python-Tutorial/blob/main/8_background_est.py
    frame_ids = [i for i in range(1, 41)]
    frames = []
    for f in frame_ids:
        vid.set(cv2.CAP_PROP_POS_FRAMES, f)
        r, f = vid.read()
        if not r:
            print("SOMETHING WENT WRONG!!!!")
            exit()
        frames.append(f)
    return cv2.cvtColor(np.median(frames, axis=0).astype(np.uint8), cv2.COLOR_BGR2GRAY)

def resizr(image, scale):
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

def vid_setup(c, s):
    ret, frame = c.read()
    n_frames = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    gray = resizr(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), s)
    h, w = gray.shape[:2]
    return n_frames, h, w

def feats_detect(vid, min: int = 10000, max: int = 10000000, ksize: int = 155):
    """

    :param min:
    :param max:
    :param vid:
    :param ksize:
    :return:
    """
    hist = cv2.equalizeHist(median_frame(vid))
    line_image = cv2.cvtColor(np.zeros_like(hist), COLOR_GRAY2BGR)
    height, width = hist.shape[:2]
    minR = round(width / 3)
    maxR = round(width / 1.5)
    minDis = round(width / 1)
    blur = cv2.GaussianBlur(hist, (31, 31), cv2.BORDER_DEFAULT)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, minDis, param1=14, param2=100, minRadius=minR, maxRadius=maxR)
    details = []
    '''
    lines = cv2.HoughLinesP(blur, rho = 1, theta = np.pi / 180, threshold = 500, lines = np.array([]), minLineLength = 500, maxLineGap = 20 )
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
            details.append({'type': 'line','x1': x1,'y1': y1,'x2/radius': x2,'y2': y2})
    '''
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(line_image, (i[0], i[1]), 3, color = (255, 255, 0), thickness=5)
            cv2.circle(line_image, (i[0], i[1]), i[2], color = (255, 0, 255), thickness=5)
            details.append({'type': 'circle', 'x1': i[0], 'y1': i[1], 'x2/radius': i[2], 'y2': 0})
    return pd.DataFrame(details), cv2.addWeighted(cv2.cvtColor(median_frame(vid), cv2.COLOR_GRAY2BGR), 0.5, line_image, beta =0.9, gamma = 0)

def join_tracks(f: pd.DataFrame, maxdist: int):
    """
    :param f: data frame with columns for index, frame (aka time point), centroid_x, centroid_y, area, convex_hull_area (all of which define a track)
    :param maxdist: maximum distance over which tracks can be joined. tracks further away cannot be joined.
    :return: new pandas dataframe. has tracks joined (joined tracks have same index, different frames)
    """
    f['index'] = [i for i in range(0, len(f))]
    for i in pd.unique(f['frame']):
        sub = f[f["frame"] == i]
        x = sub['centroid_x'].values
        y = sub['centroid_y'].values
        # optimize distance between tracks in this frame and tracks in the following frame. Join tracks by updating indices in the following frame to match the joined tracks.
        # maybe also optimize area difference between subsequent frames? cost matrix from https://github.com/Tierpsy/tierpsy-tracker/blob/development/tierpsy/analysis/traj_join/joinBlobsTrajectories.py
    return f

VIDEO = cv2.VideoCapture("N2.avi")
d, b = feats_detect(VIDEO)
while True:
    b = cv2.resize(b, (1024, 768))
    cv2.imshow('Video', b)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break
