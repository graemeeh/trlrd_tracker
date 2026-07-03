

FOLDER = 'E:/GE/GE Ethanol Quadrants Resampled'
SCALE = 0.5
BATCH_SIZE = 30

from tracker_utilities import feats_detect, vid_setup, resizr, join_tracks
from background_utilities import chunks, play_vid_t, grab_vids, shrink

import cv2
import pandas as pd
import torch
import numpy as np
import tensorly

tensorly.set_backend('pytorch')

# tensor ring decomposition: https://arxiv.org/abs/1606.05535
# I don't understand this too well but https://proceedings.mlr.press/v139/malik21b/malik21b.pdf has a better explanation than above arxiv preprint.
def trd(X, ranks):
    """
    :param X: input tensor (video)
    :param ranks: dimensions for tensor cores
    :return: input tensor reconstructed from cores
    """
    cores = tensorly.decomposition.tensor_ring(X, rank=ranks)
    return tensorly.tr_tensor.tr_to_tensor(cores)

# based on https://www.nature.com/articles/s41598-025-18059-x but without extra sparse tensor W (it was giving me grief!!)
def trlrd(Z, rho=5, betamax=1e5, maxiter=10, tol=0.8e-2):
    """
    :param Z: input video
    :param rho: factor by which penalty "beta" is increased by each iteration
    :param betamax: maximum value for beta. it is pretty redundant in this
    :param maxiter: maximum number of iterations before the program quits. use tolerance rather than maxiter to limit iterations spent on one batch
    :param tol: tolerance for error norm
    :return: sparse foreground video S
    """
    Z = torch.from_numpy(Z) if isinstance(Z, np.ndarray) else Z
    m, n, b = Z.shape
    ranks = (b//2, b//2, b//2, b//2)
    # ranks = (b, b, b, b)
    S = torch.zeros_like(Z)
    A = torch.zeros_like(Z)
    lamda = 1 / np.sqrt(max(m, n) * b)
    beta = 1e-4
    for i in range(maxiter):
        L = trd(Z-S+ A/beta, ranks)
        S = shrink(Z-L+ A/beta, lamda/beta)
        A = A + beta * (Z-L-S)
        beta = min(beta * rho, betamax)
        err = torch.Tensor.norm(Z - L - S, 'fro') / torch.Tensor.norm(Z, 'fro')
        print("Iteration: ", i, "err:", err.item())
        if err.item() < tol:
            break
    return torch.Tensor.numpy(torch.abs(S))

def load_vid(path, scale, batch_size, dtype=np.float32, **trlrd_kwargs):
    """
    :param path: input path .avi file that you want to track worms on
    :param scale: Scaling factor to apply to video frames. I find that 0.5 is pretty good
    :param batch_size: size of "batches" of frames to process at once. Dependent on how much your worms move and what your framerate is.
    :param dtype: float32 for speed.
    :param trlrd_kwargs: in case you want to change defaults
    :return: data frame with all the worm track information on it
    """
    vid = cv2.VideoCapture(path)
    n_frames, h, w , g = vid_setup(vid, scale)
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename=path + 'output.avi', fourcc=fourcc, fps=7.5, frameSize=(int(w*scale), int(h*scale)), isColor=False)
    pf = 0
    startframe = 0
    df_l = []
    while pf < n_frames:
        chunk_l = []
        count = 0
        for _ in range(chunks(n_frames, batch_size, 3*batch_size/4)):
            ret, frame = vid.read()
            if not ret:
                break
            f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            chunk_l.append(resizr(f, scale).astype(dtype))
            count += 1
        if not chunk_l:
            break
        X = np.transpose(np.array(chunk_l, dtype=dtype), (1, 2, 0))
        endframe = startframe + count
        print(f"frames {startframe}-{endframe - 1}")
        Sb = trlrd(X, **trlrd_kwargs)
        df_l.append(play_vid_t(Sb, h, w, index = startframe, path=path, scale=scale, out = out))
        pf = pf + count
        startframe = endframe
    vid.release()
    out.release()
    return pd.concat(df_l)

def batch_track(folder):
    """
    :param folder: recursively track worms on each .avi file in this folder.
    :return: nothing, saves data frames in each folder that it finds a .avi file in.
    """
    vids = grab_vids(folder)
    for i in vids:
        df = load_vid(i, scale=SCALE, batch_size=BATCH_SIZE)
        df.to_csv(i + '_worms.csv')
        VIDEO = cv2.VideoCapture(i)
        d, b = feats_detect(VIDEO)
        while True:
            cv2.imshow('Video', b)
            if cv2.waitKey(1000):
                break
        VIDEO.release()
        cv2.destroyAllWindows()
        join_tracks(df, 500)
        df.to_csv(i + '_worms.csv')
        d.to_csv(i + '_details.csv')
        print("all done!")

batch_track(FOLDER)
