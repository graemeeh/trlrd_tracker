import torch
import numpy as np
import tensorly as tl
import cv2
import pandas as pd
from tracker_utilities import feats_detect, vid_setup, resizr, join_tracks
from background_utilities import chunks, play_vid_t, grab_vids

FOLDER = '/Users/graeme'
SCALE = 0.5
BATCH_SIZE = 30

tl.set_backend('pytorch')

# sth from https://www.nature.com/articles/s41598-025-18059-x. this is just the standard l1 shrinkage operator but tensorified
def shrink(X, tau):
    return torch.mul(torch.sign(X), torch.max(torch.abs(X) - tau, torch.zeros_like(X)))

# tensor ring decomposition: https://arxiv.org/abs/1606.05535
# I don't understand this too well but https://proceedings.mlr.press/v139/malik21b/malik21b.pdf has a better explanation than above arxiv preprint.
def trd(X, ranks):
    cores = tl.decomposition.tensor_ring(X, rank=ranks)
    return tl.tr_tensor.tr_to_tensor(cores)

# based on https://www.nature.com/articles/s41598-025-18059-x but without extra sparse tensor W (it was giving me grief!!)
def trlrd_(Z, rho=5, betamax=1e5, maxiter=10, tol=0.8e-2):
    Z = torch.from_numpy(Z)
    m, n, b = Z.shape
    ranks = (b//2, b//2, b//2, b//2)
    # ranks = (b, b, b, b)
    L = torch.zeros_like(Z)
    S = torch.zeros_like(Z)
    A = torch.zeros_like(Z)
    lamda = 1 / np.sqrt(max(m, n) * b)
    beta = 1e-4
    for i in range(maxiter):
        # update L
        L = trd(Z-S+ A/beta, ranks)
        # update S
        S = shrink(Z-L+ A/beta, lamda/beta)
        # update multiplier A
        A = A + beta * (Z-L-S)
        # update beta
        beta = min(beta * rho, betamax)
        err = torch.Tensor.norm(Z - L - S, 'fro') / torch.Tensor.norm(Z, 'fro')
        print("Iteration: ", i, "err:", err.item())
        if err < tol:
            break
    return torch.Tensor.numpy(L), torch.Tensor.numpy(torch.abs(S))

def load_vid(path, scale, batch_size, dtype=np.float32, **trlrd_kwargs):
    vid = cv2.VideoCapture(path)
    n_frames, h, w = vid_setup(vid, scale)
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
        Lb, Sb = trlrd_(X, **trlrd_kwargs)
        df_l.append(play_vid_t(np.abs(Sb), h, w, index = startframe, path=path, scale=scale))
        pf = pf + count
        startframe = endframe
    vid.release()
    return pd.concat(df_l)

def batch_track(folder):
    vids = grab_vids(folder)
    for i in vids:
        df = load_vid(i, scale=SCALE, batch_size=BATCH_SIZE)
        df.to_csv(i + '_worms.csv')
        VIDEO = cv2.VideoCapture(i)
        d, b = feats_detect(VIDEO)
        while True:
            cv2.imshow('Video', b)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break
        VIDEO.release()
        cv2.destroyAllWindows()
        join_tracks(df, 500)
        df.to_csv(i + '_worms.csv')
        d.to_csv(i + '_details.csv')
        print("all done!")

batch_track(FOLDER)
