import sys

import nibabel as nib
import numpy as np
import torch
import cv2
import pandas as pd 

from ptv import register_pair_min
from skopt import gp_minimize
from skopt.space import Real, Integer
import json
from pathlib import Path
from tqdm.auto import tqdm
import time
from datetime import timedelta

def load_data(path_to_img, path_to_landmark):
    data_folder = f"/mnt/asgard0/data/y2026-p13-Data"
    img = cv2.imread(data_folder + '/ANHIR/' + path_to_img, cv2.IMREAD_GRAYSCALE)
    landmarks = pd.read_csv(data_folder +'/Landmarks/' +  path_to_landmark)
    if img is None:
        raise FileNotFoundError(path_to_img)
    if landmarks is None:
        raise FileNotFoundError(path_to_landmarks)
    return img, landmarks

def truncate_landmarks(mov_landmarks, fix_landmarks):
    # Truncate both to the shorter length
    n = min(len(mov_landmarks), len(fix_landmarks))
    truncated = False
    if(len(mov_landmarks) != len(fix_landmarks)):
        mov_landmarks = mov_landmarks.iloc[:n].reset_index(drop=True)
        fix_landmarks = fix_landmarks.iloc[:n].reset_index(drop=True)
        truncated = True
    return truncated, mov_landmarks, fix_landmarks

def img_to_tensor(img, device, dtype):
    img = img.astype(np.float32) / 255.0
    return torch.as_tensor(img, device=device, dtype=dtype)[None, None]
    
def tensor_to_img(tensor):
    img = tensor.detach().cpu()[0, 0].numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img

def sobel(img, ksize):
    ddepth = cv2.CV_16S
    scale = 1
    delta = 0

    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=ksize, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=ksize, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

def downsample(img, landmarks, fx, fy):

    landmarks["X"] *= fx
    landmarks["Y"] *= fy

    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA), landmarks


def preprocess_img(img, landmarks, fx, fy):
    img = sobel(img, ksize=7)
    img = cv2.medianBlur(img, 5)
    img = downsample(img, landmarks, fx, fy)
    return img

def preprocess_img_V2(img, landmarks, fx, fy, ksize=3):
    f = img.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = np.sqrt(gx**2 + gy**2)

    mag = cv2.medianBlur(mag, 5)                      # float32 works for ksize 3 or 5
    mag = cv2.resize(mag, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    landmarks["X"] *= fx
    landmarks["Y"] *= fy

    mag = np.clip(mag / (np.percentile(mag, 99) + 1e-8), 0, 1)
    return (mag * 255).astype(np.uint8), landmarks

def add_padding(mov, fix, mov_landmarks, fix_landmarks):

    x_diff, y_diff = mov.shape[1] - fix.shape[1], mov.shape[0] - fix.shape[0]

    mov_padding = [np.abs(min(x_diff, 0)), np.abs(min(y_diff, 0))]
    fix_padding = [max(x_diff, 0), max(y_diff, 0)]

    stop, sbot = mov_padding[1]//2, mov_padding[1]//2 + mov_padding[1]%2
    ttop, tbot = fix_padding[1]//2, fix_padding[1]//2 + fix_padding[1]%2

    sleft, sright = mov_padding[0]//2, mov_padding[0]//2 + mov_padding[0]%2
    tleft, tright = fix_padding[0]//2, fix_padding[0]//2 + fix_padding[0]%2

    mov = cv2.copyMakeBorder(mov, stop, sbot, sleft, sright, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    fix = cv2.copyMakeBorder(fix, ttop, tbot, tleft, tright, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    mov_landmarks["X"] += sleft
    fix_landmarks["X"] += tleft 

    mov_landmarks["Y"] += stop
    fix_landmarks["Y"] += ttop 


    return mov, fix

def load_training_csv(csv_file):
    data_folder = f"/mnt/asgard0/data/y2026-p13-Data/"
    df = pd.read_csv(data_folder + csv_file)
    training_df = df[df["status"] == "training"]
    return training_df

def load_evaluation_csv(csv_file):
    data_folder = f"/mnt/asgard0/data/y2026-p13-Data/"
    df = pd.read_csv(data_folder + csv_file)
    evaluation_df = df[df["status"] == "evaluation"]
    return evaluation_df


def compute_tre(diagonal, pts_fix, pts_mov, Tmm, pix_resolution):
    """Median per-landmark error of the stored map, as a fraction of the diagonal.

    pts_fix, pts_mov: (N, nd) paired landmarks, row then column, in pixels.
    Tmm: (1, nd, ...) displacement on the target grid, in pix_resolution units.
    diagonal: target image diagonal in pixels, sqrt(height**2 + width**2).
    """
    pix = torch.as_tensor(pix_resolution, device=Tmm.device, dtype=Tmm.dtype)
    pts_fix = pts_fix.to(device=Tmm.device, dtype=Tmm.dtype)
    pts_mov = pts_mov.to(device=Tmm.device, dtype=Tmm.dtype)
    diagonal = torch.as_tensor(diagonal, device=Tmm.device, dtype=Tmm.dtype)
    nd = pts_fix.shape[1]
    size = torch.tensor(Tmm.shape[-nd:], device=Tmm.device, dtype=Tmm.dtype)

    norm = pts_fix / (size - 1) * 2 - 1
    grid = norm.flip(-1).view(1, -1, *([1] * (nd - 1)), nd)
    disp = torch.nn.functional.grid_sample(
        Tmm, grid, mode="bilinear", padding_mode="border", align_corners=True,
    ).reshape(nd, -1).T

    err = pts_fix + disp / pix - pts_mov
    return float(err.norm(dim=-1).median() / diagonal)


def data_to_tensors(device, fixed_path, moving_path, fixed_landmarks_path, moving_landmarks_path,
                    fx, fy):
    dtype = torch.float32
    ## 3d - ignore 
    # fix_t, fix_nii = load(sys.argv[1], device, dtype)
    # mov_t, _ = load(sys.argv[2], device, dtype)
    # spacing = tuple(float(z) for z in fix_nii.header.get_zooms()[:3])

    # 2d overwrite 
    fix_img, fix_landmarks = load_data(fixed_path, fixed_landmarks_path)
    mov_img, mov_landmarks = load_data(moving_path, moving_landmarks_path)

    truncated, mov_landmarks, fix_landmarks = truncate_landmarks(mov_landmarks, fix_landmarks)

    # ------- Preprocessing ---------
    fix_img, fix_landmarks = preprocess_img_V2(fix_img, fix_landmarks, fx, fy)
    cv2.imwrite("fix_img.jpg", fix_img)

    mov_img, mov_landmarks = preprocess_img_V2(mov_img, mov_landmarks, fx, fy)
    cv2.imwrite("mov_img.jpg", mov_img)

    fix_img, mov_img = add_padding(fix_img, mov_img, fix_landmarks, mov_landmarks)

    # ------ Create tensor ---------
    fix_t_tens = img_to_tensor(fix_img, device, dtype)
    mov_t_tens = img_to_tensor(mov_img, device, dtype)
    return truncated, mov_t_tens, fix_t_tens, mov_landmarks, fix_landmarks


def compute_metric(diagonal, opts, mov_t_tens, fix_t_tens, mov_landmarks, fix_landmarks):

    _voldef, disp_mm, _knots, _infos = register_pair_min(mov_t_tens, fix_t_tens, opts)
    
    pts_mov = torch.tensor(mov_landmarks[["Y", "X"]].to_numpy(), dtype=torch.float32)  # (N, 2)
    pts_fix = torch.tensor(fix_landmarks[["Y", "X"]].to_numpy(), dtype=torch.float32)  # (N, 2)

    spacing = opts["pix_resolution"]
    tre = compute_tre(diagonal, pts_fix, pts_mov, disp_mm, pix_resolution=spacing)

    voldef_img = tensor_to_img(_voldef)
    cv2.imwrite("voldef.jpg", voldef_img)

    disp = disp_mm[0].detach().cpu().numpy().astype(np.float32)
    for ax in range(2):
        disp[ax] /= spacing[ax]

    return tre, _infos[-1]


def objective_function(params, fx, fy, training_df, device, run_iter):

    out_rows = []    
    for _, row in tqdm(training_df.iterrows(), total=len(training_df), desc="Processing"):
        start_time = time.perf_counter()

        fixed = row.loc["Target image"]
        fix_landmarks = row.loc["Target landmarks"]
        moving = row.loc["Source image"]
        mov_landmarks = row.loc["Source landmarks"]
        diagonal = row.loc["Image diagonal [pixels]"]

        diagonal_scaled = diagonal * fx

        k, w, lamb = int(params[0]), float(params[1]), float(params[2])
        opts = {
        "pix_resolution": (1.0, 1.0),
        "grid_spacing": [k,k],#[8, 8], #k
        "cp_refinements": 1,
        "k_down": 0.7,
        "metric": "lcc", # try lcc or mi and choose the best one :)
        "metric_param": (w, w, w),#(2.1, 2.1, 2.1), #w
        "scale_metric_param": True,
        "isoTV": lamb,#0.11, #regularization
        "csqrt": 5e-3,
        "border_mask": 5,
        "opt_method": "admm",
        "jac_reg": 0.0,
        "jac_reg_dense": 2.0,
        "device": device,
        "verbose": False,
        "track_progress": False}

        truncated, mov_t_tens, fix_t_tens, mov_landmarks, fix_landmarks = data_to_tensors(device, fixed, moving, fix_landmarks, mov_landmarks, fx, fy)
#(#opts, mov_t_tens, fix_t_tens, mov_landmarks, fix_landmarks):
        rTRE, infos = compute_metric(diagonal_scaled, opts, mov_t_tens, fix_t_tens, mov_landmarks, fix_landmarks) # params? 

        elapsed = time.perf_counter() - start_time

        out_row = {
        "source": moving,
        "target": fixed,
        "truncated": truncated,
        "rTRE": rTRE,
        "total_loss_norm": infos.total_loss_norm,
        "elapsed_seconds": round(elapsed, 2),
        "elapsed_readable": str(timedelta(seconds=elapsed))}
        #print(out_row)
        out_rows.append(out_row)

    out_df = pd.DataFrame(out_rows, columns=["source", "target", "truncated", "rTRE", "total_loss_norm", "elapsed_seconds", "elapsed_readable"])
    results_dir = f"results/run{run_iter}"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    out_df.to_csv(f"{results_dir}/results.csv", index=False)
    
    param_dict = {"k": k, "w": w, "lamb": lamb}
    with open(f"{results_dir}/params.json", "w") as f:
        json.dump(param_dict, f)
    
    mean_rTRE = float(out_df["rTRE"].mean())
    elapsed_total = float(out_df["elapsed_seconds"].sum())
    row = pd.DataFrame([{"mean_rTRE": mean_rTRE } | param_dict | {"elapsed_readable": str(timedelta(seconds=elapsed_total))}])
    bayesian_opt_results_path = f"results/bayesian_opt_results.csv"
    row.to_csv(bayesian_opt_results_path, mode="a", header=not Path(bayesian_opt_results_path).exists(), index=False)

    return mean_rTRE

def filter_training_data(df, n_pairs):

    df["category"] = df["Source image"].str.split("/").str[0].str.split("_").str[0]

    n_total = n_pairs
    n_per_cat = n_total // df["category"].nunique()  

    samples = (
        df.groupby("category", group_keys=False)
        .sample(n=n_per_cat, random_state=42)     
    )

    return samples

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    training_df = load_training_csv("dataset_medium.csv")
    training_df = filter_training_data(training_df, 8)
    if training_df["status"].unique() == "training":
        print(f"Successfully loaded training dataset. ({training_df.shape[0]} rows)")

    evaluation_df = load_evaluation_csv("dataset_medium.csv")
    if evaluation_df["status"].unique() == "evaluation":
        print(f"Successfully loaded evaluation dataset. ({evaluation_df.shape[0]} rows)")

    out_df = pd.DataFrame(columns=['Source landmarks', 'Target landmarks', 'rTRE']) 

    space = [
        Integer(4, 32, name="k"),
        Real(1, 10, name="w"),
        Real(1e-4, 10.0, prior="log-uniform", name="lamb")
    ]
    
    run_iter = 0
    n_calls = 4

    fx, fy = (0.05, 0.05)
    def objective(params):
        nonlocal run_iter
        print(f"BO Run: {run_iter+1}/{n_calls}")
        result = objective_function(params, fx, fy, training_df, device, run_iter)
        run_iter += 1
        return result

    res = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_initial_points=2,
        random_state=0,
    )

    print("Bayesian optimization done:")
    print(f"Params: {res.x}, rTRE: {res.fun}")
    return 0 

if __name__ == "__main__":
    main()
