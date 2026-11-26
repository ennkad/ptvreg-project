import sys

import nibabel as nib
import numpy as np
import torch
import cv2
import pandas as pd 

from ptv import register_pair_min

def load_data(path_to_img, path_to_landmark):
    data_folder = f"/mnt/asgard0/data/y2026-p13-Data"
    img = cv2.imread(data_folder + '/ANHIR/' + path_to_img, cv2.IMREAD_GRAYSCALE)
    landmarks = pd.read_csv(data_folder +'/Landmarks/' +  path_to_landmark)
    if img is None:
        raise FileNotFoundError(path_to_img)
    if landmarks is None:
        raise FileNotFoundError(path_to_landmarks)
    return img, landmarks

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


def preprocess_img(img, landmarks):
    img = sobel(img, ksize=7)
    img = cv2.medianBlur(img, 5)
    img = downsample(img, landmarks, fx, fy)
    return img

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


def compute_metric(fixed_path, moving_path, fixed_landmarks_path, moving_landmarks_path, params=None):

    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    ## 3d - ignore 
    # fix_t, fix_nii = load(sys.argv[1], device, dtype)
    # mov_t, _ = load(sys.argv[2], device, dtype)
    # spacing = tuple(float(z) for z in fix_nii.header.get_zooms()[:3])

    # 2d overwrite 
    fix_img, fix_landmarks = load_data(fixed_path, fixed_landmarks_path)
    mov_img = load_data(moving_path, moving_landmarks_path)
    spacing = (1.0, 1.0)

    # ------- Preprocessing ---------
    fix_img = preprocess_img(fix_img, fix_landmark)
    cv2.imwrite("fix_img.jpg", fix_img)

    mov_img = preprocess_img(mov_img, mov_landmarks)
    cv2.imwrite("mov_img.jpg", mov_img)

    fix_img, mov_img = add_padding(fix_img, mov_img, fix_landmarks, mov_landmarks)

    # ------ Create tensor ---------
    fix_t_tens = img_to_tensor(fix_img, device, dtype)
    mov_t_tens = img_to_tensor(mov_img, device, dtype)

    opts = {
        "pix_resolution": spacing,
        "grid_spacing": [8, 8], #
        "cp_refinements": 1,
        "k_down": 0.7,
        "metric": "lcc",
        "metric_param": (2.1, 2.1, 2.1), #
        "scale_metric_param": True,
        "isoTV": 0.11, #
        "csqrt": 5e-3,
        "border_mask": 5,
        "opt_method": "admm",
        "jac_reg": 0.0,
        "jac_reg_dense": 2.0,
        "device": device,
        "verbose": True,
    }
    
    _voldef, disp_mm, _knots, _infos = register_pair_min(mov_t_tens, fix_t_tens, opts)
    
    voldef_img = tensor_to_img(_voldef)
    cv2.imwrite("voldef.jpg", voldef_img)

    disp = disp_mm[0].detach().cpu().numpy().astype(np.float32)
    for ax in range(2):
        disp[ax] /= spacing[ax]

    print(_infos[-1].total_loss_norm)
    return _infos[-1]

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

    training_df = load_training_csv("dataset_medium.csv")
    training_df = filter_training_data(training_df, 8)
    evaluation_df = load_evaluation_csv("dataset_medium.csv")
    if training_df["status"].unique() == "training":
        print(f"Successfully loaded training dataset. ({training_df.shape[0]} rows)")
    if evaluation_df["status"].unique() == "evaluation":
        print(f"Successfully loaded evaluation dataset. ({evaluation_df.shape[0]} rows)")

    out_df = pd.DataFrame(columns=['Source image', 'Target image', 'loss']) 

    for _, row in training_df.iterrows():
        fixed = row.loc["Target image"]
        fix_landmarks = row.loc["Target landmarks"]
        moving = row.loc["Source image"]
        mov_landmarks = row.loc["Source landmarks"]
        
        # ladda in data 

        infos = compute_metric(fixed, moving, fix_landmarks, mov_landmarks) # params? 
        # new_row = pd.DataFrame({"Name": ["Eve"], "Age": [28]}) # byt till våra kolumner lol 
        # out_df = pd.concat([out_df, new_row], ignore_index=True)

    return 0 

if __name__ == "__main__":
    main()
