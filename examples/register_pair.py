import sys

import nibabel as nib
import numpy as np
import torch
import cv2
import pandas as pd 

from ptv import register_pair_min

def load_img(path):
    data_folder = f"/mnt/asgard0/data/y2026-p13-Data/ANHIR/"
    img = cv2.imread(data_folder + path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img

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

def down_sample_img(img):
    return cv2.resize(img, fx=0.5, fy=0.5 interpolation=cv2.INTER_AREA)

def preprocess_img(img):
    img = sobel(img, ksize=7)
    img = cv2.medianBlur(img, 5)
    img = down_sample_img(img)
    return img

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


def compute_metric(fixed, moving, params=None):

    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    ## 3d - ignore 
    # fix_t, fix_nii = load(sys.argv[1], device, dtype)
    # mov_t, _ = load(sys.argv[2], device, dtype)
    # spacing = tuple(float(z) for z in fix_nii.header.get_zooms()[:3])

    # 2d overwrite 
    fix_img = load_img(fixed)
    mov_img = load_img(moving)
    spacing = (1.0, 1.0)

    # ------- Preprocessing ---------
    fix_img = preprocess_img(fix_img)
    cv2.imwrite("fix_img.jpg", fix_img)

    mov_img = preprocess_img(mov_img)
    cv2.imwrite("mov_img.jpg", mov_img)

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


def main():


    training_df = load_training_csv("dataset_medium.csv")
    evaluation_df = load_evaluation_csv("dataset_medium.csv")
    if training_df["status"].unique() == "training":
        print(f"Successfully loaded training dataset. ({training_df.shape[0]} rows)")
    if evaluation_df["status"].unique() == "evaluation":
        print(f"Successfully loaded evaluation dataset. ({evaluation_df.shape[0]} rows)")

    out_df = pd.DataFrame(columns=['Source image', 'Target image', 'loss']) 

    src = "breast_1/scale-20pc/HE.jpg"
    target = "breast_1/scale-20pc/ER.jpg"
    infos = compute_metric(src, target)

    # for row in training_df.rows:
    #     src = row.loc["Source image"]
    #     target = row.loc["Target image"]
    #     infos = compute_metric(src, target, params) # params? 
    #     # new_row = pd.DataFrame({"Name": ["Eve"], "Age": [28]}) # byt till våra kolumner lol 
    #     # out_df = pd.concat([out_df, new_row], ignore_index=True)

    return 0 

if __name__ == "__main__":
    main()
