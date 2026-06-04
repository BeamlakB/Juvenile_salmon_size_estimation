import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import itertools
import random
from pathlib import Path
import os

#enhance image 

#enhance image
# ─────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────

def clahe_white_balance(img_rgb, clip=3.0, grid=(8, 8)):
    """Higher clipLimit (3.0 vs 2.0) = more aggressive brightening."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    channels = cv2.split(img_rgb)
    return cv2.merge([clahe.apply(c) for c in channels])

def gamma_correction(img_rgb, gamma=1.8):
    """
    gamma > 1 → brightens dark regions (good for underexposed images)
    gamma < 1 → darkens bright regions
    Try: 1.5, 1.8, 2.2 for increasingly strong brightening
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img_rgb, table)

def gaussian_denoise(img_rgb, kernel_size=(5, 5), sigma=1.2):
    return cv2.GaussianBlur(img_rgb, kernel_size, sigma)

def stochastic_enhance(img_rgb, seed=42,
                        sat_range=(0.9, 1.2),
                        bright_range=(1, 1.5),
                        contrast_range=(1.0, 1.3),
                        sharp_range=(1.2, 2.0)):
    """
    Randomly adjusts saturation, brightness, contrast, and sharpness.
    Set seed for reproducibility; change seed to get different variants.
    """
    random.seed(seed)
    pil_img = Image.fromarray(img_rgb)

    sat    = random.uniform(*sat_range)
    bright = random.uniform(*bright_range)
    cont   = random.uniform(*contrast_range)
    sharp  = random.uniform(*sharp_range)

    pil_img = ImageEnhance.Color(pil_img).enhance(sat)
    pil_img = ImageEnhance.Brightness(pil_img).enhance(bright)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(cont)
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(sharp)

    params = dict(saturation=round(sat,2), brightness=round(bright,2),
                  contrast=round(cont,2), sharpness=round(sharp,2))
    return np.array(pil_img), params


#gama1.8+galhe+gausse
def best_dark_pipeline(img_rgb, seed=512):
    # Step 1: Stochastic (adds warmth + contrast like seed=512 did)
    stoch, _ = stochastic_enhance(img_rgb, seed=seed)
    
    # Step 2: Gamma to lift the dark regions
    gamma = gamma_correction(stoch, gamma=1.8)
    
    # Step 3: CLAHE for local contrast
    cl = clahe_white_balance(gamma, clip=2.0)
    
    # Step 4: Gaussian to smooth noise
    return gaussian_denoise(cl)


def save_best_seed(image_paths, seed, output_dir="seed_results"):
    """Once you pick the best seed, save all processed images."""
    os.makedirs(output_dir, exist_ok=True)
    for img_path in image_paths:
        img_cv  = cv2.imread(str (img_path))
        if img_cv is None:
            continue
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        result = best_dark_pipeline(img_rgb)
        out_name = f"{Path(img_path).stem}_seed{seed}.png"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"Saved: {out_path} ")

if __name__ == "__main__":
    right_images = sorted(Path("HD720_SN15044540_14-44-20/right").glob("*.png"))
    left_images  = sorted(Path("HD720_SN15044540_14-44-20/left").glob("*.png"))
    
    outut_dir_right = "HD720_SN15044540_14-44-20/masked_image/right"
    outut_dir_left = "HD720_SN15044540_14-44-20/masked_image/left"
    #print (all_frames)
    save_best_seed(right_images, seed=512, output_dir= outut_dir_right)
    save_best_seed(left_images, seed=512, output_dir= outut_dir_left)

