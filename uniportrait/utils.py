import os
from io import BytesIO

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler, AutoencoderKL, ControlNetModel, StableDiffusionControlNetPipeline, LCMScheduler
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from diffusers.utils import load_image
import logging
from torch import nn



def pad_np_bgr_image(np_image, scale=1.25):
    assert scale >= 1.0, "scale should be >= 1.0"
    pad_scale = scale - 1.0
    h, w = np_image.shape[:2]
    top = bottom = int(h * pad_scale)
    left = right = int(w * pad_scale)
    ret = cv2.copyMakeBorder(np_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return ret, (left, top)

def process_faceid_image(pil_faceid_image, face_app):
    np_faceid_image = np.array(pil_faceid_image.convert("RGB"))
    img = cv2.cvtColor(np_faceid_image, cv2.COLOR_RGB2BGR)
    faces = face_app.get(img)  # bgr
    if len(faces) == 0:
        # padding, try again
        _h, _w = img.shape[:2]
        _img, left_top_coord = pad_np_bgr_image(img)
        faces = face_app.get(_img)
        if len(faces) == 0:
            gr.Info("Warning: No face detected in the image. Continue processing...")

        min_coord = np.array([0, 0])
        max_coord = np.array([_w, _h])
        sub_coord = np.array([left_top_coord[0], left_top_coord[1]])
        for face in faces:
            face.bbox = np.minimum(np.maximum(face.bbox.reshape(-1, 2) - sub_coord, min_coord), max_coord).reshape(4)
            face.kps = face.kps - sub_coord

    faces = sorted(faces, key=lambda x: abs((x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])), reverse=True)
    faceid_face = faces[0]
    norm_face = face_align.norm_crop(img, landmark=faceid_face.kps, image_size=224)
    pil_faceid_align_image = Image.fromarray(cv2.cvtColor(norm_face, cv2.COLOR_BGR2RGB))

    return pil_faceid_align_image

def prepare_single_faceid_cond_kwargs(pil_faceid_image=None, pil_faceid_supp_images=None,
                                      pil_faceid_mix_images=None, mix_scales=None,
                                      face_app = None):
    pil_faceid_align_images = []
    if pil_faceid_image:
        pil_faceid_align_images.append(process_faceid_image(pil_faceid_image, face_app))
    # if pil_faceid_supp_images and len(pil_faceid_supp_images) > 0:
    #     for pil_faceid_supp_image in pil_faceid_supp_images:
    #         if isinstance(pil_faceid_supp_image, Image.Image):
    #             pil_faceid_align_images.append(process_faceid_image(pil_faceid_supp_image))
    #         else:
    #             pil_faceid_align_images.append(
    #                 process_faceid_image(Image.open(BytesIO(pil_faceid_supp_image)))
    #             )

    mix_refs = []
    mix_ref_scales = []
    if pil_faceid_mix_images:
        for pil_faceid_mix_image, mix_scale in zip(pil_faceid_mix_images, mix_scales):
            if pil_faceid_mix_image:
                mix_refs.append(process_faceid_image(pil_faceid_mix_image, face_app))
                mix_ref_scales.append(mix_scale)

    single_faceid_cond_kwargs = None
    if len(pil_faceid_align_images) > 0:
        single_faceid_cond_kwargs = {
            "refs": pil_faceid_align_images
        }
        if len(mix_refs) > 0:
            single_faceid_cond_kwargs["mix_refs"] = mix_refs
            single_faceid_cond_kwargs["mix_scales"] = mix_ref_scales

    return single_faceid_cond_kwargs



import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

# from pytorch_lightning import seed_everything

import dlib
from PIL import Image, ImageDraw

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def draw_landmarks(image, landmarks, color="white", radius=2.5):
    draw = ImageDraw.Draw(image)
    for dot in landmarks:
        x, y = dot
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)

def get_68landmarks_img(img, detector, predictor):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)
    landmarks = []
    for face in faces:
        shape = predictor(gray, face)
        for i in range(68):
            x = shape.part(i).x
            y = shape.part(i).y
            landmarks.append((x, y))
    con_img = Image.new('RGB', (img.shape[1], img.shape[0]), color=(0, 0, 0))
    draw_landmarks(con_img, landmarks)
    con_img = np.array(con_img)
    return con_img

def dlib_process(input_image, detector= None, predictor = None, landmark_direct_mode = False, ):#input_image np array

    if detector == None:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")

    img = HWC3(input_image)
    H, W, C = img.shape

    if landmark_direct_mode:
        detected_map = img
    else:
        detected_map = get_68landmarks_img(img, detector, predictor)
    detected_map = HWC3(detected_map)

    return detected_map


def histogram_matching(source, target):
    assert source.size() == target.size(), "Source and target must have the same shape"
    B, C, H, W = source.shape
    matched_output = torch.zeros_like(source)

    for b in range(B):  # Loop through each image in the batch
        for c in range(C):  # Loop through each channel
            # Flatten and mask non-zero elements for both source and target
            source_flat = source[b, c].flatten().numpy()
            target_flat = target[b, c].flatten().numpy()

            # Filter out zeros
            source_nonzero = source_flat[source_flat != 0]
            target_nonzero = target_flat[target_flat != 0]

            # Calculate histograms and cumulative distributions
            source_hist, source_bins = np.histogram(source_nonzero, bins=256, range=(source_nonzero.min(), source_nonzero.max()))
            source_cdf = np.cumsum(source_hist).astype(float)
            source_cdf /= source_cdf[-1]  # Normalize

            target_hist, target_bins = np.histogram(target_nonzero, bins=256, range=(target_nonzero.min(), target_nonzero.max()))
            target_cdf = np.cumsum(target_hist).astype(float)
            target_cdf /= target_cdf[-1]  # Normalize

            # Interpolate
            interp_values = np.interp(source_nonzero, source_bins[:-1], target_cdf)

            # Map values to target values using interpolated CDF positions
            source_values_to_target = np.interp(interp_values, target_cdf, target_bins[:-1])

            # Place mapped values back into the source
            source_matched = source_flat.copy()
            nonzero_indices = source_flat != 0
            source_matched[nonzero_indices] = source_values_to_target

            # Convert back to tensor and reshape
            matched_output[b, c] = torch.tensor(source_matched, dtype=torch.float32).view(H, W)
    
            

    return matched_output, nn.MSELoss(reduction = 'sum')(matched_output, source)/torch.sum(matched_output != 0)


def getLogger(log_dir, basename = 'logfile.log'):
    # Create a logger
    logger = logging.getLogger(__name__)  # Use the current module's name for the logger

    # Set the minimum severity level for logging
    logger.setLevel(logging.INFO)  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

    # Create a console handler (to print logs to the console)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a file handler (to write logs to a file)
    file_handler = logging.FileHandler(os.path.join(log_dir, basename))
    file_handler.setLevel(logging.INFO)

    # Define a log message format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
