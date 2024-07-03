import gradio as gr
from utils import *
from segment_anything import sam_model_registry, SamPredictor
import os
import numpy as np
import cv2
import rawpy
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from pathlib import Path

sam_checkpoint = os.path.join("models", "sam_vit_h_4b8939.pth")
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

def init_tab2():
    try:
        image = cv2.imread(os.path.join("outputs", "output_tab1", "input_WB_new.png"))
        cv2.imwrite(os.path.join("outputs", "input_tab3", "image_0.png"), image) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)              
        input_image = gr.Image(value=image)
        return input_image
    except:
        input_image = gr.Image()
        return input_image
    
def add_point(input_image, x, y):
    predictor.set_image(input_image)
    # check file config    
    if not os.path.isfile(os.path.join("outputs", "temps", "point_tab2.npy")):
        save_point = np.array([[x, y]])
        save_label = np.array([1])
        with open(os.path.join("outputs", "temps", "point_tab2.npy"), 'wb') as f:
            np.save(f, save_point)
            np.save(f, save_label)
    else:
        with open(os.path.join("outputs", "temps", "point_tab2.npy"), 'rb') as f:
            save_point = np.load(f)
            save_label = np.load(f)
        save_point = np.concatenate((save_point, np.array([[x, y]])), axis=0)
        save_label = np.concatenate((save_label, np.array([1])), axis=0)
        with open(os.path.join("outputs", "temps", "point_tab2.npy"), 'wb') as f:
            np.save(f, save_point)
            np.save(f, save_label)
    # input_point = np.array([[x, y]])
    # input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=save_point,
        point_labels=save_label,
        multimask_output=True,
    )
    max_score = 0
    max_mask = None
    for (mask, score) in (zip(masks, scores)):
        if score > max_score:
            max_score = score
            max_mask = mask
    plt.imshow(input_image)
    show_mask(max_mask, plt.gca())
    show_points(save_point, save_label, plt.gca())
    plt.axis('scaled')
    plt.axis('off')
    plt.gcf().set_size_inches(input_image.shape[1]/plt.rcParams['figure.dpi'], input_image.shape[0]/plt.rcParams['figure.dpi'])
    
    plt.savefig(os.path.join("outputs", "temps", "tab2.png"),bbox_inches='tight', pad_inches=0)
    predict_image = cv2.imread(os.path.join("outputs", "temps", "tab2.png"))
    predict_image = cv2.cvtColor(predict_image, cv2.COLOR_BGR2RGB)
    return predict_image

def segment_func(input_image, method, evt: gr.SelectData):
    # if method == "None":
    #     output_image = gr.Image()
    # if method == "add_point":
    #     output_image = add_point(input_image, evt.index[0], evt.index[1])
    predictor.set_image(input_image)
    # check file config    
    x = evt.index[0]
    y = evt.index[1]
    if not os.path.isfile(os.path.join("outputs", "temps", "point_tab2.npy")):
        save_point = np.array([[x, y]])
        if method == "add_point":
            save_label = np.array([1])
        if method == "remove_point":
            save_label = np.array([0])
        with open(os.path.join("outputs", "temps", "point_tab2.npy"), 'wb') as f:
            np.save(f, save_point)
            np.save(f, save_label)
    else:
        with open(os.path.join("outputs", "temps", "point_tab2.npy"), 'rb') as f:
            save_point = np.load(f)
            save_label = np.load(f)
        save_point = np.concatenate((save_point, np.array([[x, y]])), axis=0)
        if method == "add_point":
            save_label = np.concatenate((save_label, np.array([1])), axis=0)
        if method == "remove_point":
            save_label = np.concatenate((save_label, np.array([0])), axis=0)
        with open(os.path.join("outputs", "temps", "point_tab2.npy"), 'wb') as f:
            np.save(f, save_point)
            np.save(f, save_label)
    # input_point = np.array([[x, y]])
    # input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=save_point,
        point_labels=save_label,
        multimask_output=True,
    )
    max_score = 0
    max_mask = None
    for (mask, score) in (zip(masks, scores)):
        if score > max_score:
            max_score = score
            max_mask = mask
    plt.imshow(input_image)
    show_mask(max_mask, plt.gca())
    show_points(save_point, save_label, plt.gca())
    plt.axis('scaled')
    plt.axis('off')
    plt.gcf().set_size_inches(input_image.shape[1]/plt.rcParams['figure.dpi'], input_image.shape[0]/plt.rcParams['figure.dpi'])
    
    plt.savefig(os.path.join("outputs", "temps", "tab2.png"),bbox_inches='tight', pad_inches=0)
    plt.close()
    predict_image = cv2.imread(os.path.join("outputs", "temps", "tab2.png"))
    predict_image = cv2.cvtColor(predict_image, cv2.COLOR_BGR2RGB)
    return predict_image

def apply_func(input_image):
    predictor.set_image(input_image)
    with open(os.path.join("outputs", "temps", "point_tab2.npy"), 'rb') as f:
        save_point = np.load(f)
        save_label = np.load(f)
    masks, scores, logits = predictor.predict(
        point_coords=save_point,
        point_labels=save_label,
        multimask_output=True,
    )
    max_score = 0
    max_mask = None
    for (mask, score) in (zip(masks, scores)):
        if score > max_score:
            max_score = score
            max_mask = mask
    # dilate mask to avoid unmasked edge effect
    # if args.dilate_kernel_size is not None:
    #     masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]
    max_mask = max_mask.astype(np.uint8) * 255
    max_mask = dilate_mask(max_mask)
    
    # for idx, mask in enumerate(masks):
        # mask_p = out_dir / f"mask_{idx}.png"
        # img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
    img_inpainted_p = os.path.join("outputs", "output_tab2", "remove.png")
    # print(input_image.shape)
    # print(max_mask.shape)
    img_inpainted = inpaint_img_with_lama(
        input_image, max_mask, "Inpaint_Anything/lama/configs/prediction/default.yaml", "models/big-lama", device=device)
    save_array_to_img(img_inpainted, img_inpainted_p)
    image = cv2.imread(os.path.join("outputs", "output_tab2", "remove.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return gr.Image(value=image)
    
def undo_func(input_image):
    input_image = cv2.imread(os.path.join("outputs", "input_tab2", "image_0.png"))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    with open(os.path.join("outputs", "temps", "point_tab2.npy"), 'rb') as f:
        save_point = np.load(f)
        save_label = np.load(f)
    os.remove(os.path.join("outputs", "temps", "point_tab2.npy"))
    save_point = np.delete(save_point, len(save_point)-1, 0)
    save_label = np.delete(save_label, len(save_label)-1, 0)
    with open(os.path.join("outputs", "temps", "point_tab2.npy"), 'wb') as f:
            np.save(f, save_point)
            np.save(f, save_label)        
    masks, scores, logits = predictor.predict(
        point_coords=save_point,
        point_labels=save_label,
        multimask_output=True,
    )
    max_score = 0
    max_mask = None
    for (mask, score) in (zip(masks, scores)):
        if score > max_score:
            max_score = score
            max_mask = mask
    plt.imshow(input_image)
    show_mask(max_mask, plt.gca())
    show_points(save_point, save_label, plt.gca())
    plt.axis('scaled')
    plt.axis('off')
    plt.gcf().set_size_inches(input_image.shape[1]/plt.rcParams['figure.dpi'], input_image.shape[0]/plt.rcParams['figure.dpi'])
    
    plt.savefig(os.path.join("outputs", "temps", "tab2.png"),bbox_inches='tight', pad_inches=0)
    plt.close()
    predict_image = cv2.imread(os.path.join("outputs", "temps", "tab2.png"))
    predict_image = cv2.cvtColor(predict_image, cv2.COLOR_BGR2RGB)
    return predict_image
    
def reset_func():
    # for image_file in os.listdir(os.path.join("outputs", "input_tab2")):
    #     if image_file != "image_0.png":
    #         os.remove(os.path.join("outputs", "input_tab2", image_file))
    try:
        os.remove(os.path.join("outputs", "temps", "point_tab2.npy"))
    except:
        pass
    image = cv2.imread(os.path.join("outputs", "input_tab2", f"image_0.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return gr.Image(value=image)

def upload_file(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join("outputs", "input_tab2", "image_0.png"), image)