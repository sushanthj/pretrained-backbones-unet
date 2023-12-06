import os
import json
import numpy as np
import cv2
from pycocotools.coco import COCO
from tqdm import tqdm
from PIL import Image, ImageDraw
import ipdb
import sys


def flatten_segmentation(segmentation):
    # Flatten the segmentation list
    return [coord for polygon in segmentation for coord in polygon]

def convert_coco_to_mask(input_json, image_folder,
                         output_mask_folder, output_image_folder,
                         output_txt_file_path, image_relative_path,
                         mask_relative_path):
    # Load COCO annotations
    coco = COCO(input_json)

    # Create output folder if it doesn't exist
    os.makedirs(output_mask_folder, exist_ok=True)
    os.makedirs(output_image_folder, exist_ok=True)

    # create the train_gt.txt file
    with open(output_txt_file_path, "w") as f:
        pass

    img_ids = coco.getImgIds()

    # Loop through each image in the COCO dataset
    for img_id in tqdm(img_ids, desc="Converting to Mask", unit="image"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_folder, img_info['file_name'])

        # Load image using PIL
        img = Image.open(img_path).convert('RGB')

        # Create a blank mask image
        mask = Image.new('L', img.size, 0)

        # Get annotations for the current image
        ann_ids = coco.getAnnIds(imgIds=img_info['id'])
        annotations = coco.loadAnns(ann_ids)

        # skip image if it has not been annotated yet
        if len(annotations) == 0:
            continue

        # Draw each annotation on the mask image
        draw = ImageDraw.Draw(mask)

        for ann in annotations:
            segmentation = ann['segmentation'] # list of polygons
            category_id = ann['category_id']

            # segmentation_flattened = flatten_segmentation(segmentation)
            for polygon in segmentation:
                # Draw polygon on the mask
                draw.polygon(polygon, fill=255)

        # Save the selected image and mask image
        mask_filename = f"{img_info['file_name'].split('.')[0].split('/')[1]}_mask.png"
        image_filename = f"{img_info['file_name'].split('.')[0].split('/')[1]}.png"
        mask_path = os.path.join(output_mask_folder, mask_filename)
        image_path = os.path.join(output_image_folder, image_filename)
        mask.save(mask_path)
        img.save(image_path)

        with open(output_txt_file_path, "a") as file:
            file.write(f"{os.path.join(image_relative_path,image_filename)} {os.path.join(mask_relative_path, mask_filename)} \n")


if __name__ == "__main__":
    # Specify the path to COCO annotations and image folder
    coco_annotation_json = "/home/sush/klab2/rosbags_collated/round_3/annotations.json"
    image_folder_path = "/home/sush/klab2/rosbags_collated/round_3/"

    # Specify the output folder for mask images
    output_mask_folder = "/home/sush/klab2/rosbags_collated/round_3/masks-clean"
    output_image_folder = "/home/sush/klab2/rosbags_collated/round_3/images-clean"
    output_image_relative_path = "round_3/images-clean"
    output_mask_reltive_path = "round_3/masks-clean"
    output_txt_file_path = "/home/sush/klab2/rosbags_collated/round_3/train_gt.txt"

    convert_coco_to_mask(coco_annotation_json, image_folder_path, output_mask_folder,
                         output_image_folder, output_txt_file_path, output_image_relative_path,
                         output_mask_reltive_path)
