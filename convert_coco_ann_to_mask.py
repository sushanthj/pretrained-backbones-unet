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

def convert_coco_to_mask(input_json, image_folder, output_folder):
    # Load COCO annotations
    coco = COCO(input_json)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

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

        # Draw each annotation on the mask image
        draw = ImageDraw.Draw(mask)
        for ann in annotations:
            segmentation = ann['segmentation'] # list of polygons
            category_id = ann['category_id']

            # segmentation_flattened = flatten_segmentation(segmentation)
            for polygon in segmentation:
                try:
                    # Draw polygon on the mask
                    draw.polygon(polygon, fill=255)
                except:
                    pass
                    # ipdb.set_trace()
                    # print("polygon \n", polygon)
                    # sys.exit(0)


        # Save the mask image
        mask_path = os.path.join(output_folder, f"{img_info['file_name'].split('.')[0].split('/')[1]}_mask.png")
        mask.save(mask_path)


if __name__ == "__main__":
    # Specify the path to COCO annotations and image folder
    coco_annotation_json = "/home/sush/klab2/rosbags_collated/sensors_2023-08-03-15-19-03_0/annotations.json"
    image_folder_path = "/home/sush/klab2/rosbags_collated/sensors_2023-08-03-15-19-03_0"

    # Specify the output folder for mask images
    output_mask_folder = "/home/sush/klab2/rosbags_collated/sensors_2023-08-03-15-19-03_0/masks"

    convert_coco_to_mask(coco_annotation_json, image_folder_path, output_mask_folder)
