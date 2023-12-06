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

def get_middle_four(lst):
    middle_index = len(lst) // 2

    # If the list has an odd number of elements
    # adjust the middle index to lean left
    if len(lst) % 2 == 1:
        middle_index -= 1

    # Get the middle 4 elements
    middle_four = []
    for i in range(middle_index - 1, middle_index + 3):
        middle_four.append(i)

    assert len(middle_four) == 4

    return middle_four


def convert_coco_to_mask(input_json, image_folder,
                         output_mask_folder, output_image_folder,
                         output_txt_file_path, image_relative_path,
                         mask_relative_path):
    # Load COCO annotations
    coco = COCO(input_json)

    # Create output folder if it doesn't exist
    os.makedirs(output_mask_folder, exist_ok=True)
    os.makedirs(output_image_folder, exist_ok=True)

    img_ids = coco.getImgIds()

    # create the train_gt.txt file
    with open(output_txt_file_path, "w") as f:
        pass

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
            polygon_list = [polygon for polygon in segmentation]

        # the 1st and 4th index of each polygon are the two points user selects
        # in annotation tool to draw the rectangle
        mid_point_list = [(i[1] + i[4])/2 for i in polygon_list]
        indexes = [i for i in range(len(mid_point_list))]
        # sort the mid_point_list in ascending order and accordingly sort the indexes list

        # Zip the lists, sort based on values in list a, and unzip them
        sorted_pairs = sorted(zip(mid_point_list, indexes), key=lambda x: x[0])
        mid_point_list_sorted, indexes_sorted = map(list, zip(*sorted_pairs))

        if len(mid_point_list_sorted) > 4:
            selected_polygon_indexes = get_middle_four(indexes_sorted)
        else:
            selected_polygon_indexes = indexes_sorted

        # pad the selected_polygon_indexes with 0s if len < 4
        if len(selected_polygon_indexes) < 4:
            selected_polygon_indexes = selected_polygon_indexes + ((4 - len(selected_polygon_indexes)) * [0])

        for i in selected_polygon_indexes:
            # Draw polygon on the mask
            draw.polygon(polygon_list[i], fill=255)

        # Save the selected image and mask image
        mask_path = os.path.join(output_mask_folder, f"{img_info['file_name'].split('.')[0].split('/')[1]}_mask.png")
        image_path = os.path.join(output_image_folder, f"{img_info['file_name'].split('.')[0].split('/')[1]}.png")
        mask.save(mask_path)
        img.save(image_path)

        # write image_file_path, mask_file_path, and lane_count to the .txt file
        # lane count is like 0 1 0 0 = 1 lane and other 3 are not visible
        # in our case, we count lanes from left to right and we fill accordingly
        with open(output_txt_file_path, "a") as f:
            sel = selected_polygon_indexes
            f.write(f"{image_relative_path} {mask_relative_path} {sel[0]},{sel[1]},{sel[2]},{sel[3]} \n")


if __name__ == "__main__":
    # Specify the path to COCO annotations and image folder
    coco_annotation_json = "/home/sush/klab2/rosbags_collated/round_2/annotations.json"
    image_folder_path = "/home/sush/klab2/rosbags_collated/round_2/"

    # Specify the output folder for mask images
    output_mask_folder = "/home/sush/klab2/rosbags_collated/round_2/masks-clean"
    output_image_folder = "/home/sush/klab2/rosbags_collated/round_2/images-clean"
    output_image_relative_path = "/round_2/images-clean"
    output_mask_reltive_path = "/round_2/masks-clean"
    output_txt_file_path = "/home/sush/klab2/rosbags_collated/round_2/train_gt.txt"

    convert_coco_to_mask(coco_annotation_json, image_folder_path, output_mask_folder,
                         output_image_folder, output_txt_file_path, output_image_relative_path,
                         output_mask_reltive_path)
