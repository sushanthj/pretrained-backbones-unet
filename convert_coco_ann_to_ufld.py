import os
import json
import numpy as np
import cv2
from pycocotools.coco import COCO
from tqdm import tqdm
from PIL import Image, ImageDraw
import ipdb
import sys

IMAGE_CENTER_X = 640

def flatten_segmentation(segmentation):
    # Flatten the segmentation list
    return [coord for polygon in segmentation for coord in polygon]

def get_middle_four(mid_point_list, index_list):
    """
    Args:
             mid_point_list : sorted list of mid points of the polygons
                 index_list : indexes of original list of polygons after sorting

    Returns:
        filtered_index_list : the final lane indexes to be selected
    """
    # filter the images based on whether the mid point is to left or right of image center
    # and filter the index list as well
    left_half_mid_points = []
    right_half_mid_points = []
    left_half_indexes = []
    right_half_indexes = []

    for i in range(len(mid_point_list)):
        if mid_point_list[i] < IMAGE_CENTER_X:
            left_half_mid_points.append(mid_point_list[i])
            left_half_indexes.append(index_list[i])

    for i in range(len(mid_point_list)):
        if mid_point_list[i] > IMAGE_CENTER_X:
            right_half_mid_points.append(mid_point_list[i])
            right_half_indexes.append(index_list[i])

    # if there are no points on the left half, select the 4 points on the right half
    if len(left_half_mid_points) == 0:
        return right_half_indexes[:4]

    # if there are no points on the right half, select the 4 points on the left half
    if len(right_half_mid_points) == 0:
        return left_half_indexes[-4:]

    # if there is only 1 point on the left half, select the 3 points on the right half closest to the image center
    if len(left_half_mid_points) == 1:
        return left_half_indexes + right_half_indexes[:3]

    # if there is only 1 point on the right half, select the 3 points on the left half closest to the image center
    if len(right_half_mid_points) == 1:
        return left_half_indexes[-3:] + right_half_indexes

    else:
        # if there are points on both halves, select the 2 points on each half closest to the image center
        # and return the indexes of those points
        final_indexes = left_half_indexes[-2:] + right_half_indexes[:2]
        return final_indexes


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

        # skip image if it has not been annotated yet
        if len(annotations) == 0:
            continue

        # Draw each annotation on the mask image
        draw = ImageDraw.Draw(mask)

        for ann in annotations:
            segmentation = ann['segmentation'] # list of polygons
            category_id = ann['category_id']

            # segmentation_flattened = flatten_segmentation(segmentation)
            polygon_list = [polygon for polygon in segmentation]

        # the 1st and 4th index of each polygon are the two points user selects
        # in annotation tool to draw the rectangle
        # NOTE: i[2] = x1, i[8] = x2
        # mid_point_list = [(i[2] + i[8])/2 for i in polygon_list]
        mid_point_list = []
        for i in polygon_list:
            if i[3] > i[9]:
                mid_point_list.append(i[2])
            else:
                mid_point_list.append(i[8])
        indexes = [i for i in range(len(mid_point_list))]
        # sort the mid_point_list in ascending order and accordingly sort the indexes list

        # Zip the lists, sort based on values in list a, and unzip them
        sorted_pairs = sorted(zip(mid_point_list, indexes), key=lambda x: x[0])
        mid_point_list_sorted, indexes_sorted = map(list, zip(*sorted_pairs))

        if len(mid_point_list_sorted) > 4:
            selected_polygon_indexes = get_middle_four(mid_point_list_sorted, indexes_sorted)
        else:
            selected_polygon_indexes = indexes_sorted

        # pad the selected_polygon_indexes with 0s if len < 4
        if len(selected_polygon_indexes) < 4:
            # selected_polygon_indexes = selected_polygon_indexes + ((4 - len(selected_polygon_indexes)) * [0])
            #! SKIPPING IMAGES IF LESS THAN 4 LANES FOR NOW
            continue

        for i in selected_polygon_indexes:
            # Draw polygon on the mask
            draw.polygon(polygon_list[i], fill=1) # fill must be one if not torch.NLL gives errors

        # Save the selected image and mask image
        mask_filename = f"{img_info['file_name'].split('.')[0].split('/')[1]}_mask.png"
        image_filename = f"{img_info['file_name'].split('.')[0].split('/')[1]}.png"
        mask_path = os.path.join(output_mask_folder, mask_filename)
        image_path = os.path.join(output_image_folder, image_filename)
        mask.save(mask_path)
        img.save(image_path)

        # Resize images to match CULane shapes
        img = cv2.imread(mask_path)
        img = cv2.resize(img, (1640, 590))
        cv2.imwrite(mask_path, img)

        img = cv2.imread(image_path)
        img = cv2.resize(img, (1640, 590))
        cv2.imwrite(image_path, img)

        with open(output_txt_file_path, "a") as file:
            file.write(f"{os.path.join(image_relative_path,image_filename)} {os.path.join(mask_relative_path, mask_filename)} {1} {1} {1} {1} \n")


if __name__ == "__main__":
    # Specify the path to COCO annotations and image folder
    coco_annotation_json = "/home/sush/klab2/rosbags_collated/round_2_test/annotations.json"
    image_folder_path = "/home/sush/klab2/rosbags_collated/round_2_test/"

    # Specify the output folder for mask images
    output_mask_folder = "/home/sush/klab2/rosbags_collated/round_2_test/masks-clean"
    output_image_folder = "/home/sush/klab2/rosbags_collated/round_2_test/images-clean"
    output_image_relative_path = "/round_2_test/images-clean"
    output_mask_reltive_path = "/round_2_test/masks-clean"
    output_txt_file_path = "/home/sush/klab2/rosbags_collated/round_2_test/train_gt.txt"

    # NOTE: SET IMAGE CENTER X HERE

    convert_coco_to_mask(coco_annotation_json, image_folder_path, output_mask_folder,
                         output_image_folder, output_txt_file_path, output_image_relative_path,
                         output_mask_reltive_path)
