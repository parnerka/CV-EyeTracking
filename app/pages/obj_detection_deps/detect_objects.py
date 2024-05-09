# This script is used to detect objects in the frames of a video or a set of images using the GroundingDINO model.
# The script is used in the object detection page of the application

import sys
sys.path.append("C:\\Users\\ARL")   # Path where the GroundingDINO library was cloned - The library won't be detected if this is not set properly
import numpy as np
import pandas as pd
from datetime import datetime
import cv2
import os
import torch
import csv
from time import sleep
from stqdm import stqdm
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict, load_model
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    # Device to run the model

# Path to model weights
groundingdino_model = load_model("C:\\Users\\ARL\\GroundingDINO\\groundingdino\\config\\GroundingDINO_SwinT_OGC.py", "C:\\Users\\ARL\\GroundingDINO\\weights\\groundingdino_swint_ogc.pth").to(device=device)

# Global Variables
csv_data = [] # list for storing the results
bb_size = 1.0
prompt = ''

# Function to detect objects in an image - related to GroundingDINO model
def detect(image_source, image, text_prompt, model, box_threshold=0.3, text_threshold=0.25):
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] #BGR to RGB
    return annotated_frame, boxes, phrases, logits

# bounding box format - [x_center, y_center, width, height] - for reference
# function to convert the bounding box to point pairs
def convert_to_point_pairs(bbox):
    x_center, y_center, width, height = bbox
    
    # increase the size of the bouding box to avoid minor gaze-related errors
    increased_half_width = (width / 2) * bb_size
    increased_half_height = (height / 2) * bb_size

    # calculating (x, y) pairs
    top_left = (x_center - increased_half_width, y_center - increased_half_height)
    top_right = (x_center + increased_half_width, y_center - increased_half_height)
    bottom_right = (x_center + increased_half_width, y_center + increased_half_height)
    bottom_left = (x_center - increased_half_width, y_center + increased_half_height)

    return top_left, top_right, bottom_right, bottom_left

# function to check if a point is inside a bounding box
def is_point_inside_bbox(point, bbox):
    x, y = point
    x_min , y_min = bbox[0]
    x_max, y_max = bbox[2]
    if x_min <= x <= x_max and y_min <= y <= y_max:
        return True
    return False

# function to find the entity that encapsulates the gaze point
def find_encapsulating_entity(df, point):
    overlap_ents_labels, overlap_ents_bbs = [], []
    closest_distance = float('inf')
    closest_entity = None
    for index, row in df.iterrows():
        bbox = row['bounding_box']
        bbox_point_pairs = convert_to_point_pairs(bbox)
        if is_point_inside_bbox(point, bbox_point_pairs):
            overlap_ents_labels.append(row['label'])
            overlap_ents_bbs.append(bbox)
            distance = abs(bbox[2] * bbox[3])
            if distance < closest_distance:
                closest_distance = distance
                closest_entity = (row['label'], bbox)

    return closest_entity, overlap_ents_labels, overlap_ents_bbs

# Function to process object detection for a single image (main function for an image)
def obj_det_single_img(image_source, image, gaze2d, timestamp, op_ts):
    if gaze2d is None:
        return
    annotated_frame, detected_boxes, d_labels, d_logits = detect(image_source, image, text_prompt=prompt, model=groundingdino_model)
    data = {
        'label': d_labels,
        'bounding_box': detected_boxes.tolist(),
        'pred': d_logits.tolist()
    }
    df = pd.DataFrame(data)
    df['label'] = df['label'].astype(str)
    df = df[~df['label'].str.contains('screen', case=False, regex=True)]
    entity, overlapping_ents_labels, overlapping_ents_bbs = find_encapsulating_entity(df, gaze2d)

    if entity is not None:
        label, bbox = entity
        csv_data.append({
            'timestamp': timestamp,
            'Output_Video_Timestamp': op_ts, 
            'gaze2d': gaze2d,
            'Object_Label': label,
            'Object_BB': bbox,
            'Overlapping_Objects': overlapping_ents_labels,
            'Overlapping_Objects_BBs': overlapping_ents_bbs,
            'Detected_Objects': df['label'].tolist(),
            'BBs_all': df['bounding_box'].tolist()
        })
    else:
        csv_data.append({
            'timestamp': timestamp,
            'Output_Video_Timestamp': op_ts,
            'gaze2d': gaze2d,
            'Object_Label': "",
            'Object_BB': np.NaN,
            'Overlapping_Objects': "",
            'Overlapping_Objects_BBs': np.NaN,
            'Detected_Objects': df['label'].tolist(),
            'BBs_all': df['bounding_box'].tolist()
        })

# Function to process the object detection for a video
def batch_process_obj_det(video_loc, matched_rows, bounding_box_size_increase, obj_input_prompt, sr):
    global bb_size, prompt, csv_data
    bb_size = bounding_box_size_increase
    prompt = obj_input_prompt
    csv_data.clear()
    cap = cv2.VideoCapture(video_loc)
    ti = 1 / sr
    op_ts = 0.0
    for i in stqdm(range(len(matched_rows)), desc='Processing'):
        row = matched_rows.iloc[i]
        timestamp = row['timestamp']
        if pd.notna(timestamp):
            frame_number = int(timestamp * cap.get(cv2.CAP_PROP_FPS))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                gaze2d = (row['gaze2d_x'], row['gaze2d_y'])
                cv2.imwrite('temp.png', frame)
                image_source, image = load_image('temp.png')
                obj_det_single_img(image_source, image, gaze2d, timestamp, op_ts)
                op_ts += ti
            sleep(0.5)

    cap.release()
    csv_df = pd.DataFrame(csv_data)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_df.to_csv(f"object_detection_results_{current_datetime}.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    return csv_df, f"object_detection_results_video_{current_datetime}.mp4"

# Function to process the object detection for a set of images
def batch_process_obj_det_v2(ss_folder, matched_rows, bounding_box_size_increase, obj_input_prompt, sr):
    global bb_size, prompt, csv_data
    bb_size = bounding_box_size_increase
    prompt = obj_input_prompt
    csv_data.clear()
    ti = 1 / sr
    op_ts = 0.0

    for i in stqdm(range(len(matched_rows)), desc='Processing'):
        row = matched_rows.iloc[i]
        fr_name = row['frame']
        fr_path = os.path.join(ss_folder, fr_name)
        if os.path.exists(fr_path):
            gaze2d = (row['gaze2d_x'], row['gaze2d_y'])
            # cv2.imwrite('temp.png', frame)
            image_source, image = load_image(fr_path)
            obj_det_single_img(image_source, image, gaze2d, fr_name, op_ts)
            op_ts += ti
    
        sleep(0.5)

    csv_df = pd.DataFrame(csv_data)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_df.to_csv(f"object_detection_results_{current_datetime}.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    return csv_df, f"object_detection_results_video_{current_datetime}.mp4"