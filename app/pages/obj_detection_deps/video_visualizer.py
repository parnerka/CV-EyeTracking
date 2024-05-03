import cv2
import os
import numpy as np
import pandas as pd
from time import sleep
from stqdm import stqdm

bbsize = 1.0

def get_norm_coor(coord):
    # if not coord:
    #     return []  # Return an empty list if coord is empty
    if isinstance(coord, list) and len(coord) == 4:
    # Standard frame size
    # frame_width = 1920
    # frame_height = 1080

    # Extract normalized coordinates
        x_center, y_center, width, height = coord

        # Convert normalized coordinates to standard coordinates
        x_center_std = x_center * 1920
        y_center_std = y_center * 1080
        inc_width_std = ((width * 1920) / 2) * bbsize  
        inc_height_std = ((height * 1080) / 2) * bbsize   

        # Calculate the 4 (x, y) coordinates of the bounding box
        x1 = int(x_center_std - inc_width_std)
        y1 = int(y_center_std - inc_height_std)
        x2 = int(x_center_std + inc_width_std)
        y2 = int(y_center_std + inc_height_std)

        # Resulting (x, y) coordinates
        bounding_box_coordinates = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return bounding_box_coordinates
    else:
        return []

def visualize(data, video_loc, bb_size, sr, file_name):
    global bbsize
    bbsize = bb_size
    data['Object Coordinates'] = data['Object Coordinates'].apply(get_norm_coor)

    # visualizer code to display the detected screen and object together along with the gaze point
    cap = cv2.VideoCapture(video_loc)

    # Create a VideoWriter object to save the output video as MP4
    output_video = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'H264'), sr, (1920, 1080))    # CODEC might throw a warning if the right library format is not installed

    # Iterate through each row in the DataFrame
    for i in stqdm(range(len(data)), desc='Generating Visualization'):
        row = data.iloc[i]
        timestamp = row['timestamp']

        # Seek to the specified timestamp in the video
        cap.set(cv2.CAP_PROP_POS_MSEC, int(timestamp * 1000))

        # Read the frame at the specified timestamp
        ret, frame = cap.read()

        if not ret:
            break
        
        gaze_column = row['gaze2d']
        if (len(gaze_column) == 2) and (pd.isna(gaze_column[0]) is not np.nan and pd.isna(gaze_column[1]) is not np.nan):
            gaze_m = (int((gaze_column[0] * 1920)), int((gaze_column[1] * 1080)))
        else:
            gaze_m = (None, None)

        object_coordinates = row['Object Coordinates']
        object_label = row['Object Label']

        # If gaze_m is empty, display 'Gaze not found'
        if pd.isna(gaze_m[0]) or pd.isna(gaze_m[1]):
            gaze_text = 'Gaze not found'
        else:
            gaze_text = 'Gaze Available'

            if pd.isna(object_label) or object_coordinates is None or not object_coordinates:
                object_text = 'Object not detected'
            else:
                object_text = f'Object: {object_label}'
                object_coordinates = np.array(object_coordinates, dtype=np.int32)
                object_hull = cv2.convexHull(object_coordinates, clockwise=True)
                cv2.drawContours(frame, [object_hull], -1, (0, 0, 255), 2)

            cv2.putText(frame, object_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.circle(frame, gaze_m, 15, (0, 255, 255), -1)

        sleep(0.5)
        cv2.putText(frame, gaze_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Save the frame to the output video
        output_video.write(frame)

        # Pause for a brief moment (adjust the delay as needed)
        cv2.waitKey(50)  # 50 milliseconds delay

    # Release the video objects and close the output video
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

def visualize_v2(data, bb_size, sr, ss_folder, file_name):
    global bbsize
    bbsize = bb_size
    data['Object Coordinates'] = data['Object Coordinates'].apply(get_norm_coor)
    # visualizer code to display the detected screen and object together along with the gaze point
    # cap = cv2.VideoCapture(video_loc)

    # Create a VideoWriter object to save the output video as MP4
    output_video = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'H264'), sr, (1920, 1080))    # CODEC might throw a warning if the right library format is not installed

    # Iterate through each row in the DataFrame
    for i in stqdm(range(len(data)), desc='Generating Visualization'):
        row = data.iloc[i]
        timestamp = row['timestamp']
        fr_path = os.path.join(ss_folder, timestamp)

        # Seek to the specified timestamp in the video
        #cap.set(cv2.CAP_PROP_POS_MSEC, int(timestamp * 1000))

        # Read the frame at the specified timestamp
        # ret, frame = cap.read()

        # if not ret:
        #     break
        frame = cv2.imread(fr_path)
        gaze_column = row['gaze2d']
        if (len(gaze_column) == 2) and (pd.isna(gaze_column[0]) is not np.nan and pd.isna(gaze_column[1]) is not np.nan):
            gaze_m = (int((gaze_column[0] * 1920)), int((gaze_column[1] * 1080)))
        else:
            gaze_m = (None, None)

        object_coordinates = row['Object Coordinates']
        object_label = row['Object Label']

        # If gaze_m is empty, display 'Gaze not found'
        if pd.isna(gaze_m[0]) or pd.isna(gaze_m[1]):
            gaze_text = 'Gaze not found'
        else:
            gaze_text = 'Gaze Available'

            if pd.isna(object_label) or object_coordinates is None or not object_coordinates:
                object_text = 'Object not detected'
            else:
                object_text = f'Object: {object_label}'
                object_coordinates = np.array(object_coordinates, dtype=np.int32)
                object_hull = cv2.convexHull(object_coordinates, clockwise=True)
                cv2.drawContours(frame, [object_hull], -1, (0, 0, 255), 2)

            cv2.putText(frame, object_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.circle(frame, gaze_m, 15, (0, 255, 255), -1)

        sleep(0.5)
        cv2.putText(frame, gaze_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Save the frame to the output video
        output_video.write(frame)

        # Pause for a brief moment (adjust the delay as needed)
        cv2.waitKey(50)  # 50 milliseconds delay

    # Release the video objects and close the output video
    #cap.release()
    output_video.release()
    cv2.destroyAllWindows()