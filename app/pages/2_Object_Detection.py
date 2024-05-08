# This file contains the code for the Object Detection page of the web app.

import streamlit as st
import pandas as pd
from pages.obj_detection_deps.detect_objects import batch_process_obj_det, batch_process_obj_det_v2
from pages.obj_detection_deps.video_visualizer import visualize, visualize_v2

st.set_page_config(page_title="Gaze-Based Object Detection")
st.header("Gaze-Based Object Detection")

st.write('Currently gaze data is only supported in CSV format with the following header names:')
st.write('Type 1 is for video format uses video timestamps to link gaze and frame:')
demo_df = pd.read_csv('./app_data_demo/app_data.csv')
st.dataframe(demo_df[:5])
st.write('Type 2 is for image format uses image name to link gaze and frame:')
demo_df_v2 = pd.read_csv('./app_data_demo/app_data_v2.csv')
st.dataframe(demo_df_v2[:5])
gaze_data_loc = st.text_input('Enter the Gaze Data\'s Path (eg. C:&#92;&#92;User&#92;&#92;GroundingDINO&#92;&#92;GazeData.csv). Currently supports only CSV format.', help='For Windows: "path&#92;&#92;to&#92;&#92;data".\nFor Mac/Linux: "path/to/data".')
gaze_data_type = st.selectbox('Gaze data type', ('Type 1', 'Type 2'))
if gaze_data_type == 'Type 1':
    video_loc = st.text_input('Enter the Video\'s Path (eg. C:&#92;&#92;User&#92;&#92;GroundingDINO&#92;&#92;VideoData.mp4). Currently supports only MP4 format.', help='For Windows: "path&#92;&#92;to&#92;&#92;video".\nFor Mac/Linux: "path/to/video".')
elif gaze_data_type == 'Type 2':
    ss_folder = st.text_input('Enter the Screenshot Folder\'s Path (eg. C:&#92;&#92;User&#92;&#92;GroundingDINO&#92;&#92;screenshots). Currently supports only PNG format.', help='For Windows: "path&#92;&#92;to&#92;&#92;folder".\nFor Mac/Linux: "path/to/folder".')
sr = st.number_input('Enter desired frame rate for the output video. Can be useful for presentation purposes of speeding up or slowing down the video.', value=0.0)
bounding_box_size_increase = st.number_input("Input bounding box size (scale is 1 to 2 times as large). This can be helpful if gaze accuracy is low to have a higher chance of assigning fixations to the object.", min_value=1.0, max_value=2.0, value=1.0, step=0.25)
# select_model = st.selectbox('Choose Your Preferred Zero-Shot Model', ('GroundingDINO','')) # For the future
obj_input_prompt = st.text_input('Prompt the objects to be detected. For optimal performance avoid prompting more than 10 Objects (eg. yellow star . blue rectangle . hand .) ', help='Prompt Format: "Obj1<space>.<space>Obj2<space>.<space>Obj3<space>."')

submit = st.button('Start Processing', type='primary')

# Process the data
if submit:
    tobii_data = pd.read_csv(gaze_data_loc)
    if gaze_data_type == 'Type 1':
        csv_df, file_name = batch_process_obj_det(video_loc, tobii_data, bounding_box_size_increase, obj_input_prompt, sr)
    elif gaze_data_type == 'Type 2':
        csv_df, file_name = batch_process_obj_det_v2(ss_folder, tobii_data, bounding_box_size_increase, obj_input_prompt, sr)
    st.write("Results Saved In Current Directory!")
    if gaze_data_type == 'Type 1':
        visualize(csv_df, video_loc, bounding_box_size_increase, sr, file_name)
    elif gaze_data_type == 'Type 2':
        visualize_v2(csv_df, bounding_box_size_increase, sr, ss_folder, file_name)
    st.write("Rendered Video")
    with open(file_name, 'rb') as video_file:
        video_bytes = video_file.read()
    st.video(video_bytes, format="video/mp4", start_time=0)
    st.write("Video Saved In Current Directory!") 
    csv_df = pd.DataFrame()
    file_name = ""