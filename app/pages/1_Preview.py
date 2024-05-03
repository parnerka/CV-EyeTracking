import sys
sys.path.append("C:\\Users\\ARL")   # Path where the GroundingDINO library was cloned - The library won't be detected if this is not set properly
import torch
import streamlit as st
from PIL import Image
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict, load_model
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path to model weights
groundingdino_model = load_model("C:\\Users\\ARL\\GroundingDINO\\groundingdino\\config\\GroundingDINO_SwinT_OGC.py", "C:\\Users\\ARL\\GroundingDINO\\weights\\groundingdino_swint_ogc.pth").to(device=device)

st.set_page_config(page_title="Gaze-Based Object Detection")
st.header("Grounding DINO - Preview")

st.write('To make sure your prompt does not fail during the video analysis, we recommend you to upload mulitple frames (one-by-one) with different objects here and check if your prompts are working.')

img_loc = st.text_input('Enter the Frame\'s Path (eg. C:&#92;&#92;User&#92;&#92;GroundingDINO&#92;&#92;test.png)', help='For Windows: "path&#92;&#92;to&#92;&#92;frame".\nFor Mac/Linux: "path/to/frame".')

show = st.button("Show Frame")

if show:
    image = Image.open(img_loc)
    st.image(image)

test_prompt = st.text_input('Prompt the Objects to be Detected (For optimal performance avoid prompting more than 10 Objects) (eg. yellow star . blue rectangle . hand .)', help='Prompt Format: "Obj1<space>.<space>Obj2<space>.<space>Obj3<space>."')

test_button = st.button('Detect Objects', type='primary')

if test_button:
    image_source, image = load_image(img_loc)

    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=image,
        caption=test_prompt,
        box_threshold=0.3,
        text_threshold=0.25
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1]
    st.image(annotated_frame)