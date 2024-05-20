<div style="text-align:center;">***Start of README***</div>
<div style="text-align:center;font-style: italic;">***LAST UPDATED ON: 04/30/2024***</div><br>



<b><u>BASIC SOFTWARE REQUIREMENT DETAILS:</u></b>

* Download `Python` (above version 3.9) 
* Download `Anaconda` (any version but latest version preferred) to manage virtual environments.
* Download `Visual Studio` and install the following libraries for ML/DL:
    - Python
    - Data storage and processing
    - data science and analytical applications
    - C++ libraries (desktop development for C++ Universal windows platform and linux and embeded development for C++).
* `GPU Setup` - You have a NVIDIA GPU and have the specified CUDA version (v12.2) and its toolkit installed properly then skip the next two sub-steps. 
    * Setup CUDA Version 12.2 (The pipeline was implemented using this version) - Instructions mentioned on <a href='https://developer.nvidia.com/cuda-12-2-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local'>this website</a>. Choose your corresponding options that match your system. Use default options.
    * You can check your CUDA version by opening a command prompt and typing `nvidia-smi`. Check your environmental path variables to ensure that CUDA is in your path. 
* Install `Git` (Optional but makes life easier) - <a href='https://git-scm.com/'> Download</a>. Choose visual studio code as default editor used by GIT. Use default options except not to use the credential manager. 
# Project 1: Gaze and Prompt-based Object Detection
<b><u>SETUP:</u></b>
 
1. Open `anaconda prompt` (via windows search box).

2. Install `PyTorch` (using pip) <a href='https://pytorch.org/'>from this website</a>. Select PyTorch Build (e.g. Stable; 'YOUR  OS' (eg. Windows); Python; CUDA 12.1 (use CPU option if you have mac)) and run the install command it outputs in an anaconda prompt. 
    - Set up a virtual environment using the following commands (<b>recommended practice</b>):
        ```
        conda create -n cv-final python=3.11.7

         conda activate cv-final
  
        [YOUR CODE FROM WEBSITE] --> eg. pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

3.  Install `GroundingDINO` (ONLY AFTER ABOVE PROGRAMS HAVE BEEN INSTALLED) - Follow installation steps mentioned on this <a href='https://github.com/IDEA-Research/GroundingDINO'>GitHub page</a> (scroll down to install section). Either download the repository via the zip option or use `git clone` (this is preferable otherwise you have to manually set it up) and then install groundingDINO using anaconda prompt. Choose a folder location where you will easily remember the path. 
    - To clone open git bash and enter the following commands:
        - Type: `git clone https://github.com/IDEA-Research/GroundingDINO.git`
    - Open anaconda prompt and enter the following commands:
        ```
        cd GroundingDINO/

        pip install -e .
        
        mkdir weights
    - If you are on Windows, you will have to manually download the weights to the `weights` folder. If not on Windows, then follow the repository's commands. 
        - Go <a href='https://github.com/IDEA-Research/GroundingDINO/releases'>Here</a>.
        - Expand assets within `v0.1.0-alpha`.
        - Download `groundingdino_swint_ogc.pth` to the weights folder. 
    - Once weights are downloaded and placed correctly type `cd ..`
    - Your path in the anaconda prompt will point to the `GroundingDINO` folder now. Verify this before moving forward.
            
4. Install additional libraries with the following command:
    ```
    pip install streamlit stqdm ipykernel pillow jupyter
5. Once the requirements and Grounding DINO are installed, close the terminal window.

6.  Download the EyetrackingCV repository and copy the `app` folder into the `GroundingDino` folder. In `app\pages\obj_detection_deps` and `app\pages`, locate files called `detect_objects.py` and `1_Preview.py` respectively. Open these files with a text editor.
    * Add the path to where Grounding DINO was installed on your computer on the second line: `sys.path.append("YOUR PATH TO GROUNDING DINO")`. 
        * NOTE - you must change directory slashes to be either a double back slash or a single forward slash.   
    * Find the line that uses paths to model weights and config. 
    * Add your specific paths to the model weights (`.\\GroundingDINO\\weights\\groundingdino_swint_ogc.pth`) and the config file (`.\\GroundingDINO\\groundingdino\\config\\GroundingDINO_SwinT_OGC.py`).

### <u>Steps to run the Gaze and Prompt-based Object Detection GUI:-</u>

1.  Locate and open the `app` folder.

2.  Open an anaconda prompt and start the virtual environment using the command `conda activate cv-final`. The leftward (base) should change to (cv-final).

3.  Change the folder directory to the path of the app folder using the command `cd path/to/app/folder`. 

4.  Once the path is updated, type in the command `streamlit run Home.py` to start the GUI. A browser window will appear that shows the GUI. 

5.  Once the GUI is loaded, fill in the entries mentioned appropriately to output a CSV file containing Object Detection Results and the Corresponding Video with bounding boxes (if detected). 

<br>

# Project 2: Screen Tracking using April Tags

<b>NOTE:</b> Install Pupil-Apriltags package before running the screen tracking code - Open an Anaconda prompt/command prompt and paste this: `pip install pupil-apriltags`. Once the package is installed, follow these steps to run the screen tracking code. 

### <u>Steps to run the screen tracking algorithm:-</u>

1.  Open the jupyter notebook titled `screen_tracking.ipynb`. 
    
    --> (PATH: `.\\GroundingDINO\\app\\Screen Tracking\\screen_tracking.ipynb`)

2.  Assign the path of the video to the `video_loc` variable and the eye tracking data path to the `tobii_data` variable. Both these variables are located in the third cell of the notebook. Make sure to check syntax for paths (need \\ and '). Open up a new kernel environment if need be, choose base and run. 

3.  That's it! Now run all the cells in the python notebook to get the outputs. Both the results CSV file (`screen_detection_results/date_unique_id.csv`) and the visualizer video (`screen_detection_video_outputs/date_unique_id.mp4`) will be saved on disk in the current directory (location of the notebook).

## Important points to remember for both the projects

* Valid Path Formats:
    * path/to/file
    * path&#92;&#92;to&#92;&#92;file

* Please save the results (CSV and Video) of a specific run in a different directory after running the code otherwise it'll be overwritten in the next run.

* Input format of the eye tracking data should exactly resemble this including header names:

    | timestamp (in seconds) | gaze2d_x (normalized) | gaze2d_y (normalized) |
    |-----------------|-----------------|-----------------|
    | <center>t<sub>1</sub></center>    | <center>x<sub>1</sub></center>    | <center>y<sub>1</sub></center>    |
    | <center>t<sub>2</sub></center>    | <center>x<sub>2</sub></center>    | <center>y<sub>2</sub></center>    |
    | <center>t<sub>3</sub></center>    | <center>x<sub>3</sub></center>    | <center>y<sub>3</sub></center>    |
    | <center>.</center>    | <center>.</center>    | <center>.</center>    |
    | <center>.</center>    | <center>.</center>    | <center>.</center>   |
    | <center>t<sub>n</sub></center>    | <center>x<sub>n</sub></center>    | <center>y<sub>n</sub></center>    |

    Eg: | 348.48    | 0.360786    | 0.422736    |

* Standard video frame resolution of (1920 x 1080) is assumed for both the pipelines.

<br>

# Details about the results

## 1. Object Detection Results
- The GUI will output two files: 
    1) CSV file - which logs object detection results for each frame in the gaze data
    2) Output Video - which combines all the extracted frames/screenshots and visualizes the object that is being looked at, along with the gaze point for reference. 

- Headers in the output CSV and what they signify:
    1. timestamp: Original video timestamp. Units: seconds
    2. Output_Video_Timestamp: Output video timestamp. Units: seconds
    3. gaze2d: Gaze coordinates in a frame. Units: Normalized coordinates (Value between 0 and 1. Normalized using the width and height of the frame i.e. 1920x1080 in our case)
    4. Object_Label: The label of the object that's being looked at (Mentioned in the prompt).
    5. Object_BB: The bounding box of the object that's being looked at. Format: Normalized YOLO format - [x_center, y_center, width, height]
    6. Overlapping_Objects: List of overlapping object labels that encompass the gaze point. Can be useful when objects are cluttered around the gaze point.
    7. Overlapping_Objects_BBs: List of bounding boxes of overlapping object labels that encompass the gaze point. Format: Normalized YOLO format - [x_center, y_center, width, height]
    8. Detected_Objects: All the objects that are detected in a frame based on the prompt that was provided.
    9. BBs_all: Bounding boxes of all the objects that are detected in a frame based on the prompt that was provided. Format: Normalized YOLO format - [x_center, y_center, width, height]

## 2. Screen Tracking Results
- The screen tracking jupyter notebook will output two files: 
    1) CSV file - which logs screen tracking results for each frame in the gaze data
    2) Output Video - which combines all the extracted frames/screenshots and visualizes the screen that is being looked at, along with the gaze point for reference. 

- Headers in the output CSV and what they signify:
    1. timestamp: Original video timestamp. Units: seconds
    2. Output_Video_Timestamp: Output video timestamp. Units: seconds
    3. gaze2d_x: X-coordinate of the gaze point in a frame. Units: Standard (not normalized)
    4. gaze2d_y: Y-coordinate of the gaze point in a frame. Units: Standard (not normalized)
    4. Screen: The number of the screen that's being looked at.
    5. BL: (x, y) coordinates of the top-right point of the bottom left April tag. Units: Standard (not normalized)
    6. BR: (x, y) coordinates of the top-left point of the bottom right April tag. Units: Standard (not normalized)
    7. TR: (x, y) coordinates of the bottom-left point of the top right April tag. Units: Standard (not normalized)
    8. TL: (x, y) coordinates of the bottom-right point of the top left April tag. Units: Standard (not normalized)

## How can we convert standard image co-ordinates to normalized co-ordinates:

- Let's understand this with an example: Consider a standard image point (x, y). Let's also assume that the resolution of the image is 1920x1080. 
    * Normalized x = (Standard x) / 1920
    * Normalized y = (Standard y) / 1080

- Eg. (1158.817056, 406.1419488) ---> (0.60355055, 0.37605736) ---> Can be verified in the demo app data.

<b>FOOTER:</b>

* <b>FOR ANY QUERIES RELATED TO THE REPOSITORY, PLEASE REACH OUT TO DR. RUSSELL COHEN HOFFING.</b>

* <b>CONTRIBUTORS: PRANAV M PARNERKAR, DR. RUSSELL COHEN HOFFING.</b>

<div style="text-align:center;">***End of README***</div> 
