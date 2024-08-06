# Data Collection Readme

This `README` contains the data collection protocol for the custom unreal environment.

1. Need to start the docker system, by opening up the TackBak folder: `C:\Users\ARL\Desktop\TackBak\scripts\windows`

    - run `start.ps1` in powershell and wait for it to finish (powershell window will dissappear once complete)
    - Check status by running `status.ps1` to make sure that its running
        - open power shell via explorer by typing in powershell into the folder url
        - type `status` and press tab to run the `status.ps1` script
        - all should be running except for create-topics and create-connectors which will start when you run the simulation


2. Setup TobiiG3: `C:\Users\ARL\Desktop\TobiiGlassesData\TG3Controller`
- first calibrate the eye tracker
    - go to `C:\Users\ARL\Desktop\TobiiGlassesData\TG3Controller\TG3Calibrate` and open up a command window by typing cmd
    - first ping to make sure the tg3 is recognized on the network
        - type `ping [serial number of TG3 ours is found in the tobiiSN.txt file in TobiiGlassesData]`
        - if there is a reply that means its on the network
    - run calibration by typing `TG3Calibrate.exe TG03B-080201220441` [or whatever the SN is]
    - look at calibration target and check to make sure it says calibration successful

3. Run kafka connecter
    - open cmd window in `C:\Users\ARL\Desktop\TobiiGlassesData\TG3Controller\TG3Kafka`
    - and run `TG3Kafka.exe TG03B-080201220441 tobiidatacollection` [or whatever string input]

4. Run simulation:
    <br>
    - GO TO: `C:\Users\ARL\Desktop\TobiiGlassesData\Tobiipro_glasses\RussellTestProject\WindowsNoEditor`
    - run: `RussellEyetracking.exe`
    - type in `Tack.Start` when you want to begin recording.

5. Once the data collection session starts, look at following objects in the environment. Note that you need to press the letter key `L` when you start to fixate on an object and press `L` again after you are done looking at the object for 5 seconds. This denotes the time interval you looked at that specific object (helps in setting up the ground truth). Follow the same process for all the objects mentioned below:
    - Table
    - Chair
    - Blue Oil Barrel
    - Orange Traffic Cone
    - Porta Potty
    - Big Brown Box (behind the porta potty)
    - Yellow Vehicle (adjacent to the porta potty)
    - The solid orange barricade (next to the light pole)
    - The bigger barrier (next to the orange barricade)
<br>

    ---> NOTE: You can view these objects from any angle you want but make sure to follow the fixation process mentioned above to avoid errors during validation. 

6. When finished with the data collection:
    - type `Tack.end`
    - go to kafka cmd window and do ctrl + c

7. Export data:
    - GO TO: `C:\Users\ARL\Desktop\TobiiGlassesData\tack-postgres-exporter-main\app` and run `tobiiglassesexporter.bat`

<br>
<em>Last updated on: 8th May, 2024</em>
