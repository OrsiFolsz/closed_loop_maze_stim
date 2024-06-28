# Closed loop maze stim

## Description

Contains scripts for closed-loop optogenetic stimulation on the grid maze.

The maze task runs on Pycontrol, which controls LED opto stimulation. LED commands are sent to Pycontrol from an API, which communicates with both Pycontrol and Bonsai, in order to achieve stimulation based on online animal position. Animal coordinates are sent from Bonsai to the API via OSC messages, and are coomuncated to Pycontrol every ~10ms. This way, opto stimulation can be turned on and off based on distance from goal in maze coordinates.
This can be used as basis to implement more complex online processing, e.g. API can call scritps for route intfernces. 

Main components:
- Pycontrol task file for running m25 task on 7x7 maze.
- Custom bonsai workflow for centroid tracking, video recording, and sending OSC messages of mouse x-y position.
- Camera aquisition files (based on Thomas Akam's scripts).
- Api class that communicates between Pycontrol task and Bonsai (via OSC messages).
- Helper modules for OSC communication, API helper functions, and maze tools.
- Data files that specify the m25 maze and camera calibration based on example video.
- Modules used for calibration (based on Peter's preprecessing pipeline).

Notes for use:
- Move files in api_classes into api_classes folder in your Pycontrol folder.
- Move files in pycontrol_tasks into tasks folder in your Pycontrol folder.
- This code also expects files in helpers and data to be moved into api_classes folder (would be good to organise this better later).
- Update camera acqusition file based on experiment.
- If you change maze config, update maze_config file and task file.
- Before first run or changes to the camera, need to run calibration based on example video. This calibates pixel positions to maze positions to handle camera distortions. See calibration tools (contains necessary modules from preprocesing pipeline). Calibration gives you tower_center_pixel_coords.tsv that specifies in pixel coordinates the centre of maze nodes. This is then used to get maze position by getting node/edge closest to x-y position of animal. To do calibration, run convert_coordiantes_for_tracking.py.
- In Bonsai, if you drag largest binary region node then centroid on greyscale visualiser, you can follow tracking and on the recording. Extra click on the mouse makes trajectory appear too.
- Make sure to run calibration in the crooped version of the bonsai video.
- For calibraion to work, I had to reverse y coordinates in calibraion function in maze_registration. So now 0,0 coordinate is top left corner in both Bonsai and tower_center_pixel_coord.tsv.
- For now need to close and reopen Pycontrol befoe starting new experiment, because doesn't unsubscribe from OSC messages when ening task. Would be nice to solve this. 

This has files for basic m25 maze structure and m25_corners structure. 

## Installation

Requires Pycontrol and Bonsai.
Pycontrol uses packages form base python environment so API dependencies need to be there (?).

```bash
pip install -r requirements.txt