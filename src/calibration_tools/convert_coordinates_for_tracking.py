# %% imports
import os
import csv
import cv2
import numpy as np
import json
import pickle as pkl
import pandas as pd
import moviepy.editor as mpy
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from representations import _get_node_positions_dict, get_simple_nodes_dict
import maze_registration as maze_reg
import pixels_to_position as pix2pos 




# %% Global variables
ALIGNMENT_NODES = ["A1", "A4", "A7", "C3", "C5", "D1", "D4", "D7", "E3", "E5", "G1", "G4", "G7"]
RAW_VIDEO_PATH ="C:/Users/Behre/Documents/LED_API"
# SESSIONS_DATA_DIRECTORY_DF = get_data_directory_df()
MAZE_REGISTRATION_PATH = os.path.join(RAW_VIDEO_PATH, "tower_center_pixel_coords.tsv")
FIRST_SESSION_VIDEO_PATH = "C:/Users/Behre/Documents/LED_API/polygon_cropped_video_for_calibration_vid.mp4"
%matplotlib qt 


# %% Get experiment alignment coordinates
##uncomment below to get calibration running
# # get preliminary tower pixel coords
# # use first session to define preliminary tower pixel coords
tower2pixel_coord_replicates = []
n_replicates = 3
for i in range(n_replicates):  # get ready to click!
    tower2pixel_coord_replicates.append(
         maze_reg.get_tower_center_coordinates_from_video(FIRST_SESSION_VIDEO_PATH, ALIGNMENT_NODES)
     )
preliminary_tower_pixel_coordinates = maze_reg.get_average_tower_coords(tower2pixel_coord_replicates)

# # quality control
click_variance = maze_reg.get_tower_click_variance(tower2pixel_coord_replicates)
click_variance_threshold = 5 * click_variance

maze_reg.save_tower_alignment_coords(RAW_VIDEO_PATH, preliminary_tower_pixel_coordinates)
print("tower alignment coordinates saved")

# %%matplotlib inline
#using calibratinon coordinates fird optimal video distortion and lt_matrix
CALIBRATION_COORDINATES_DF = maze_reg.get_calibration_coordinates_df()
IMAGE_SIZE = maze_reg.get_image_size_from_video()   

OPTIMAL_VIDEO_DISTORTION_STRENGTH = pix2pos.get_optimal_distortion_strength()
VIDEO_LT_MATRIX = pix2pos.get_linear_transormation_matrix()

CORRECTION_PARAMS = {
    'distortion_strenght' : OPTIMAL_VIDEO_DISTORTION_STRENGTH,
    'lt_matrix' : [list(i) for i in VIDEO_LT_MATRIX]
}

with open("video_correction_params.json", "w") as outfile:
    outfile.write(json.dumps(CORRECTION_PARAMS, indent=4))
# %%
