# User config setting for Point Grey Bonsai multi camera acquisition.

# --------------------------------------------------------------------
# Experiment config.
# --------------------------------------------------------------------

subjects = {
            # 3:'KM008_L'
            # 3: 'KM009_R'
            # 3: 'KM010_NM'
            # 3: 'KM011_R'    
            3: 'test'
            }  # {box: subject_ID} '

# subjects = {1:'KM006_R',
#             2: 'KM007_LR'}  # {box: subject_ID} '

# data_dir = 'C:\\DATA\\video_test'  # Directory where data will be saved.
data_dir = 'C:\\DATA\\vgat_opto'  # Directory where data will be saved.

# --------------------------------------------------------------------
# Hardware config.
# --------------------------------------------------------------------

camera_IDs = {3: 0,      # {box: camera_index}
              # 2: 1
              # 3: 2,
              # 4: 3
              }

camera_res = (1280,1024) # Should be set to match camera setting: (width, height) in pixels.

framerate = 60           # Should be set to match camera settings.

# --------------------------------------------------------------------
# Video format config
# --------------------------------------------------------------------

downsample = False # Set to a number to spatialy downsample output to lower resolution, 
                   # e.g. downsample=2 will reduce a 1280x1024 input to a 640x512 output.

# --------------------------------------------------------------------
# Bonsai path config.
# --------------------------------------------------------------------

bonsai_path = 'C:\\Users\\Behre\\AppData\\Local\\Bonsai\\Bonsai.exe'

workflow_path = 'C:\\Camera_acquisition\\multi_recorder_cuda_CLI_bigmaze_tracking_OSC.bonsai'