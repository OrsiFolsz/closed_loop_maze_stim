# Collections of functions that m25_utils uses from preexisting grid maze modules.

# Imports
#%% Imports
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean
from scipy.spatial import KDTree
import json
import networkx as nx
import os
import csv
import cv2
import pandas as pd
import moviepy.editor as mpy

# Initial globals

BASE_PATH = 'C:/pyControl/api_classes'

with open(os.path.join(BASE_PATH, 'm25_maze_config.json')) as input_file:
    MAZE_CONFIGS = json.load(input_file)

with open(os.path.join(BASE_PATH, 'maze_measurements.json')) as input_file:
    MAZE_MEASUREMENTS = json.load(input_file)

MAZE_NODE_DIMENSIONS = MAZE_MEASUREMENTS["maze_node_dimensions"]
LOWER_LEFT = MAZE_MEASUREMENTS["lower_left_node_cartesian_center"]  # meters
DISTNACE_BETWEEN_TOWERS = MAZE_MEASUREMENTS["distance_between_node_centers"]  # meters

RAW_VIDEO_PATH = BASE_PATH
MAZE_REGISTRATION_PATH = os.path.join(RAW_VIDEO_PATH, "tower_center_pixel_coords.tsv")
FIRST_SESSION_VIDEO_PATH = os.path.join(BASE_PATH, 'polygon_cropped_video_for_calibration_vid.mp4')



### 1. From representations
"""
This module contains classes to build simple represntation of maze.
"""

# Main functions
def get_simple_maze(maze_name):
    edges = MAZE_CONFIGS[maze_name]["structure"]
    return simple_maze(edges)

def get_maze_label2coord(simple_maze):
    """
    Returns a dictionary of node and edge labels (same as label attributes in simple maze object) to coordinates
    (network x node and edge positions).
    """
    node_coord2label = get_maze_coord2label(simple_maze)
    return {v: k for k, v in node_coord2label.items()}

def get_maze_coord2label(simple_maze):
    """
    Returns a dictionary of node and edge coordinates (network x node and edge positions) to
    standard alpha-neumeric labels
    """
    node_coord2label = nx.get_node_attributes(simple_maze, "label")
    edge_coord2label = nx.get_edge_attributes(simple_maze, "label")
    return {**node_coord2label, **edge_coord2label}


# %%
def simple_maze(edges):
    """
    Creates a networkx graph representation of a maze, where nodes are towers, and edges are walkways
    Input: edges (list of str): list of edges in the maze, in the format 'A1-A2'
    Output: maze (networkx graph): graph representation of the maze
    """
    maze = nx.Graph()
    maze.add_nodes_from(nx.grid_2d_graph(*MAZE_NODE_DIMENSIONS).nodes())
    edge_coords = get_edge_coords(edges)
    maze.add_weighted_edges_from([i + (DISTNACE_BETWEEN_TOWERS,) for i in edge_coords])
    nx.set_node_attributes(maze, _get_node_positions_dict(), "position")
    nx.set_node_attributes(maze, {v: k for k, v in get_simple_nodes_dict().items()}, "label")
    nx.set_edge_attributes(maze, {key: _get_center_edge_positions_dict()[key] for key in edge_coords}, "position")
    nx.set_edge_attributes(maze, {key: get_edge_coords2label_dict()[key] for key in edge_coords}, "label")
    return maze


# Secondary functions
# %% Simple maze functions

def _get_letter_codes(n):
    """Returns a list of n capital letters of the alphabet"""
    return [chr(i) for i in range(65, 65 + n)]


def get_simple_nodes_dict():
    """Returns a dictionary of alphanumeric nodes ('A1') to simple maze node coordinates (0,0)"""
    nodes_letter2coord = {}
    letter_codes = _get_letter_codes(MAZE_NODE_DIMENSIONS[1])
    for i, l in enumerate(letter_codes):
        for j in np.arange(0, MAZE_NODE_DIMENSIONS[0]):
            letter_node = l + str(j + 1)
            coord_node = (i, j)
            nodes_letter2coord[letter_node] = coord_node
    return nodes_letter2coord


def get_edge_coords(edges):
    """converts a list of alphanumeric edges ('A1-A2') to simple maze coordinates ((0,0),(0,1))"""
    nodes_dict = get_simple_nodes_dict()
    edges_letter_tuple = [edge.split("-") for edge in edges]
    edge_coords = []
    for i, edge in enumerate(edges_letter_tuple):
        edge_coords.append((nodes_dict[edge[0]], nodes_dict[edge[1]]))
    return edge_coords


def get_edge_coords2label_dict():
    """Returns a dictionary of edge coordinates ((0,0),(0,1)) to alphanumeric edge labels ('A1-A2')"""
    all_edges = nx.grid_2d_graph(*MAZE_NODE_DIMENSIONS).edges()
    node_coord2label = {v: k for k, v in get_simple_nodes_dict().items()}
    edge_coords2label = {}
    for edge in all_edges:
        edge_coords2label[edge] = node_coord2label[edge[0]] + "-" + node_coord2label[edge[1]]
    return edge_coords2label


def _get_node_positions_dict():
    """Returns a dictionary of node positions to cartesian coordinates, defined as center of maze tower"""
    nodes_dict = get_simple_nodes_dict()
    discrete_nodes2cartesian_nodes = {}
    node_discrete_coords = nodes_dict.values()
    for discrete_x, discrete_y in node_discrete_coords:
        cartesian_x = discrete_x * DISTNACE_BETWEEN_TOWERS + LOWER_LEFT[0]
        cartesian_y = discrete_y * DISTNACE_BETWEEN_TOWERS + LOWER_LEFT[1]
        discrete_nodes2cartesian_nodes[(discrete_x, discrete_y)] = (cartesian_x, cartesian_y)
    return discrete_nodes2cartesian_nodes


def _split_edges_into_horizontal_vertical(edges):
    """Splits a list of edge coors into horizontal and vertical edges (separate list outputs as tuple)"""
    horizontal_edges = []
    vertical_edges = []
    for edge in edges:
        if edge[0][0] == edge[1][0]:
            vertical_edges.append(edge)
        elif edge[0][1] == edge[1][1]:
            horizontal_edges.append(edge)
    return horizontal_edges, vertical_edges


def _get_center_edge_positions_dict():
    """Returns a dictionary of edge positions to cartesian coordinates"""
    center_edge_positions = {}
    all_edges = nx.grid_2d_graph(*MAZE_NODE_DIMENSIONS).edges()
    node_positions = _get_node_positions_dict()
    # sort edge coords into horizontal and vertical edges
    horizontal_edges, vertical_edges = _split_edges_into_horizontal_vertical(all_edges)
    # get center positions of horizontal edges
    for edge in horizontal_edges:
        node1, node2 = edge
        x1, y1 = node_positions[node1]
        x2, y2 = node_positions[node2]
        x = (x1 + x2) / 2
        y = y1
        center_edge_positions[edge] = (x, y)
    # get center positions of vertical edges
    for edge in vertical_edges:
        node1, node2 = edge
        x1, y1 = node_positions[node1]
        x2, y2 = node_positions[node2]
        x = x1
        y = (y1 + y2) / 2
        center_edge_positions[edge] = (x, y)
    return center_edge_positions




### 2. From maze_registration (pix2pos needs this)
"""This file registers video pixel coordinates of select towers on a maze for later use to correct for fish-eye distortion. A quality control step is included
to ensure that the camera does not move over the recording period."""

# Main functions
# %% Load calibration coordinates df
def get_calibration_coordinates_df():
    """Returns a dataframe with pixel and physical coordinates for each tower"""
    label2pixel_coords = load_tower_pixel_coords(MAZE_REGISTRATION_PATH)
    label2simple_node = get_simple_nodes_dict()
    simple_node2physical_coords = _get_node_positions_dict()
    calibration_node2pixel_coords = {
        label2simple_node[tower]: label2pixel_coords[tower] for tower in label2pixel_coords.keys()
    }
    calibration_node2physical_coords = {
        k: v for k, v in simple_node2physical_coords.items() if k in calibration_node2pixel_coords.keys()
    }
    pixel_physical_coords_df = pd.DataFrame(
        {
            "simple_node_label": label2pixel_coords.keys(),
            "calibration_node": calibration_node2pixel_coords.keys(),
            "physical_coords": calibration_node2physical_coords.values(),
            "pixel_coords": calibration_node2pixel_coords.values(),
        }
    )
    return pixel_physical_coords_df

def get_image_size_from_video():
    """Returns the image size of the videos in the raw_data directory, using 1st frame from the 1st video as an example image
    Output is (height, width)"""
    example_video_path = FIRST_SESSION_VIDEO_PATH
    temp_image_path = os.path.join(RAW_VIDEO_PATH, "temp_image.png")
    video = mpy.VideoFileClip(example_video_path)
    video.save_frame(temp_image_path, t=0)
    image = cv2.imread(temp_image_path)
    image_size = image.shape[:2]
    os.remove(temp_image_path)
    return image_size


# Secondary functions
# %% Save and load tower alignment coordinates
def load_tower_pixel_coords(tower_alignment_coords_path) -> dict:
    """Loads tower alignment coordinates from tower_center_pixel_coords.tsv (raw video directory) as dict of tower_label: (x,y)"""
    tower_label2pixel_coords = {}
    with open(tower_alignment_coords_path, "r") as infile:
        reader = csv.reader(infile, delimiter="\t")
        for row in [i for i in reader][1:]:
            tower_label2pixel_coords[row[0]] = (float(row[1]), float(row[2]))
    return tower_label2pixel_coords

# %% New Global variables from maze registration functions
CALIBRATION_COORDINATES_DF = get_calibration_coordinates_df()
IMAGE_SIZE = get_image_size_from_video()





### 3. From pix2pos
"""This module contains functions for correcting for radial distortion in the video images, 
and for converting raw pixel coordinates to physical coordinates."""

# Secondary functions
# %% Functions for non-linear distortion correction (raw_pixels -> distorted/corrected_pixels)
def distort_pixel_coords(pixel_coords:tuple, strength:int)->tuple:
    """Apply radial distortion to the set of (x,y) pixel coordinates, using the radial distortion function.
    strength>0 gives barrel distortion, strength<0 gives pincushion distortion, strength=0 gives no distortion."""
    centered_pixel_coords = center_pixel_coords(pixel_coords) #center coordinates
    x,y = vectorise_coordinates(centered_pixel_coords)
    y_size, x_size = IMAGE_SIZE
    x_distort, y_distort = radial_distortion(x,y,x_size,y_size, strength) #tuplise coordinates
    distorted_centered_coords = [(i,j) for i,j in zip(x_distort, y_distort)] 
    return uncenter_pixel_coords(distorted_centered_coords) #uncenter coordinates

def radial_distortion(x,y,x_size,y_size, strength):
    '''Apply radial distortion to the set of x,y points. strength>0
    gives barrel distortion, strength<0 gives pincushion distortion,
    strength=0 gives no distortion.'''
    r = strength*np.sqrt(x**2+y**2)/np.sqrt(x_size**2+y_size*2)
    r[r==0] = 1e-6 # Avoid divide by 0
    if strength>0: 
        theta = np.arctan(r)/r
    else:
        theta = r/np.arctan(r)
    xn = x*theta
    yn = y*theta
    return xn, yn

def vectorise_coordinates(coordinates:tuple):
    """Vectorise a set of (x,y) coordinates"""
    x = np.array([i[0] for i in coordinates])
    y = np.array([i[1] for i in coordinates])
    return x,y

def center_pixel_coords(pixel_coords):
    y_size, x_size = IMAGE_SIZE
    return [center_coordinate(x, y, x_size, y_size) for x, y in pixel_coords]

def center_coordinate(x,y,x_size,y_size):
    """Centers a coordinate to the center of an image (where the original origin was in the bottom left corner)"""
    x_center = x_size/2
    y_center = y_size/2
    x_new = x-x_center
    y_new = y-y_center
    return x_new,y_new

def uncenter_pixel_coords(centered_pixel_coords):
    y_size, x_size = IMAGE_SIZE
    return [uncenter_coordinate(x, y, x_size, y_size) for x, y in centered_pixel_coords]

def uncenter_coordinate(x,y,x_size,y_size):
    """Uncenters a set of (x,y) coordinates to new origin at bottom left corner"""
    x_center = x_size/2
    y_center = y_size/2
    x_new = x+x_center
    y_new = y+y_center
    return x_new,y_new

#%% Find optimal distortion strength for correcting non-linear video distortion
def get_optimal_distortion_strength(initial_guess=0.0):
    """Calculates the optimal distortion strength for correcting non-linear video distortion.
    i.e finds the distortion strength that, when applied, removes radial distortion from the video,
    and raw pixel coords."""
    physical_coords = CALIBRATION_COORDINATES_DF['physical_coords'].values
    pixel_coords = CALIBRATION_COORDINATES_DF['pixel_coords'].values
    physical_coords_distance_vector = coord_distance_vector(physical_coords)
    result = minimize(distortion_cost, 
                      initial_guess, 
                      args=(physical_coords_distance_vector, pixel_coords), 
                      method='BFGS', 
                      options={'eps': 1e-4}) #smaller step sizes don't find minima
    return result.x[0]

def distortion_cost(strength, physical_coords_distance_vector, pixel_coords):
    """Calculates the cost of a distortion strength, by comparing the relative distances between 
    all points in both distorted_pixel and physical space"""
    distorted_pixel_coords = distort_pixel_coords(pixel_coords, strength)
    distorted_pixel_coords_distance_vector = coord_distance_vector(distorted_pixel_coords)
    return np.sum((physical_coords_distance_vector-distorted_pixel_coords_distance_vector)**2)

def coord_distance_vector(coords, normalised=True):
    """Calculates the distance between all points in a set of (x,y) coordinates"""
    distances = []
    for i in range(len(coords)):
        for j in range(i+1,len(coords)):
            distances.append(euclidean(coords[i],coords[j]))
    if normalised:
        return np.array(distances)/np.std(distances)
    return np.array(distances)

#%% New Global variable for optimal distortion strength
OPTIMAL_VIDEO_DISTORTION_STRENGTH = get_optimal_distortion_strength()

#%% Functions for linear transformtation between corrected_pixel and physical coordinates
def get_linear_transormation_matrix():
    """Finds the Linear Transformation Matrix (T) that maps the distortion-corrected pixel coordinates to the physical coordinates.
    From calibration coordinates in CALIBRATION_COORDINATES_DF, """
    physical_coords = CALIBRATION_COORDINATES_DF['physical_coords'].to_list()
    corrected_pixel_coords = distort_pixel_coords(CALIBRATION_COORDINATES_DF['pixel_coords'].to_list(), 
                                                  strength=OPTIMAL_VIDEO_DISTORTION_STRENGTH)
    physical_system = np.asarray(physical_coords)
    corrected_pixel_system = np.asarray(corrected_pixel_coords)
    # homogenise coordinate matrices to allow for translation
    homog = lambda x:  np.hstack([x, np.ones((x.shape[0], 1))])
    # solve the least squares problem X * A = Y
    X = homogenise(corrected_pixel_system)
    Y = homogenise(physical_system)
    T, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return T

def homogenise(x):
    """Homogenise a matrix x by appending a column of ones"""
    return np.hstack([x, np.ones((x.shape[0], 1))])

#%% New Global variable for transformation matrix
VIDEO_LT_MATRIX = get_linear_transormation_matrix()

#%%
# Main functions
#%% Full raw pixel to physical coordinate translation functions

def translate_pixel2physical_coords(raw_pixel_coords):
    """Translates a set of raw pixel coordinates to physical coordinates.
    Input can be a tuple or list of tuples."""
    if isinstance(raw_pixel_coords,tuple): raw_pixel_coords=[raw_pixel_coords]
    corrected_pixel_coords = distort_pixel_coords(raw_pixel_coords, strength=OPTIMAL_VIDEO_DISTORTION_STRENGTH)
    physical_coords = translate_corrected_pixel2physical_coords(corrected_pixel_coords)
    if len(physical_coords)==1: return physical_coords[0]
    return physical_coords

def translate_corrected_pixel2physical_coords(corrected_pixel_coords):
    """Applies the linear transformation matrix (VIDEO_LT_MATRIX) to a set of distortion corrected pixel coordinates."""
    unhomog = lambda x: x[:, :-1]
    transform = lambda x: unhomog(np.dot(homogenise(x), VIDEO_LT_MATRIX))
    return transform(np.asarray(corrected_pixel_coords))





### 4. From mtraj
"""This module translates subject centroid tracking data into simple maze trajectories of nodes and edges."""

# Main functions
def get_nearest_simple_coord(centroid_positions, maze):
    """Get the nearest maze node for each position in the centroid_positions array."""
    position2maze_coord = get_position2simple_coord_dict(maze)
    coord_positions = list(position2maze_coord.keys())
    kd_tree = KDTree(coord_positions) #Create a KDTree from the maze coord positions
    _, nearest_node_indices = kd_tree.query(centroid_positions) # Query the KDTree to get the nearest maze node for each position
    nearest_nodes = [position2maze_coord[tuple(coord_positions[i])] for i in nearest_node_indices]
    return nearest_nodes


# Secondary functions
def get_position2simple_coord_dict(simple_maze):
    """Get a dictionary mapping maze node/edge positions to maze node/edge coordinates"""
    position2simple_node = {tuple(position):node for node, position in simple_maze.nodes(data='position')}
    position2simple_edge={tuple(position):(node1, node2) for node1,node2,position in simple_maze.edges(data='position')}
    position2simple_coord = position2simple_node.copy(); position2simple_coord.update(position2simple_edge)
    return position2simple_coord