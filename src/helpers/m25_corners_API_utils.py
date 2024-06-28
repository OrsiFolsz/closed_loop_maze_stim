
# %% This will have the helper methods we need to contol the LED via API based on tracking and maze structure
import json
import networkx as nx
import numpy as np
import pandas as pd
import os
from api_classes import all_maze_functions_corners as maze_funcs

 
# %% Global Variables
with open('C:/pyControl/api_classes/m25_corners_maze_config.json') as input_file:
    MAZE_CONFIGS = json.load(input_file)

OUTER_CIRCLE_NODES = MAZE_CONFIGS["m25"]["goal_sets"]["outer_circle"]
TARGET_NODES = MAZE_CONFIGS["m25"]["goal_sets"]["m25"]
MAZE = maze_funcs.get_simple_maze("m25")
COORD2LABEL = maze_funcs.get_maze_coord2label(MAZE)
LABEL2COORD = maze_funcs.get_maze_label2coord(MAZE)

# %% Functions for stim
def get_maze_locations_x_steps_away(upcoming_goal_label, x):
    """
    Returns a list of nodes that are x edges away from the specified start node.
    
    Parameters:
    maze (networkx.Graph): The maze graph.
    upcoming_goal_label (str): The label of the start node, which will be the goal in the task
    x (int): The number of steps/edges away from the start node.

    Returns:
    List of node labels that are x steps away from the start node.
    """

    # Get the coordinate of the start node
    start_node_coord = LABEL2COORD[upcoming_goal_label]
    
    # Compute shortest path lengths from the start node
    path_lengths = nx.single_source_shortest_path_length(MAZE, start_node_coord)
    
    # Find nodes that are exactly x steps away
    nodes_x_steps_away = [node for node, length in path_lengths.items() if length == x]
    
    # Convert coordinates back to labels
    labels_x_steps_away = [COORD2LABEL[node] for node in nodes_x_steps_away]
    
    return labels_x_steps_away

def get_outer_circle_locations_x_steps_away(upcoming_goal_label, x):
    """
    Returns a list of nodes that are x edges away from the specified start node AND on outer m25 circle.
    
    Parameters:
    maze (networkx.Graph): The maze graph.
    start_node_label (str): The label of the start node.
    x (int): The number of steps/edges away from the start node.

    Returns:
    List of node labels that are x steps away from the start node AND on outer circle.
    """

    labels_x_steps_away = get_maze_locations_x_steps_away(upcoming_goal_label, x)
    outer_circle_labels_x_steps_away = [label for label in labels_x_steps_away if label in OUTER_CIRCLE_NODES]
    return outer_circle_labels_x_steps_away


# %% Functions for xy to maze coordinates (based on: tracking_pxls_to_maze.py)
def get_maze_coords_online(x,y):
    
    # Assume x and y are already the latest coordinates

    # Check if x or y are NaN
    if np.isnan(x) or np.isnan(y):
        return "NaN"

    pixel_coords = np.array([x, y])
    pixel_coords_tuple = (pixel_coords[0], pixel_coords[1])  # Wrap pixel_coords in a tuple if it is a single set of coordinates, this is what functions needs

    estimated_physical_coords = maze_funcs.translate_pixel2physical_coords(pixel_coords_tuple)
    estimated_physical_coords_tuples = [(float(estimated_physical_coords[0]), float(estimated_physical_coords[1]))]

    maze_coord = maze_funcs.get_nearest_simple_coord(estimated_physical_coords_tuples, MAZE)

    maze_label = [COORD2LABEL[node] for node in maze_coord]
    current_maze_label = str(maze_label[0])
    return current_maze_label # Return the label of the maze coordinate at string
# %%
