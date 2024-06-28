# Helper methods to control the LED via API based on tracking and maze structure
import json
import networkx as nx
import numpy as np
from api_classes import all_maze_functions as maze_funcs

 
# %% Global Variables
with open('C:/pyControl/api_classes/m25_maze_config.json') as input_file:
    MAZE_CONFIGS = json.load(input_file)

OUTER_CIRCLE_NODES = MAZE_CONFIGS["m25"]["goal_sets"]["outer_circle"]
TARGET_NODES = MAZE_CONFIGS["m25"]["goal_sets"]["m25"]
MAZE = maze_funcs.get_simple_maze("m25")
COORD2LABEL = maze_funcs.get_maze_coord2label(MAZE)
LABEL2COORD = maze_funcs.get_maze_label2coord(MAZE)

# %% Functions for stimulation control
def get_maze_locations_x_steps_away(upcoming_goal_label, x):
    """
    Returns a list of nodes that are x edges away from the specified start node.

    Args:
        upcoming_goal_label (str): The label of the start node, which will be the goal in the task.
        x (int): The number of steps/edges away from the start node.

    Returns:
        list: List of node labels that are x steps away from the start node.
    """
    start_node_coord = LABEL2COORD[upcoming_goal_label]
    path_lengths = nx.single_source_shortest_path_length(MAZE, start_node_coord)
    nodes_x_steps_away = [node for node, length in path_lengths.items() if length == x]
    labels_x_steps_away = [COORD2LABEL[node] for node in nodes_x_steps_away]
    return labels_x_steps_away

def get_outer_circle_locations_x_steps_away(upcoming_goal_label, x):
    """
    Returns a list of nodes that are x edges away from the specified start node AND on the outer m25 circle.

    Args:
        upcoming_goal_label (str): The label of the start node.
        x (int): The number of steps/edges away from the start node.

    Returns:
        list: List of node labels that are x steps away from the start node AND on the outer circle.
    """
    labels_x_steps_away = get_maze_locations_x_steps_away(upcoming_goal_label, x)
    outer_circle_labels_x_steps_away = [label for label in labels_x_steps_away if label in OUTER_CIRCLE_NODES]
    return outer_circle_labels_x_steps_away
# %% Functions for XY to maze coordinates (based on: tracking_pxls_to_maze.py)
def get_maze_coords_online(x,y):
    """
    Translates the latest XY coordinates to maze coordinates.

    Args:
        x (float): The x-coordinate.
        y (float): The y-coordinate.

    Returns:
        str: The label of the maze coordinate.
    """

    if np.isnan(x) or np.isnan(y):
        return "NaN"

    pixel_coords = np.array([x, y])
    pixel_coords_tuple = (pixel_coords[0], pixel_coords[1])  # Wrap pixel_coords in a tuple if it is a single set of coordinates

    estimated_physical_coords = maze_funcs.translate_pixel2physical_coords(pixel_coords_tuple)
    estimated_physical_coords_tuples = [(float(estimated_physical_coords[0]), float(estimated_physical_coords[1]))]

    maze_coord = maze_funcs.get_nearest_simple_coord(estimated_physical_coords_tuples, MAZE)

    maze_label = [COORD2LABEL[node] for node in maze_coord]
    current_maze_label = str(maze_label[0])
    return current_maze_label
# %%
