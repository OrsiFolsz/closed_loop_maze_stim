# This script is used to set up OSC server and client for communication between Bonsai and the task. 
# The server receives data from Bonsai and the client sends data to Bonsai. 
# %%
import numpy as np
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher

# %%
class DataBlock:
    """Class to store and handle OSC data."""
    
    def __init__(self, n_rows, n_cols):
        """
        Initialize the DataBlock with a zero-filled numpy array.
        
        Args:
            n_rows (int): Number of rows for the data array.
            n_cols (int): Number of columns for the data array.
        """
        self.data = np.zeros((n_rows, n_cols))
       
        
    def data_handler(self, address, *args):
        """
        Handle incoming OSC data and store it in the data array.
        
        Args:
            address (str): OSC address pattern.
            *args: OSC message arguments.
        """
        self.x, self.y = args
        self.data[0, 0] = self.x
        self.data[1, 0] = self.y


def setup_osc_server(ip, receive_port, data_block):
    """
    Setup an OSC server to receive data and map it to the data handler.
    
    Args:
        ip (str): IP address of the OSC server.
        receive_port (int): Port number to receive OSC messages.
        data_block (DataBlock): Instance of DataBlock to handle incoming data.
    
    Returns:
        BlockingOSCUDPServer: Configured OSC server instance.
    """
    dispatcher = Dispatcher()
    dispatcher.map('/api', data_block.data_handler)
    server = BlockingOSCUDPServer((ip, receive_port), dispatcher)
    return server