#  API user class to trigger LED stimulation based on mouse position

from source.gui.api import Api
from api_classes import m25_API_utils as m25
from api_classes import bonsai_osc

# This class should be have the same name as the file and inherit the API class
# look at gui/api.py to see what functions can be redefined and called
class m25_API(Api):
    def __init__(self):
        """Initialize the user API class when the task is uploaded to the board."""
        # Initialize instance attributes available to all methods
        self.current_goal_label = None 
        
        self.stim_trial = False   # Boolean indicating if we are stimulating on this trial
        
        self.stim_start_nodes = None
        self.stim_stop_nodes = None
        
        self.stim_start_distance = None
        self.stim_stop_distance = None

        self.stim_on_happened = False
        self.stim_off_happened = False

        self.update_counter = 0
        
        # Initialize OSC server
        ip = '127.0.0.1'
        receive_port = 2323
        data_shape = (2, 1)
        self.data_block = bonsai_osc.DataBlock(*data_shape)
        self.server = bonsai_osc.setup_osc_server(ip, receive_port, self.data_block)

   
    def process_data_user(self, data):
        """Process data sent from the task, evry time thereis now data.

        Args:
            data (dict): Dictionary containing task data.
        """
        # Get the stim distances from task via prints at start
        if self.stim_start_distance is None or self.stim_stop_distance is None:
            stim_start_msgs = [printed[0] for printed in data["prints"] if "Stim_start_distance_is_" in printed[0]]
            for stim_start_msg in stim_start_msgs:
                self.stim_start_distance = int(stim_start_msg.split("_")[-1])
                #self.print_to_log(f"Stim start dist is: {self.stim_start_distance}")
                
            stim_stop_msgs = [printed[0] for printed in data["prints"] if "Stim_stop_distance_is_" in printed[0]]
            for stim_stop_msg in stim_stop_msgs:
                self.stim_stop_distance = int(stim_stop_msg.split("_")[-1])
                #self.print_to_log(f"Stim stop dist is: {self.stim_stop_distance}")

        # On every trial, get stim decision and goal
        self.new_prestim_timer = [event.name == "pre_stim_timer" for event in data["events"]]

        if self.new_prestim_timer:
            stim_msgs = [printed[0] for printed in data["prints"] if "Stim_decision_is_" in printed[0]]
            for stim_msg in stim_msgs:
                stim_decision_str = stim_msg.split("_")[-1]
                if stim_decision_str == "True":
                     self.stim_trial = True
                else:
                    if self.stim_trial:  # Reflects previous trial stim decision still.
                        # Ensures new trial will be stim if previous trial was stim but no stim happened. 
                        # To help maintain stim prob specified in task, even if mice are already closer to goal than stim start distance.
                        if not self.stim_on_happened and not self.stim_off_happened:
                            self.stim_trial = True 
                        else: 
                            self.stim_trial = False
                    else: 
                        self.stim_trial = False
            

            goal_msgs = [printed[0] for printed in data["prints"] if "Current_goal_is_" in printed[0]]
            for goal_msg in goal_msgs:
                self.current_goal_label = str(goal_msg.split("_")[-1])

                # Optional prints about pervious trial
                self.print_to_log(f"Stim on happened: {self.stim_on_happened}")
                self.print_to_log(f"Stim off happened: {self.stim_off_happened}")       

                # Determine where to stim based on current goal label
                if self.stim_trial:
                    
                    # Reset stim on/off happened flags
                    self.stim_on_happened = False
                    self.stim_off_happened = False

                    if self.current_goal_label is not None:
                        self.stim_start_nodes = m25.get_outer_circle_locations_x_steps_away(self.current_goal_label, self.stim_start_distance)
                        self.stim_stop_nodes = m25.get_outer_circle_locations_x_steps_away(self.current_goal_label, self.stim_stop_distance)
                        
                        self.print_to_log(f"New stim decision: {self.stim_trial}")
                        self.print_to_log(f"Stim start nodes:{self.stim_start_nodes}")
                        self.print_to_log(f"Stim stop nodes:{self.stim_stop_nodes}")
                else:
                    # Reset stim on/off happened flags
                    self.stim_on_happened = False
                    self.stim_off_happened = False

        # Reset stim decision to false after poked in for reward
        new_poked_in_state = [state.name == "poked_in" for state in data["states"]]
        if new_poked_in_state:
            self.stim_trial = False
   
              
    def plot_update(self):
        """Called whenever the plots are updated.

        The default plotting update interval is 10ms
        and can be adjusted in the settings dialog.
        """
        # Get centroid xy from OSC server
        self.server.handle_request()
        coordinates = self.data_block.data

        # Check if coordinates contain NaN values
        centroid_position_label = m25.get_maze_coords_online(coordinates[0][0], coordinates[1][0])

        # Increment the update counter - this is to have fewer prints to log
        self.update_counter += 1
        # Print to log every 10th message - approx every second
        if self.update_counter % 30 == 0:
            self.print_to_log(centroid_position_label)

        if self.stim_trial:
            if not self.stim_on_happened:
                if centroid_position_label in self.stim_start_nodes: 
                    self.trigger_event("stim_on_trigger")  # Trigger stimulation event in pycontrol
                    self.print_to_log(f"Starting stim on {centroid_position_label}")
                    self.stim_on_happened = True

            else: # self.stim_on_happened is True
                if not self.stim_off_happened:  #Only check to turn off if it hasn't been switched off yet
                    if centroid_position_label in self.stim_stop_nodes:
                        self.trigger_event("stim_off_trigger") # so stim timer is up=off, so either this or timer stops stim
                        self.print_to_log(f"Ending stim on {centroid_position_label}")
                        self.stim_off_happened = True
            
    def run_stop(self):
       self.print_to_log("\nMessage from config/user_classes/Example_user_class.py at the end of the run")