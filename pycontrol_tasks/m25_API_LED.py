from pyControl.utility import *
from devices import Grid_maze_7x7, Rsync, LED_driver #, Solenoid_calibration

# sol_cal = Solenoid_calibration()
maze = Grid_maze_7x7()

sync_output = Rsync(pin=maze.BNC_1, mean_IPI=5000)
stim = LED_driver(maze.port_1) # Instantiate LED - WHAT PORT DO WE NEED (LED_driver(port))

v.api_class = 'm25_API'

# States and events
states = [
    "cue",  # Period between goal cue and goal nose poke
    "poked_in",  # Successfull nose poke at goal port
    "reward",  # Delivery of water reward at goal port
    "ITI",
]  # Inter-trial interval (period between trials)

events = maze.events + [  # Pre-programmed maze events
    "reward_consumption_timer",  # Time mice can be poked out of a reward port before moving into the ITI
    "cue_noise_off_timer",  # How long the cue nose is played
    "reward_duration_timer",  # Timer for the amount of reward delivered
    "session_timer",  # Total session time
    "rsync", 
    "pre_stim_timer", # timer between trial start and stim on
    "max_stim_timer", # timer for maximum stimulation duration
    "pre_stim_off_timer", # short timer after stim of trigger arrives but stim is still on for a bit
    "stim_on_trigger", # trigger for stimulation coming from api
    "stim_off_trigger" # trigger to end stimulation coming from api
]  # Camera sync pulse

# Starting state
initial_state = "ITI"

# Variables
GOAL_SETS = {
    "all": [
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
        "E7",
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
        "G1",
        "G2",
        "G3",
        "G4",
        "G5",
        "G6",
        "G7",
    ],
    "m25": ["B2", "B3", "B4", "B5", "B6",
            "E6", "D6", "C6",
            "E2", "D2", "C2",
            "F2", "F3", "F4", "F5", "F6"]}

v.goal_set = "m25"

#variables to pick a subset of pokes opposite from the previous choice
v.goal_subset = []
v.current_index = 0  
v.goals_to_pick = ["B2", "B3", "B4", "B5", "B6", "C6", "D6", "E6",
                "F6","F5",  "F4",  "F3","F2",  "E2", "D2","C2",
                "B2", "B3", "B4", "B5", "B6", "C6", "D6", "E6",
                "F6","F5",  "F4",  "F3","F2",  "E2", "D2","C2"]

v.n_trials = 0  # Number of trials completed
v.current_goal = None  # Current goal from v.goal_set being cued
v.trial_start_time = 0  # Time port is cued at the start of the trial
v.trial_end_time = 0  # Time mouse has finished geting the reward and enters ITI
v.error_poke_list = []  # List of incorrect poke locations for each trial
v.n_stims = 0 # number of times stimulation was delivered

# Parameters
v.session_duration = 40  # Duration of entire session
v.audio_cue_duration = 0.5 * second  # Duration of audio tone during goal cueing
v.min_ITI_dur = 4  # Minimum ITI duration (seconds).
v.max_ITI_dur = 8  # Maximum ITI duration (seconds).
v.reward_consumption_dur = 1 * second  # Timer for drinking breaks during reward state
v.reward_duration = None
v.reward_vol_ul = 40
v.max_reward_state_dur = 15 * second  # Timer for max drinking time during reward state

# Stim parameters
v.pre_stim_duration = 0.5 * second # this is how much will need to pass between cue on a possible stim start
v.max_stim_duration = 15 * second # note that pre_stim_off_timer_duration is added to this to get fullmex duration
v.pre_stim_off_timer_duration = 0.5 * second # this is a short timer after stim of trigger arrives but stim is still on for a bit

# v.stim_freq = 0 #in Hz (not gonna use cuz we have contiunous stimulation here)
v.stim_prob = 0.2 #stim decision could also be made in api but maybe better here
v.stim_decision = False # this will be a boolean that is true if we're stimulating on this trial
v.stim_start_distance  = 3 # how many nodes away from goal to start stim
v.stim_stop_distance = 1 # how many nodes away from goal to end stim

# Custom controls dialog declaration - probs dont need this, stangard custom diagog has trigger of events
# Uncomment 1 of the 3 below. (files found in config/user_controls_dialogs/)
#v.custom_controls_dialog = "example_blinker_gui"  # example dialog
# v.custom_controls_dialog = 'example_blinker_gui_tabs' # example dialog with tabs
# v.custom_controls_dialog = 'example_blinker_gui_from_py' # advanced example dialog that is loaded from a .py file

v.hw_solenoid_calibration = {"slope": 7, "intercept": 1.65}  # Linear fit ms/ul.

# Define behaviours.
def run_start():
    maze.audio.set_volume(10)
    v.reward_duration = v.reward_vol_ul * v.hw_solenoid_calibration["slope"] + v.hw_solenoid_calibration["intercept"]
    v.goal_sampler = sample_without_replacement(
        GOAL_SETS[v.goal_set]
    )  # Sample goals without replacement (repeats when empty)
    set_timer("session_timer", v.session_duration * minute)
    print(f'Stim_start_distance_is_{v.stim_start_distance}')
    print(f'Stim_stop_distance_is_{v.stim_stop_distance}')

# This state sets a randomised time interval (between min_ITI_dur and max_ITI_dur) between successive trials and resets counter variables
def ITI(event):
    if event == "entry":
        v.error_poke_list = []
        timed_goto_state("cue", randint(v.min_ITI_dur * second, v.max_ITI_dur * second))


# This state chooses a random goal port from the goal set (without replacement), turns on the LED at this port and plays whitenose at the goal port
# for a duration set by 'cue_noise_duration'. If mice poke in the correct goal they are sent to the poked_in state, if they poke a non-goal port
# an error is recorded
# It also prints start trial information: T = trial no., S = current goal port, sT = start time
def cue(event):
    if event == "entry":
        v.n_trials += 1
        v.trial_start_time = get_current_time()
        v.current_goal = v.goal_sampler.next() #this we can access in vars in api
        maze.LED_on(v.current_goal)
        maze.speaker_on(v.current_goal)
        maze.audio.noise()
        set_timer("cue_noise_off_timer", v.audio_cue_duration)
        set_timer("pre_stim_timer", v.pre_stim_duration) # ensures stim cannot be triggered too early
        v.stim_decision = False

    elif event == "cue_noise_off_timer":
        maze.speaker_off(v.current_goal)
        maze.audio.off()

    elif event == "pre_stim_timer":
        if withprob(v.stim_prob): #TODO: test is this works correctly
            v.stim_decision = True 
        print(f'Stim_decision_is_{v.stim_decision}')
        print(f'Current_goal_is_{v.current_goal}')

    elif event == "stim_on_trigger": # this is stim on event that will be triggered by the api
        stim.on()
        set_timer("max_stim_timer", v.max_stim_duration)
        v.n_stims += 1

    elif event == "max_stim_timer":  # stim either turns off if timer up
        stim.off() # in either case stim turns off after short delay

    elif event == "stim_off_trigger":  # or if api sends a trigger
        set_timer("pre_stim_off_timer", v.pre_stim_off_timer_duration)
        disarm_timer("max_stim_timer")

    elif event == "pre_stim_off_timer":
        stim.off() #stim turns off after short delay
        
    elif event[-3:] == "_in":
        if event[:2] != v.current_goal:
            if not v.error_poke_list:
                v.error_poke_list.append(event[:2])
            elif event[:2] != v.error_poke_list[-1]: #THIS RECORDS EVERY POKE THAT IS NOT IN THE SAME PORT AS THE LAST ERROR
                v.error_poke_list.append(event[:2])
        elif event[:2] == v.current_goal:
            goto_state("poked_in")
    elif event == "exit":
        maze.speaker_off(v.current_goal)
        maze.audio.off()
        maze.LED_off(v.current_goal)
        disarm_timer("cue_noise_off_timer")
        disarm_timer("pre_stim_timer")
        disarm_timer("max_stim_timer")
        disarm_timer("pre_stim_off_timer")

        #redefine a goal sampler to use an opposite subset in the next trial
        v.current_index = GOAL_SETS[v.goal_set].index(v.current_goal)
        v.goal_subset = v.goals_to_pick[v.current_index+6:v.current_index+12]
        v.goal_sampler = sample_without_replacement(v.goal_subset) 


# This state counts successful trials and sends mice to the reward state with a fixed delay of 200ms.
def poked_in(event):
    if event == "entry":
        v.trial_end_time = get_current_time()
        timed_goto_state("reward", 200)


# This state activates the solenoid at the goal port once mice have poked into the correct port, for a given duration (reward_duration), delivering
# a fixed amount of reward (see solenoid calibration spreadsheet). Mice are allowed to have drinking breaks where they may poke out of the reward port
# for a max duration set by reward_consumption_timer, once this timer has elapsed or mice reach the max_reward_state_dur they move to the ITI state.
# Trial variables are also printed here. E = no. incorrect ports visted before goal port (doesn't count multiple pokes into the same incorrect poke). eT = end trial time. D = trial duration
def reward(event):
    if event == "entry":
        # v.reward_duration = Solenoid_calibration.get_release_duration(v.current_goal, v.reward_vol_ul)
        set_timer("reward_duration_timer", v.reward_duration)
        timed_goto_state("ITI", v.max_reward_state_dur)
        v.trial_duration = v.trial_end_time - v.trial_start_time
        v.n_errors = len(v.error_poke_list)
        print_variables(
            [
                "n_trials",
                "n_errors",
                "n_stims",
                "current_goal",
                "trial_start_time",
                "trial_end_time",
                "trial_duration",
                "reward_duration",
                "stim_decision"
            ]
        )
        maze.SOL_on(v.current_goal)
    elif event == "exit":
        disarm_timer("reward_consumption_timer")
    elif event == "reward_duration_timer":
        maze.SOL_off(v.current_goal)
    elif event[-4:] == "_out" and event[:2] == v.current_goal:
        set_timer("reward_consumption_timer", v.reward_consumption_dur)
    elif event[-3:] == "_in" and event[:2] == v.current_goal:
        disarm_timer("reward_consumption_timer")
    elif event == "reward_consumption_timer" or event == "max_reward_state_durr":
        goto_state("ITI")


def all_states(event):
    if event == "session_timer":
        stop_framework()
