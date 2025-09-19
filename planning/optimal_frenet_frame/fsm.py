from config import DrivingMode

class StateMachine():
    def __init__(self):
        self.state = DrivingMode.VELOCITY_KEEPING
    
    def transition(self):
        '''
            (current_state, event) -> new_state
        '''
        transition_table = {
            (DrivingMode.VELOCITY_KEEPING, "stop_pos"): DrivingMode.STOPPING,
            (DrivingMode.VELOCITY_KEEPING, "stop_pos"): DrivingMode.STOPPING,
            (DrivingMode.VELOCITY_KEEPING, "stop_pos"): DrivingMode.STOPPING,
            (DrivingMode.VELOCITY_KEEPING, "stop_pos"): DrivingMode.STOPPING,
            (DrivingMode.VELOCITY_KEEPING, "stop_pos"): DrivingMode.STOPPING,
            (DrivingMode.VELOCITY_KEEPING, "stop_pos"): DrivingMode.STOPPING,
            (DrivingMode.VELOCITY_KEEPING, "stop_pos"): DrivingMode.STOPPING,
        }
