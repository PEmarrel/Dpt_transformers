import numpy as np

class Environment6:
    """ The grid """
    def __init__(self):
        """ Initialize the grid """
        self.grid = np.array([[1, 0, 0, 1]])
        self.position = 1
        
    def get_actions(self):
        """Return the list of actions in this environment"""
        return [0, 1]
    
    def get_outcomes(self):
        """Return the list of outcomes in this environment"""
        return [0, 1]
    

    def outcome(self, _action):
        """Take the action and generate the next outcome """
        if _action == 0:
            # Move left
            if self.position > 1:
                # No bump
                self.position -= 1
                self.grid[0, 3] = 1
                _outcome = 0
            elif self.grid[0, 0] == 1:
                # First bump
                _outcome = 1
                self.grid[0, 0] = 2
            else:
                # Subsequent bumps
                _outcome = 0
        else:
            # Move right
            if self.position < 2:
                # No bump
                self.position += 1
                self.grid[0, 0] = 1
                _outcome = 0
            elif self.grid[0, 3] == 1:
                # First bump
                _outcome = 1
                self.grid[0, 3] = 2
            else:
                # Subsequent bumps
                _outcome = 0
        return _outcome