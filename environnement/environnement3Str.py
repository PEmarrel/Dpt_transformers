class Environment3:
    """ Environment 3 yields outcome 1 only when the agent alternates actions 0 and 1 """
    def __init__(self):
        """ Initializing Environment3 """
        self.previous_action = 'a'

    def get_actions(self):
        """Return the list of actions in this environment"""
        return ['a', 'b']
    
    def get_outcomes(self):
        """Return the list of outcomes in this environment"""
        return ['x', 'y']

    def outcome(self, _action):
        if _action == self.previous_action:
            _outcome = 'x'
        else:
            _outcome = 'y'
        self.previous_action = _action
        return _outcome