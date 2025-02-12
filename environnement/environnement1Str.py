class Environment1:
    """ In Environment 1, action 0 yields outcome 0, action 1 yields outcome 1 """
    def get_actions(self):
        """Return the list of actions in this environment"""
        return ['a', 'b']
    
    def get_outcomes(self):
        """Return the list of outcomes in this environment"""
        return ['x', 'y']
    
    def outcome(self, _action):
        # return int(input("entre 0 1 ou 2"))
        if _action == 'a':
            return 'x'
        else:
            return 'y'