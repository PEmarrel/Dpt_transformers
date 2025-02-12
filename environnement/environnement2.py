class Environment2:
   """ In Environment 2, action 0 yields outcome 1, action 1 yields outcome 0"""
   def get_actions(self):
       """
       Return the list of actions in this environment
       """
       return [0, 1]
   
   def get_outcomes(self):
        """
        Return the list of outcomes in this environment
        """
        return [0, 1]

   def outcome(self, _action):
       if _action == 0:
           return 1
       else:
           return 0