class Environment2:
   """ In Environment 2, action 0 yields outcome 1, action 1 yields outcome 0"""
   def get_actions(self):
      """
      Return the list of actions in this environment
      """
      return ['a', 'b']
   
   def get_outcomes(self):
      """
      Return the list of outcomes in this environment
      """
      return ['x', 'y']

   def outcome(self, _action):
      if _action == self.get_actions()[0]:
         return 'y'
      else:
         return 'x'