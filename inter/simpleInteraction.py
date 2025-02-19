class simpleInteraction:
    """An interaction is a tuple (action, outcome)"""
    def __init__(self, action, outcome):
        self.action = action
        self.outcome = outcome

    def getAction(self):
        return self.action
    
    def getOutcome(self):
        return self.outcome

    def key(self):
        return f'{self.action}{self.outcome}'

    def __str__(self):
        return f'{self.action}{self.outcome}:{self.valence}'

    def __repr__(self):
        return f'{self.action}{self.outcome} => {self.valence}'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.key() == other.key()
        else:
            return False
    
    def __hash__(self):
        return hash(self.key())