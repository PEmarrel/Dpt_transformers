class Interaction:
    """An interaction is a tuple (action, outcome) with a valence"""
    def __init__(self, action, outcome, valence):
        self.action = action
        self.outcome = outcome
        self.valence = valence

    def getAction(self):
        return self.action
    
    def getDecision(self):
        return f'a{self.action}'

    def getPrimitiveAction(self):
        return self.action
    
    def getOutcome(self):
        return self.outcome
    
    def getValence(self):
        return self.valence

    def key(self):
        return f'{self.action}{self.outcome}'

    def preKey(self):
        return self.key()
    
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