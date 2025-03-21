from abc import ABC, abstractmethod

class Environnement(ABC):
    @abstractmethod
    def get_actions(self):
        """
        Function abstract to get spesific actions
        """
        pass
    
    @abstractmethod
    def get_outcomes(self):
        """
        Function abstract to get spesific outcomes
        """
        pass
    
    @abstractmethod
    def outcome(self, _action):
        """
        Function abstract to have soecific mecansim of outcome
        """
        pass
    
    def save_world(self):
        """
        Function to save the world
        """
        pass
    
    def display_world(self):
        """
        Function to display the world
        """
        pass