import numpy as np
from .tools import _display_world, _save_world
from .Robot import Robot
import os

raise NotImplementedError

class gridWorldMoreFB:
    def __init__(self, predator, prey, world=None):
        """
        Class to represent the environnement of the robot and robot
        """
        self.all_actions = [
            "forward",
            "turn_left",
            "turn_right",
            "feel_front",
            "feel_left",
            "feel_right",
        ]

        self.outcomes = {
            0: "wall",
            1: "empty"
        }
        
        if world is not None:
            self.world = world
        else:
            self.world = np.array([
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ])

        # Check if the robot is in a wall
        if self.world[x, y] == 1:
            raise ValueError("The robot is in a wall")
        print(f"The robot is in : { self.world[x, y]}  x: {x} y: {y}")
        print(f"World : {self.world}")
        self.robot = Robot(x, y, theta)

        self._box_obstacle_encountered = []
        self._box_feel = []        

    def get_actions(self):
        return self.all_actions
    
    def get_outcomes(self):
        return list(self.outcomes.values())
    
    def get_world(self):
        return self.world
    
    def get_robot(self):
        return self.robot
    
    def display_world(self, out:Output=None):
        """
        Display the world with the robot
        """
        world_seen = self.world.copy()
        for x, y in self._box_obstacle_encountered:
            world_seen[y, x] = 2
        for x, y in self._box_feel:
            world_seen[y, x] = 3
        _display_world(world_seen, self.robot, out)
        
    def save_world(self, path="imgToGif"):
        """
        Save the world (plot) in a file
        """
        world_seen = self.world.copy()
        for x, y in self._box_obstacle_encountered:
            world_seen[y, x] = 2
        for x, y in self._box_feel:
            world_seen[y, x] = 3

        _save_world(world_seen, self.robot, path)
        
    def _forward(self):
        """
        Move the robot forward
        """
        x, y = self.robot.x, self.robot.y
        if self.robot.theta == 0:
            y -= 1
        elif self.robot.theta == 1:
            x += 1
        elif self.robot.theta == 2:
            y += 1
        elif self.robot.theta == 3:
            x -= 1

        if self.world[y, x] == 1:
            self._box_obstacle_encountered.append((x, y))
            return self.outcomes[0]
        self.robot.x, self.robot.y = x, y
        return self.outcomes[1]
    
    def _turn_left(self):
        """
        Turn the robot left
        """
        self.robot.turn_left()
        return self.outcomes[1]
    
    def _turn_right(self):
        """
        Turn the robot right
        """
        self.robot.turn_right()
        return self.outcomes[1]
    
    def _feel_front(self):
        """
        Feel the case in front of the robot
        """
        x, y = self.robot.x, self.robot.y
        i = 0
        if self.robot.theta == 0:
            for _ in range(0, self.range_feel + 1):
                y -= 1
                self._box_feel.append((x, y))
                if self.world[y, x] == 1:
                    break
                i += 1
        elif self.robot.theta == 1:
            for _ in range(0, self.range_feel):
                x += 1
                self._box_feel.append((x, y))
                if self.world[y, x] == 1:
                    break
                i += 1
        elif self.robot.theta == 2:
            for _ in range(0, self.range_feel):
                y += 1
                self._box_feel.append((x, y))
                if self.world[y, x] == 1:
                    break
                i += 1
        elif self.robot.theta == 3:
            for _ in range(0, self.range_feel):
                x -= 1
                self._box_feel.append((x, y))
                if self.world[y, x] == 1:
                    break
                i += 1
        return self.outcomes[i]
    
    def _feel_left(self):
        """
        Feel the case on the left of the robot
        """
        x, y = self.robot.x, self.robot.y
        i = 0
        if self.robot.theta == 0:
            for _ in range(0, self.range_feel):
                x -= 1
                self._box_feel.append((x, y))
                if self.world[y, x] == 1:
                    break
                i += 1
        elif self.robot.theta == 1:
            for _ in range(0, self.range_feel):
                y -= 1
                self._box_feel.append((x, y))
                if self.world[y, x] == 1:
                    break
                i += 1
        elif self.robot.theta == 2:
            for _ in range(0, self.range_feel):
                x += 1
                self._box_feel.append((x, y))
                if self.world[y, x] == 1:
                    break
                i += 1
        elif self.robot.theta == 3:
            for _ in range(0, self.range_feel):
                y += 1
                self._box_feel.append((x, y))
                if self.world[y, x] == 1:
                    break
                i += 1
        return self.outcomes[i]
    
    def _feel_right(self):
        """
        Feel the case on the right of the robot
        """       
        i = 0
        x, y = self.robot.x, self.robot.y
        if self.robot.theta == 0:
            for _ in range(0, self.range_feel):
                x += 1
                self._box_feel.append((x, y))
                if self.world[y, x] == 1:
                    break
                i += 1
        elif self.robot.theta == 1:
            for _ in range(0, self.range_feel):
                y += 1
                self._box_feel.append((x, y))
                if self.world[y, x] == 1:
                    break
                i += 1
        elif self.robot.theta == 2:
            for _ in range(0, self.range_feel):
                x -= 1
                self._box_feel.append((x, y))
                if self.world[y, x] == 1:
                    break
                i += 1
        elif self.robot.theta == 3:
            for _ in range(0, self.range_feel):
                y -= 1
                self._box_feel.append((x, y))
                if self.world[y, x] == 1:
                    break
                i += 1
        return self.outcomes[i]

    def make_action(self, action):
        """
        Execute an action on the robot
        
        action: str, action to execute (forward, turn_left, turn_right, feel_front, feel_left, feel_right)

        return: bool true if the robot can execute the action or feel empty case, false otherwise
        """
        self._box_obstacle_encountered = []
        self._box_feel = []
        if action == "forward":
            return self._forward()
        if action == "turn_left":
            return self._turn_left()
        if action == "turn_right":
            return self._turn_right()
        if action == "feel_front":
            return self._feel_front()
        if action == "feel_left":
            return self._feel_left()
        if action == "feel_right":
            return self._feel_right()
        raise ValueError(f"Action not recognized, you have '{action}'please choose between {self.all_actions}")
            
    def outcome(self, action):
        """
        Execute an action on the robot
        return 
        """
        return self.make_action(action)


