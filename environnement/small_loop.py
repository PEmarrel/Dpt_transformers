import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def _display_world(world, robot):
    """
    Fct fais avec GPT
    Affiche un monde 2D avec des couleurs associées aux valeurs numériques et un robot.
    """
    # Définition des couleurs personnalisées
    cmap = mcolors.ListedColormap(["white", "black", "red", "blue", "green", "yellow"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]  # Délimite chaque couleur
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(world, cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    
    # Affichage du robot
    ax.scatter(robot.y, robot.x, color='orange', s=2000, edgecolors='black', label='Robot')
    
    triangle_offsets = {
        0: [(0, -0.1), (0.18, -0.3), (-0.18, -0.3)],  # Haut
        1: [(0.1, 0), (0.3, -0.18), (0.3, 0.18)],   # Droite
        2: [(0, 0.1), (0.18, 0.3), (-0.18, 0.3)],   # Bas
        3: [(-0.1, 0), (-0.3, -0.18), (-0.3, 0.18)],    # Gauche
    }
    
    triangle = triangle_offsets[robot.theta]
    triangle_x = [robot.y + dx for dx, dy in triangle]
    triangle_y = [robot.x + dy for dx, dy in triangle]
    ax.fill(triangle_x, triangle_y, color='black', label='Direction')
    
    # plt.legend()
    plt.show()

class state_robot:
    def __init__(self, x, y, theta):
        """
        Class to represent the state of the robot
        x: int, x position of the robot
        y: int, y position of the robot
        theta: int, "angle" of the robot 0 -> up, 1 -> right, 2 -> down, 3 -> left
        """
        self.x = x
        self.y = y
        self.theta = theta

    # def move_up(self): not use
    #     self.x -= 1

    # def move_down(self):
    #     self.x += 1

    # def move_left(self):
    #     self.y -= 1

    # def move_right(self):
    #     self.y += 1

    def turn_left(self):
        self.theta = (self.theta - 1) % 4

    def turn_right(self):
        self.theta = (self.theta + 1) % 4

    def __str__(self):
        return f"x: {self.x}, y: {self.y}, theta: {self.theta}"
    
    def __repr__(self):
        return f"x: {self.x}, y: {self.y}, theta: {self.theta}"
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.theta == other.theta
    
    def __hash__(self):
        return hash((self.x, self.y, self.theta))


class small_loop:

    def __init__(self, x=0, y=0, theta=0, world=None):
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
            1: "wall",
            0: "empty"
        }
        if world is not None:
            self.world = world
        else:
            self.world = np.array([
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
            ])

        # Check if the robot is in a wall
        if self.world[x, y] == 1:
            raise ValueError("The robot is in a wall")
        self.robot = state_robot(x, y, theta)

    def get_actions(self):
        return self.all_actions
    
    def get_outcomes(self):
        return list(self.outcomes.values())
    
    def get_world(self):
        return self.world
    
    def display_world(self):
        see_world = self.world.copy()
        # see_world[self.robot.x, self.robot.y] = 3
        _display_world(see_world, self.robot)

    def _forward(self):
        """
        Move the robot forward
        """
        x, y = self.robot.x, self.robot.y
        if self.robot.theta == 0:
            x -= 1
        elif self.robot.theta == 1:
            y += 1
        elif self.robot.theta == 2:
            x += 1
        elif self.robot.theta == 3:
            y -= 1

        if self.world[x, y] == 1:
            return self.outcomes[1]
        self.robot.x, self.robot.y = x, y
        return self.outcomes[0]
    
    def _turn_left(self):
        """
        Turn the robot left
        """
        self.robot.turn_left()
        return self.outcomes[0]
    
    def _turn_right(self):
        """
        Turn the robot right
        """
        self.robot.turn_right()
        return self.outcomes[0]
    
    def _feel_front(self):
        """
        Feel the case in front of the robot
        """
        x, y = self.robot.x, self.robot.y
        if self.robot.theta == 0:
            x -= 1
        elif self.robot.theta == 1:
            y += 1
        elif self.robot.theta == 2:
            x += 1
        elif self.robot.theta == 3:
            y -= 1

        if self.world[x, y] == 1:
            return self.outcomes[1]
        return self.outcomes[0]
    
    def _feel_left(self):
        """
        Feel the case on the left of the robot
        """
        x, y = self.robot.x, self.robot.y
        if self.robot.theta == 0:
            y -= 1
        elif self.robot.theta == 1:
            x -= 1
        elif self.robot.theta == 2:
            y += 1
        elif self.robot.theta == 3:
            x += 1

        if self.world[x, y] == 1:
            return self.outcomes[1]
        return self.outcomes[0]
    
    def _feel_right(self):
        """
        Feel the case on the right of the robot
        """
        x, y = self.robot.x, self.robot.y
        if self.robot.theta == 0:
            y += 1
        elif self.robot.theta == 1:
            x += 1
        elif self.robot.theta == 2:
            y -= 1
        elif self.robot.theta == 3:
            x -= 1

        if self.world[x, y] == 1:
            return self.outcomes[1]
        return self.outcomes[0]

    def make_action(self, action):
        """
        Execute an action on the robot
        
        action: str, action to execute (forward, turn_left, turn_right, feel_front, feel_left, feel_right)

        return: bool true if the robot can execute the action or feel empty case, false otherwise
        """
        match action:
            case "forward":
                return self._forward()
            case "turn_left":
                return self._turn_left()
            case "turn_right":
                return self._turn_right()
            case "feel_front":
                return self._feel_front()
            case "feel_left":
                return self._feel_left()
            case "feel_right":
                return self._feel_right()
            case _:
                raise ValueError(f"Action not recognized, you have '{action}'please choose between {self.all_actions}")
            
    def outcome(self, action):
        """
        Execute an action on the robot
        return 
        """
        return self.make_action(action)


