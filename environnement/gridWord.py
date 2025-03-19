import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from IPython.display import display, update_display
from ipywidgets import Output
import os


def creat_plot(world, robot):
    cmap = mcolors.ListedColormap(["white", "black", "red", "blue", "green", "yellow"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(world, cmap=cmap, norm=norm)
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.grid(False)

    # Affichage du robot
    # ax.scatter(robot.y, robot.x, color='orange', s=2000, edgecolors='black', label='Robot')

    # triangle_offsets = {
    #     0: [(0, -0.1), (0.18, -0.3), (-0.18, -0.3)],  # Haut
    #     1: [(0.1, 0), (0.3, -0.18), (0.3, 0.18)],   # Droite
    #     2: [(0, 0.1), (0.18, 0.3), (-0.18, 0.3)],   # Bas
    #     3: [(-0.1, 0), (-0.3, -0.18), (-0.3, 0.18)],    # Gauche
    # }
    
    if robot.theta == 0:
        plt.scatter(robot.x, robot.y, s=1000, marker='^')
        tip = (robot.x, robot.y-0.35)
    elif robot.theta == 1:
        plt.scatter(robot.x, robot.y, s=1000, marker='>')
        tip = (robot.x+0.35, robot.y)
    elif robot.theta == 2:
        plt.scatter(robot.x, robot.y, s=1000, marker='v')
        tip = (robot.x, robot.y+0.35)
    elif robot.theta == 3:
        plt.scatter(robot.x, robot.y, s=1000, marker='<')
        tip = (robot.x-0.35, robot.y)

    # ax.fill(*zip(*triangle), color='orange', edgecolor='black')
    ax.scatter(*tip, color='green', s=50)  # Tip of the triangle in red

    return fig, ax
    
    
def _display_world(world, robot, out:Output):
    """
    Affiche un monde 2D avec des couleurs associées aux valeurs numériques et un robot.
    """
    fig, ax = creat_plot(world, robot)
    if out is not None:
        out.clear_output()
        with out:
            plt.show()
    else:
        plt.show()
    plt.close(fig)

def _save_world(world, robot, path):
    """
    Affiche un monde 2D avec des couleurs associées aux valeurs numériques et un robot.
    """
    fig, ax = creat_plot(world, robot)
    # add title to path + number of image + png
    number = str(len(os.listdir(path)))
    # Add number in plot
    ax.text(0, 0, number, fontsize=12, color='White')
    plt.savefig(path + '/' + number + ".png")
    plt.close(fig)
    
# class WorldDisplay:
#     def __init__(self):
#         pass

#     def show(self, world, robot, out):
#         """
#         Affiche ou met à jour un monde 2D avec des couleurs et un robot.
#         """

#         cmap = mcolors.ListedColormap(["white", "black", "red", "blue", "green", "yellow"])
#         bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
#         norm = mcolors.BoundaryNorm(bounds, cmap.N)

#         fig, ax = plt.subplots(figsize=(5, 5))
#         ax.imshow(world, cmap=cmap, norm=norm)
#         # ax.set_xticks([])
#         # ax.set_yticks([])
#         ax.grid(False)

#         # Affichage du robot
#         ax.scatter(robot.y, robot.x, color='orange', s=2000, edgecolors='black', label='Robot')

#         triangle_offsets = {
#             0: [(0, -0.1), (0.18, -0.3), (-0.18, -0.3)],  # Haut
#             1: [(0.1, 0), (0.3, -0.18), (0.3, 0.18)],   # Droite
#             2: [(0, 0.1), (0.18, 0.3), (-0.18, 0.3)],   # Bas
#             3: [(-0.1, 0), (-0.3, -0.18), (-0.3, 0.18)],    # Gauche
#         }

#         triangle = triangle_offsets[robot.theta]
#         triangle_x = [robot.y + dx for dx, dy in triangle]
#         triangle_y = [robot.x + dy for dx, dy in triangle]
#         ax.fill(triangle_x, triangle_y, color='black', label='Direction')
#         out.clear_output(wait=True)
#         with out:
#             plt.show()
        # plt.close(fig)  

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


class gridWord:
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
        print(f"The robot is in : { self.world[x, y]}  x: {x} y: {y}")
        print(f"World : {self.world}")
        self.robot = state_robot(x, y, theta)

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
            y -= 1
        elif self.robot.theta == 1:
            x += 1
        elif self.robot.theta == 2:
            y += 1
        elif self.robot.theta == 3:
            x -= 1
        self._box_feel.append((x, y))
        if self.world[y, x] == 1:
            return self.outcomes[1]
        return self.outcomes[0]
    
    def _feel_left(self):
        """
        Feel the case on the left of the robot
        """
        x, y = self.robot.x, self.robot.y
        if self.robot.theta == 0:
            x -= 1
        elif self.robot.theta == 1:
            y -= 1
        elif self.robot.theta == 2:
            x += 1
        elif self.robot.theta == 3:
            y += 1
            
        self._box_feel.append((x, y))
        if self.world[x, y] == 1:
            return self.outcomes[1]
        return self.outcomes[0]
    
    def _feel_right(self):
        """
        Feel the case on the right of the robot
        """
        x, y = self.robot.x, self.robot.y
        if self.robot.theta == 0:
            x += 1
        elif self.robot.theta == 1:
            y += 1
        elif self.robot.theta == 2:
            x -= 1
        elif self.robot.theta == 3:
            y -= 1
            
        self._box_feel.append((x, y))
        if self.world[x, y] == 1:
            return self.outcomes[1]
        return self.outcomes[0]

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


