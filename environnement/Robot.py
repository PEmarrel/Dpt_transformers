class Robot:
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
