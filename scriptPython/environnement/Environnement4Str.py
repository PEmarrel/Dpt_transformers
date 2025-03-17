from ColabTurtle.Turtle import *

BORDER_WIDTH = 5

class ColabTurtleEnvironment:
    def __init__(self):
        """ Creating the Turtle window """
        bgcolor("lightGray")
        penup()
        goto(window_width() / 2, window_height()/2)
        face(0)
        pendown()
        color("green")

    def get_actions(self):
        """Return the list of actions in this environment"""
        return ['f', 'l', 'r']
    
    def get_outcomes(self):
        """Return the list of outcomes in this environment"""
        return ['o', 'x']

    def outcome(self, action):
        """ Enacting an action and returning the outcome """
        _outcome = 0
        for i in range(10):
            # _outcome = 0
            if action == "f":
                # move forward
                forward(10)
            elif action == "l":
                # rotate left
                left(4)
                forward(2)
            elif action == "r":
                # rotate right
                right(4)
                forward(2)

            # Bump on screen edge and return outcome 1
            if xcor() < BORDER_WIDTH:
                goto(BORDER_WIDTH, ycor())
                _outcome = 1
            if xcor() > window_width() - BORDER_WIDTH:
                goto(window_width() - BORDER_WIDTH, ycor())
                _outcome = 1
            if ycor() < BORDER_WIDTH:
                goto(xcor(), BORDER_WIDTH)
                _outcome = 1
            if ycor() > window_height() - BORDER_WIDTH:
                goto(xcor(), window_height() -BORDER_WIDTH)
                _outcome = 1

            # Change color
            if _outcome == 0:
                color("green")
            else:
                # Finit l'interaction
                color("red")
                # if action == 0:
                #     break
                if action == 1:
                    for j in range(10):
                        left(4)
                elif action == 2:
                    for j in range(10):
                        right(4)
                break

        return "o" if _outcome == 0 else "x"