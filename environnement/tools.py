import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from .Robot import Robot

def creat_plot(world, robot:Robot):
    cmap = mcolors.ListedColormap(["white", "black", "red", "blue", "green", "yellow"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(world, cmap=cmap, norm=norm)
    ax.grid(False)

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

    ax.scatter(*tip, color='green', s=50)  # Tip of the triangle in red

    return fig, ax
    
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
    
def _display_world(world, robot):
    """
    Affiche un monde 2D avec des couleurs associées aux valeurs numériques et un robot.
    """
    fig, ax = creat_plot(world, robot)
    plt.show()
    plt.close(fig)
