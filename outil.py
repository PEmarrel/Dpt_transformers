import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

from environnement.environnement import Environnement as env

def subfinder(mylist, pattern):
    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            matches.append(pattern)
            return matches
    return matches

def inter_action_and_feedback_size(history:list, size:int):
    """
    Transform history into input and target.
    history is a sequence of action and feedback.
    We want to have all sequence of size size, associate with the feedback of the last action (targets).
    Exemple:
    history = [('0', 'x'), ('1', 'y'), ('0', 'x'), ('1', 'y'), ('0', 'x'), ('1', 'y'), ('0', 'x')]
    size = 5
    inter_action_and_feedback_size(history, size) => 
    inputs = [['0', 'x', '1', 'y', '0'], 
    ['1', 'y', '0', 'x', '1'],
    ['0', 'x', '1', 'y', '0'],
    ['1', 'y', '0', 'x', '1'],
    ['0', 'x', '1', 'y', '0']]

    targets = ['x', 'y', 'x', 'y', 'x']

    """
    inputs, targets = [], []
    for act, ff in history:
        if inputs:
            temp = inputs[-1][-int(size - 2):] + [targets[-1], act]
            inputs.append(temp)
        else:
            inputs.append([act])
        targets.append(ff)
    return inputs[size - 1:], targets[size- 1:]
    

def make_data_set(tokenizer, _env:env, train_iter:int = 100, val_iter:int=1000):
    """ 
    Creates two datasets with random actions: a training dataset and a validation dataset.
    Args:
        tokenizer: A tokenizer object used to encode actions and feedback.
        _env (env): An environment object that provides actions and outcomes.
        train_iter (int, optional): The number of iterations for generating the training dataset. Default is 100.
        val_iter (int, optional): The number of iterations for generating the validation dataset. Default is 1000.
    Returns:
        tuple: A tuple containing four elements:
            - x_train (list): Encoded actions for the training dataset.
            - y_train (list): Encoded feedback for the training dataset.
            - x_test (list): Encoded actions for the validation dataset.
            - y_test (list): Encoded feedback for the validation dataset.
    """
    x_test = []
    y_test = []
    for i in range(val_iter):
        action = np.random.choice(_env.get_actions())
        feedback = _env.outcome(action)
        x_test.append(tokenizer.encode([str(action)]))
        y_test.append(str(feedback))
    y_test = tokenizer.encode(y_test)

    x_train = []
    y_train = []
    for i in range(train_iter):
        action = np.random.choice(_env.get_actions())
        feedback = _env.outcome(action)
        x_train.append(tokenizer.encode([str(action)]))
        y_train.append(str(feedback))

    y_train = tokenizer.encode(y_train)
    return x_train, y_train, x_test, y_test

def make_data_set_previous_interaction(tokenizer, env:env, context_lenght:int, rand_iter:int = 100, rand_iter_test:int = 1000):
    """
    Creates two datasets with random actions: a training dataset and a validation dataset.
    Args:

        tokenizer: A tokenizer object used to encode actions and feedback.
        env (env): An environment object that provides actions and outcomes.
        context_lenght (int): The number of previous interactions to consider.
        rand_iter (int, optional): The number of iterations for generating the training dataset. Default is 100.
        rand_iter_test (int, optional): The number of iterations for generating the validation dataset. Default is 1000.
    Returns:

        tuple: A tuple containing four elements:
            - x_train (list): Encoded actions for the training dataset.
            - y_train (list): Encoded feedback for the training dataset.
            - x_test (list): Encoded actions for the validation dataset.
            - y_test (list): Encoded feedback for the validation dataset.
    """
    history = []

    # Create data val
    historyTest = []
    for i in range(rand_iter_test):
        action = np.random.choice(env.get_actions())
        feedback = env.outcome(action)
        historyTest.append((str(action), str(feedback)))

    tmpXtest, tmpYtest = inter_action_and_feedback_size(historyTest, context_lenght)
    x_test = []
    for i, one_input in enumerate(tmpXtest):
        x_test.append(tokenizer.encode(one_input))
    y_test = tokenizer.encode(tmpYtest)

    # Create first rand sequence
    for i in range(rand_iter):
        action = np.random.choice(env.get_actions())
        feedback = env.outcome(action)
        history.append((str(action), str(feedback)))

    tmpXfit, tmpYfit = inter_action_and_feedback_size(history, context_lenght)
    x_fit = []
    for i, one_input in enumerate(tmpXfit):
        x_fit.append(tokenizer.encode(one_input))
    y_fit = tokenizer.encode(tmpYfit)

    return x_fit, y_fit, x_test, y_test

def see_evolued_loss(train_loss:list[list]):
    """
    Plots the evolution of training loss over epochs for multiple iterations.

    Args:
        train_loss: (list of list of float): A list where each element is a list of 
        loss values for each epoch in a particular iteration.

    Returns:
        None
    """
    for i, loss_list in enumerate(train_loss):
        plt.plot(loss_list, label=f'Iteration {i}', color=plt.cm.viridis(i / len(train_loss)))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    plt.close()

def see_evolued_acc(acc:list):
    """
    Plots the evolution of training loss over epochs for multiple iterations.

    Args:
        acc: (list of float): A list where each element is a list of 
        loss values for each epoch in a particular iteration.

    Returns:
        None
    """
    plt.plot(acc)
    plt.xlabel('Itérations')
    plt.ylabel('Acc')
    plt.show()
    plt.close()
    
def save_evolued_acc(acc:list, path:str="acc.png"):
    """
    Save the evolution of train acc for one epoch
    """
    plt.plot(acc)
    plt.xlabel('Itérations')
    plt.ylabel('Acc')
    plt.savefig(path)
    plt.close()
    
def create_dico_numerate_word(all_words):
    """
    Fonction qui crée un dictionnaire avec les mots et leur index
    """
    dico = {}
    for i, word in enumerate(all_words):
        dico[word] = i
    return dico

def find_sub_list(liste):
    pattern = {}
    
    for i in range(0, len(liste) - 4, 2):
        if pattern.get(str(liste[i:i+3])) == None:
            pattern[str(liste[i:i+3])] = {"fb":liste[i + 3], "count":1}
        else:
            if pattern[str(liste[i:i+3])]["count"] != -1:
                pattern[str(liste[i:i+3])]["count"] += 1
            if pattern[str(liste[i:i+3])]["fb"] != liste[i + 3]:
                pattern[str(liste[i:i+3])]["count"] = -1
    return pattern

def creat_plot_vision_agent(array, x, y, theta):
    cmap = mcolors.ListedColormap(["white", "black", "red", "blue", "green", "yellow"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(array, cmap=cmap, norm=norm)
    ax.grid(False)

    if theta == 0:
        plt.scatter(y, x, s=1000, marker='^')
        tip = (y, x-0.35)
    elif theta == 1:
        plt.scatter(y, x, s=1000, marker='>')
        tip = (y+0.35, x)
    elif theta == 2:
        plt.scatter(y, x, s=1000, marker='v')
        tip = (y, x+0.35)
    elif theta == 3:
        plt.scatter(y, x, s=1000, marker='<')
        tip = (y-0.35, x)

    ax.scatter(*tip, color='green', s=50)  # Tip of the triangle in red

    return fig, ax

def update_position(array, x, y, theta, act, fb):   
    direction_map = {
        0: {'forward': (x-1, y), 'feel_front': (x-1, y), 'feel_left': (x, y-1), 'feel_right': (x, y+1)},
        1: {'forward': (x, y+1), 'feel_front': (x, y+1), 'feel_left': (x-1, y), 'feel_right': (x+1, y)},
        2: {'forward': (x+1, y), 'feel_front': (x+1, y), 'feel_left': (x, y+1), 'feel_right': (x, y-1)},
        3: {'forward': (x, y-1), 'feel_front': (x, y-1), 'feel_left': (x+1, y), 'feel_right': (x-1, y)}
    }

    # Update the position or sense based on the action
    if act in direction_map[theta]:
        nx, ny = direction_map[theta][act]            
        if act == 'forward' and fb == 'empty':
            x, y = nx, ny
        array[nx, ny] = 3 if fb == 'empty' else 2
        if fb == '<pad>' or fb == '<mask>':
            array[nx, ny] = 6

    return x, y, array

def process_sequence(seq, size:int, path:str|None=None):
    """
    Process a sequence of actions and feedbacks, updating the array and position of the agent.
    Args:
        seq (list): Sequence of actions and feedbacks.
        size (int): Max size of environment.
        path (str|None): Path to save the plot (optional).
    Returns:
        list: List of arrays representing the memory space of agent at each step.
    """
    size *= 2
    array = np.ones((size, size))
    x, y = size // 2, size // 2
    theta = 0
    array[x, y] = 3
    
    list_array = []

    for i in range(0, len(seq), 2):
        act, fb = seq[i], seq[i + 1]

        if act == 'turn_left':
            theta = (theta - 1) % 4
            if fb == '<pad>' or fb == '<mask>':
                array[x, y] = 6
        elif act == 'turn_right':
            theta = (theta + 1) % 4
            if fb == '<pad>' or fb == '<mask>':
                array[x, y] = 6
        else:
            x, y, array = update_position(array, x, y, theta, act, fb)
        list_array.append(array.copy())
        
        if path is not None:
            fig, ax = creat_plot_vision_agent(array, x, y, theta)
            number = str(len(os.listdir(path)))
            plt.savefig(path + '/' + number + ".png")
            plt.close(fig)

    return list_array, x, y, theta

def process_sequence_inter(seq, size:int, path:str|None=None):
    """
    Process a sequence of actions and feedbacks, updating the array and position of the agent.
    Args:
        seq (list): Sequence of actions and feedbacks.
        size (int): Max size of environment.
        path (str|None): Path to save the plot (optional).
    Returns:
        list: List of arrays representing the memory space of agent at each step.
    """
    size *= 2
    array = np.ones((size, size))
    x, y = size // 2, size // 2
    theta = 0
    array[x, y] = 3
    
    list_array = []

    for element in seq:
        if type(element) == tuple:
            act, fb = element
        else:
            act, fb = element, '<pad>'
        if act == 'turn_left':
            theta = (theta - 1) % 4
            if fb == '<pad>' or fb == '<mask>':
                array[x, y] = 6
        elif act == 'turn_right':
            theta = (theta + 1) % 4
            if fb == '<pad>' or fb == '<mask>':
                array[x, y] = 6
        else:
            x, y, array = update_position(array, x, y, theta, act, fb)
        list_array.append(array.copy())
        
        if path is not None:
            fig, ax = creat_plot_vision_agent(array, x, y, theta)
            number = str(len(os.listdir(path)))
            plt.savefig(path + '/' + number + ".png")
            plt.close(fig)

    return list_array, x, y, theta

def info_in_memory(list_array, id_info = 6):
    x, y = np.where(list_array[-1] == id_info)
    return list_array[-2] [x, y] [0] != 1

def info_in_seq(seq, size):
    if type(seq[0]) == tuple:
        list_array, _, _, _ = process_sequence_inter(seq, size, None)
    else:
        list_array, _, _, _ = process_sequence(seq, size, None)
        
    x, y = np.where(list_array[-1] == 6)
    return list_array[-2] [x, y] [0] != 1
    
def info_step_in_memory(list_array, id_info = 6):
    """
    Return step where agent have the id_info in memory
    If don't have info, return -1
    """
    x, y = np.where(list_array[-1] == id_info)
    for i in range(len(list_array) - 1):
        if list_array[i][x, y] != 1:
            return i
    return -1

def feel_info_end_sequence(seq: list, size: int, tuple_info:tuple[list, int, int, int]=None):
    """
    This function analyzes a sequence of interactions and determines for each feeling if
    sequence have information.

    Parameters:
    seq (list): A list of interaction in this format ['action', 'feedback', ...].
    size (int): This size must correspond to twice the maximum length of the environment.

    Returns:
    list: A list of boolean values indicating whether each "feeling" action at the end 
          of the sequence provides information (True) or not (False).
    """
    last_step_info = -2 if seq[-1] == '<pad>' else -1
    if tuple_info is None:
        if type(seq[0]) == tuple:
            tuple_info = process_sequence_inter(seq, size, None)
        else:
            tuple_info = process_sequence(seq, size, None)
    list_array, x, y, theta = tuple_info
    directions_x = [
        -1,
        0,
        1,
        0,
    ]
    directions_y = [
        0,
        1,
        0,
        -1,
    ]

    feel_bool = []
    for i in range(-1, 2):
        theta_feel = (theta + i) % 4
        x_feel = x + directions_x[theta_feel]
        y_feel = y + directions_y[theta_feel]
        feel_bool.append(bool(list_array[last_step_info] [x_feel, y_feel] != 1))
    return feel_bool
