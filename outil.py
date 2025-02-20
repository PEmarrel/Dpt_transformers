import numpy as np
import matplotlib.pyplot as plt

from environnement.environnement import Environnement as env

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
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.show()
    plt.close()

def create_dico_numerate_word(all_words):
    """
    Fonction qui crée un dictionnaire avec les mots et leur index
    """
    dico = {}
    for i, word in enumerate(all_words):
        dico[word] = i
    return dico