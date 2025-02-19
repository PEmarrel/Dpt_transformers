import numpy as np
import matplotlib.pyplot as plt

from environnement.environnement import Environnement as env

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
    for i in range(1000):
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