from bayes_opt import BayesianOptimization
import numpy as np
import torch.optim.adam
import torch.optim.adamw

from model.DeepNN import *
from model.RNN import *
from model.CustomDataSet import *
from model.Tokenizer import *
from environnement.environnement import Environnement as env
from environnement.small_loop import small_loop 


def create_dico_numerate_word(all_words):
    """
    Fonction qui crée un dictionnaire avec les mots et leur index
    """
    dico = {}
    for i, word in enumerate(all_words):
        dico[word] = i
    return dico

def make_data_set(env:env, train_iter:int = 100, val_iter:int=100):
    """
    Cette fonction créer un nombre d'interaction sur un environement,
    les actions sont choisit de manière aléatoire.

    :param: **env** environementrand_iter
    :param: **train_iter** nombre d'interaction pour le train
    :param: **val_iter** nombre d'interaction pour la validation

    :return: **act_fit** liste d'action pour le train
    :return: **out_fit** liste d'outcome pour le train
    :return: **act_val** liste d'action pour la validation
    :return: **out_val** liste d'outcome pour la validation
    """
    act_fit, out_fit, act_val, out_val = [], [], [], []
    for i in range(train_iter):
        action = str(np.random.choice(env.get_actions()))
        feedback = env.outcome(action)
        act_fit.append(action)
        out_fit.append(feedback)

    for i in range(val_iter):
        action = str(np.random.choice(env.get_actions()))
        feedback = env.outcome(action)
        act_val.append(action)
        out_val.append(feedback)

    return act_fit, out_fit, act_val, out_val

# list_hidden_size:list,
def dnn_cv(lr:float, weight_decay:float, epochs:int, context_lenght:int):
    """

    """
    env = small_loop(x=1, y=1, theta=0)
    all_word = create_dico_numerate_word(env.get_outcomes() + env.get_actions())
    tokenizer = SimpleTokenizerV1(all_word)
    iter_train:int = 1000
    iter_val:int = 10000
    context_lenght = int(context_lenght)
    if context_lenght % 2 == 0:
        context_lenght += 1
    input_size:int = context_lenght
    output_size:int = len(env.get_outcomes())
    
    act_fit, out_fit, act_val, out_val = make_data_set(env, train_iter=iter_train, val_iter=iter_val)
    # list_hidden_size = list(map(int, list_hidden_size))

    test_model = DeepNetwork(hidden_size=hidden_size_list,
                        input_size=input_size,
                        output_size=output_size)
    
    optimizer = torch.optim.Adam(test_model.parameters(), lr=lr, weight_decay=weight_decay) 
    loss_fn = torch.nn.CrossEntropyLoss()


    data_set_train = CustomDataSet(actions=act_fit, 
                            outcomes=out_fit,
                            tokenizer=tokenizer,
                            context_lenght=context_lenght,
                            dim_out=output_size)
    
    data_set_val = CustomDataSet(actions=act_val, 
                            outcomes=out_val,
                            tokenizer=tokenizer,
                            context_lenght=context_lenght,
                            dim_out=output_size) 
    data_loader_train = torch.utils.data.DataLoader(data_set_train, batch_size=32, shuffle=True)
    data_loader_val = torch.utils.data.DataLoader(data_set_val, batch_size=32, shuffle=False)

    loss, last_acc, _, _, _, _ =train_with_batch(model=test_model, 
                    train_loader=data_loader_train,
                    optimizer=optimizer,
                    loss_func=loss_fn, 
                    nb_epochs=round(epochs),
                    print_=False,
                    validate_loader=data_loader_val)
    return last_acc

def lstm_cv(lr:float, weight_decay:float, epochs:int, context_lenght:int, dropout:float, hidden_size:int):
    """

    """
    env = small_loop(x=1, y=1, theta=0)
    all_word = create_dico_numerate_word(env.get_outcomes() + env.get_actions())
    tokenizer = SimpleTokenizerV1(all_word)
    iter_train:int = 50
    iter_val:int = 1000
    context_lenght = int(context_lenght)
    if context_lenght % 2 == 0:
        context_lenght += 1
    output_size:int = len(env.get_outcomes())
    
    act_fit, out_fit, act_val, out_val = make_data_set(env, train_iter=iter_train, val_iter=iter_val)
    # list_hidden_size = list(map(int, list_hidden_size))

    test_model = LSTM(dropout=round(dropout, 2),
                        hidden_size=int(hidden_size),
                        output_size=output_size,
                        num_emb=len(all_word),
                        num_layers=4)
    
    optimizer = torch.optim.Adam(test_model.parameters(), lr=lr, weight_decay=weight_decay) 
    loss_fn = torch.nn.CrossEntropyLoss()


    data_set_train = CustomDataSetRNN(actions=act_fit, 
                            outcomes=out_fit,
                            tokenizer=tokenizer,
                            context_lenght=context_lenght,
                            dim_out=output_size)
    
    data_set_val = CustomDataSetRNN(actions=act_val, 
                            outcomes=out_val,
                            tokenizer=tokenizer,
                            context_lenght=context_lenght,
                            dim_out=output_size) 
    data_loader_train = torch.utils.data.DataLoader(data_set_train, batch_size=32, shuffle=True)
    data_loader_val = torch.utils.data.DataLoader(data_set_val, batch_size=32, shuffle=False)

    train_acc, test_acc = train_LSTM(model=test_model, 
                    train_loader=data_loader_train,
                    optimizer=optimizer,
                    loss_func=loss_fn, 
                    nb_epochs=round(epochs),
                    validate_loader=data_loader_val)
    return test_acc


# for hidden_size_list in [[8], [16], [32, 4]]:
#     params_gbm ={
#         # 'list_hidden_size': ([16], [32], [64, 32], [128, 64, 32]),
#         'lr': (0.0001, 0.1),
#         'weight_decay': (0.0001, 0.1),
#         'epochs': (10, 100),
#         'context_lenght': (1, 20)
#     }
#     # Créez un objet BayesianOptimization
#     bo = BayesianOptimization(
#         f=dnn_cv,
#         pbounds=params_gbm,
#         verbose=2,
#         random_state=1
#     )

#     # Effectuez l'optimisation
#     bo.maximize(init_points=10, n_iter=50)

#     # Meilleurs hyperparamètres et précision correspondante
#     best_params = bo.max['params']
#     best_accuracy = bo.max['target']

#     print(f"Meilleurs Hyperparamètres : {best_params}")
#     print(f"Best accuracie : {best_accuracy}")
#     # add lign in file
#     with open(f'resultat{str(hidden_size_list)}.txt', 'a') as f:
#         f.write(f"Meilleurs Hyperparamètres : {best_params}")
#         f.write(f"Best accuracie : {best_accuracy}")
#         f.write("\n")


params_gbm ={
    # 'list_hidden_size': ([16], [32], [64, 32], [128, 64, 32]),
    'lr': (0.0001, 0.1),
    'weight_decay': (0.0001, 0.1),
    'epochs': (70, 100),
    'context_lenght': (1, 30),
    'dropout': (0.1, 0.7),
    'hidden_size': (64, 256)
}
# Créez un objet BayesianOptimization
bo = BayesianOptimization(
    f=lstm_cv,
    pbounds=params_gbm,
    verbose=2,
    random_state=1
)

# Effectuez l'optimisation
bo.maximize(init_points=20, n_iter=100)

# Meilleurs hyperparamètres et précision correspondante
best_params = bo.max['params']
best_accuracy = bo.max['target']

print(f"Meilleurs Hyperparamètres : {best_params}")
print(f"Best accuracie : {best_accuracy}")
# add lign in file
with open(f'resultat.txt', 'a') as f:
    f.write(f"Meilleurs Hyperparamètres : {best_params}")
    f.write(f"Best accuracie : {best_accuracy}")
    f.write("\n")