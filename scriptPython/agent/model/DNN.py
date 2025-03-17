from typing import Tuple
from torch import nn
import torch

class DeepNetwork(nn.Module):
    """
    Cette classe utilise plusieurs couche cachées
    Pour définir un réseau de neurones profonds
    
    - Il y a une couche d'entrée
    - Plusieurs couches cachées dépendant de la liste hidden_size
    - Une couche de sortie

    les poids sont initialise de manière aléatoire
    """
    
    name = "DeepNetwork"
    
    def __init__(self, input_size:int, hidden_size:list[int], output_size:int):
        """
        Constructeur de la classe, applique le constructeur de la classe parent
        Et crée les couches du réseau :
        - 1 couche d'entrée
        - Plusieurs couches cachées
        - 1 couche de sortie
        
        Les couches cachées sont définies mis en place avec nn.ModuleList, pour permettre
        l'ajout dynamique de couches cachées. (Concretement nous pouvons passer le 
        modele.to(deviece) et la sauvgarde des poids par torch)

        :param input_size: la taille des données d'entrée
        :param output_size: la taille des données de sortie
        :param hidden_size: la taille des couches cachées, 
        le nombre d'éléments dans la liste correspond au nombre de couches cachées       
        """
        super(DeepNetwork, self).__init__()
        # nn.Linear initialise les poids de manière aléatoire
        print("liste hidden init",hidden_size)
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size[0])

        self.hidden_layers = nn.ModuleList()
        for size in range(len(hidden_size) - 1):
            self.hidden_layers.append(nn.Linear(hidden_size[size], hidden_size[size + 1]))
        self.fc4 = nn.Linear(hidden_size[-1], output_size)
        self._init_weights()

    def _init_weights(self):
        """
        Initialisation des poids des couches
        """
        for layer in self.hidden_layers:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc4.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        for layer in self.hidden_layers:
            x = nn.functional.relu(layer(x))
        x = self.fc4(x)
        return x
    
def evaluate(model: nn.Module, test_loader: torch.utils.data.DataLoader, loss_funct) -> tuple[float, float]:
    """
    Méthode d'évaluation du modèle

    :param model: le modèle à évaluer
    :param test_loader: le lecteur de données de test
    :param loss_funct: la fonction de perte à utiliser pour l'évaluation
    :param device: le device sur lequel effectuer les calculs
    :return: une tuple contenant (le taux de réussite, la perte moyenne)
    """
    device = next(model.parameters()).device # get device of the model

    acc = 0.0
    total_loss = 0.0
    total_samples = 0
    model.to(device)
    model.eval()  # Pour désactiver les couches dropout ou batchnorm
    # for x,t in test_loader:
    #     print(f"test data : {x}")
    #     print(f"test target : {t}")
    #     input( )
    with torch.no_grad():  # Pas besoin de calculer les gradients en mode évaluation
        x:torch.Tensor = None; t:torch.Tensor = None
        for x, t in test_loader:
            x, t = x.to(device), t.to(device)
            # Prédictions
            y = model(x)
            # Calcul de la loss
            loss = loss_funct(y, t)
            total_loss += loss.item()
            # Calcul de la précision
            if len(t.shape) > 1 and t.shape[1] > 1:  # Si les labels sont one-hot encodés
                t = torch.argmax(t, dim=1)  # On convertit en indices de classes
            acc += (torch.argmax(y, 1) == t).sum().item()            
            total_samples += t.size(0)

    avg_acc = acc / total_samples
    avg_loss = total_loss / len(test_loader)

    return avg_acc, avg_loss

def train(model:nn.Module,
        train_data: list[Tuple[torch.Tensor, torch.Tensor]],
        optimizer:torch.optim.SGD,
        loss_func:torch.nn.MSELoss,
        nb_epochs:int,
        validate_loader:torch.utils.data.DataLoader=None,
        print_:bool=False
        ) -> Tuple[float, nn.Module]:
    """
    Méthode d'entraînement du modèle

    Pour chaque époque:
    - on entraîne le modèle sur les données d'apprentissage
    - Puis on évalue le modèle sur les données de validation

    Au final nous renvoyons le modèle avec le meilleur taux de réussite
    (Nous évitons le sur-apprentissage tout en testant l'entièreté des époques)

    :param model: le modèle à entraîner
    :param train_loader: les données d'entraînement
    :param optimizer: l'optimiseur lié au modèle
    :param loss_func: la fonction de loss
    :param nb_epochs: le nombre d'époques à effectuer sur la même donnée
    :param print_: afficher ou non les résultats
    :return: le taux de réussite final et le meilleur modèle
    """
    device = next(model.parameters()).device # get device of the model
    model.train()
    loss_func.to(device)

    meilleur_acc = 0.0
    meilleur_model = model

    
    for epoch in range(nb_epochs):
        for _, (x, t) in enumerate(train_data):
            x, t = x.to(device), t.to(device)
            # on calcule la sortie du modèle
            y = model(x)
            # on met à jour les poids
            # Vérification des dimensions
            if len(t.shape) > 1 and t.shape[1] > 1:  
                raise ValueError(f"Les labels `t` doivent être sous forme d'indices de classes, mais ont la forme {t.shape} t : {t}")
            loss: torch.Tensor = loss_func(y, t)

            # Rétropropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if validate_loader:
            acc, loss_val = evaluate(model=model, test_loader=validate_loader, loss_funct=loss_func)
            if print_ and epoch % 1==0:
                print(f'Epoch {epoch+1}/{nb_epochs},\033[0;34m {acc} \033[0m, Loss: {loss.item():.4f}')
        else :
            if print_ and epoch % 1==0:
                print(f'Epoch {epoch+1}/{nb_epochs},\033[0m, Loss: {loss.item():.4f}')
    return meilleur_acc, meilleur_model

def train_with_batch(model:nn.Module,
        train_loader:torch.utils.data.DataLoader,
        validate_loader:torch.utils.data.DataLoader,
        optimizer:torch.optim.SGD,
        loss_func:torch.nn.MSELoss,
        nb_epochs:int,
        threshold:float=None,
        print_:bool=False
        ) -> Tuple[float, nn.Module]:
    """
    Méthode d'entraînement du modèle

    Pour chaque époque:
    - on entraîne le modèle sur les données d'apprentissage
    - Puis on évalue le modèle sur les données de validation

    Au final nous renvoyons le modèle avec le meilleur taux de réussite
    (Nous évitons le sur-apprentissage tout en testant l'entièreté des époques)

    :param model: le modèle à entraîner
    :param train_loader: le lecteur de données d'apprentissage
    :param validate_loader: le lecteur de données de validation
    :param optimizer: l'optimiseur lié au modèle
    :param loss_func: la fonction de loss
    :param nb_epochs: le nombre d'époques
    :param print_: afficher ou non les résultats
    :return: **loss** last loss of train
    :return: **last_acc** last accuracy of model
    :return: **meilleur_acc** best accuracy of model
    :return: **meilleur_model** best model
    :return: **history_acc** list of accuracy for each epoch
    :return: **history_val_loss** list of validation loss for
    each epoch
    [...]
    """
    device = next(model.parameters()).device # get device of the model
    meilleur_acc:int = 0
    meilleur_model = None
    model.to(device)
    loss_func.to(device)
    model.train()
    history_acc = []
    history_val_loss = []

    # for x,t in train_loader:
    #     print(f"train data : {x}")
    #     print(f"train target : {t}")
    #     break

    for epoch in range(nb_epochs):
        for x,t in train_loader:
            x, t = x.to(device), t.to(device)
            # on calcule la sortie du modèle
            y = model(x)
            # on met à jour les poids
            if t.dim() > 1 and t.shape[1] > 1:  # Vérifie si t est en one-hot encoding
                t = torch.argmax(t, dim=1)  # Convertit en indices de classes

            loss:torch.Tensor = loss_func(y,t)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if validate_loader :
            acc, loss_val = evaluate(model=model, test_loader=validate_loader, loss_funct=loss_func)
            history_acc.append(acc)
            history_val_loss.append(loss_val)
            if acc > meilleur_acc:
                meilleur_acc = acc
                meilleur_model = model

            if print_:
                print(f'Epoch {epoch+1}/{nb_epochs}, Loss: {loss.item():.4f}, Accuracy: {acc}')
        else:
            if print_ and epoch % (nb_epochs // 4) == 0:
                print(f'Epoch {epoch+1}/{nb_epochs}, Loss: {loss.item():.4f}')
        
        if threshold and round(loss.item(), 3) < threshold:
            if print_:
                if validate_loader:
                    print(f'Break early Epoch {epoch+1}/{nb_epochs}, Loss: {loss.item():.4f}, Accuracy: {acc}, Validation Loss: {loss_val}')
                else:
                    print(f'Break early Epoch {epoch+1}/{nb_epochs}, Loss: {loss.item():.4f}')
            break
    if validate_loader:
        last_acc, last_loss = evaluate(model=model, test_loader=validate_loader, loss_funct=loss_func)
    else:
        last_acc = None
        last_loss =None
    return loss, last_acc, meilleur_acc, meilleur_model, history_acc, history_val_loss

def find_no_good(model: nn.Module, test_loader: torch.utils.data.DataLoader) -> tuple[float, float]:
    """
    Méthode pour trouver les x qui ne sont pas bien prédit par le modèle

    :param model: le modèle à évaluer
    :param test_loader: le lecteur de données de test
    :param loss_funct: la fonction de perte à utiliser pour l'évaluation
    :param device: le device sur lequel effectuer les calculs
    :return: une tuple contenant (le taux de réussite, la perte moyenne)
    """
    device = next(model.parameters()).device # get device of the model

    no_good = []
    model.to(device)
    model.eval()

    with torch.no_grad():  # Pas besoin de calculer les gradients en mode évaluation
        x:torch.Tensor = None; t:torch.Tensor = None
        for x, t in test_loader:
            x, t = x.to(device), t.to(device)
            # Prédictions
            y = model(x)

            if len(t.shape) > 1 and t.shape[1] > 1:  # Si les labels sont one-hot encodés
                t = torch.argmax(t, dim=1)  # On convertit en indices de classes
            for i in range(t.size(0)):
                if torch.argmax(y[i]) != t[i]:
                    no_good.append((x[i], t[i], y[i]))

    return no_good
