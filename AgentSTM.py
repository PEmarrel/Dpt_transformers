import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import time
from tqdm.notebook import trange, tqdm

from inter.simpleInteraction import simpleInteraction as inter
from model.Tokenizer import SimpleTokenizerV1
from model.CustomDataSet import *
from outil import subfinder
from environnement.environnement import Environnement as env

class AgentLSTM:
    def __init__(self, valence:dict[inter], model:nn.Module, maxDepth:int,
                 seuil:float, optimizer, loss_fn, gap:int=11, nb_epochs:int=50, 
                 data_val:tuple=None, device="cpu"):
        """
        Create an agent with a LSTM model and spesific decision making.
        data_val is composed by list of all actions and outcomes. And 
        is not useful to train the model. It's just to have a monitoring
        of the model. \n
        valence: dict of interactions, is use to know what is a good 
        comportment \n
        model: the model to train, this agent was create for LSTM model 
        \n
        
        """
        self.model:nn.Module = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.valence:dict[inter] = valence
        self.maxDepth:int = maxDepth
        self.seuil:float = seuil
        self.gap:int = gap
        self.nbEpochs:int = nb_epochs
        self.force_fit:bool = False
        self.device = device
        
        self.seq_to_exe = [] # Sequence choice by Decide
        self.history_act = [] # History of all actions
        self.history_fb = [] # History of all feedback
        self.history_inter = [] # History of all interactions
        
        self.all_outcomes = set()
        self.all_act = set()
        key:inter = None
        for key in valence.keys():
            self.all_outcomes.add(key.getOutcome())
            self.all_act.add(key.getAction())
            
        self.all_outcomes = list(self.all_outcomes)
        self.all_act = list(self.all_act)
        
        self.tokenizer = SimpleTokenizerV1(
            vocab={key: i for i, key in enumerate(self.all_outcomes + self.all_act)})
        
        self.seq_explo = []
        self.valence_explo = -np.inf
        
        self.action_choice = self.all_act[0] # Default action, because developpemental start with action
        self.history_act.append(self.action_choice)
        self.outcome_prediction = None
        
        # Variable to monitor the model
        self.loss_train:list = []
        self.acc_train:list = []
        self.loss_test:list = []
        self.acc_test:list = []
        self.time_train:list = []
        self.time_expected_val:list = []
        self.time_train:list = []
        self.time_expected_val:list = []
        self.predictExplor:list = []
        if data_val is not None:
            dataset_test = CustomDataSetRNN(actions=data_val[0], outcomes=data_val[1], context_lenght=self.gap, 
                                    dim_out=len(self.all_outcomes), tokenizer=self.tokenizer)
            self.data_loader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)
        else:
            self.data_loader_test = None
        number_patern = 200000
        self.prealloc_df = pd.DataFrame(np.empty((number_patern, 5)), 
                                    columns=["proposition", "valence", "action", "probability", "val_sucess"])
        self.prealloc_df = self.prealloc_df.astype({"proposition": "U20", "valence": float, "action": "U20", "probability": float, "val_sucess": float})
        self.current_index = 0
        self.visu_explo = pd.DataFrame(np.empty((number_patern, 2)), columns=["seqence", "valence"])
        self.visu_explo = self.visu_explo.astype({"seqence": "U20", "valence": float})
        self.current_index_explo = 0
        
        if data_val is not None:
            self.visu_val = pd.DataFrame(np.empty((len(data_val[0]), 3)), 
                                        columns=["seqence", "probablility", "good"])
            self.visu_val = self.visu_val.astype({"seqence": "U20", "probablility": float, "good": bool})
            self.current_index_val = 0
        
    def fit(self):
        """
        train model
        """
        dataset = CustomDataSetRNN(actions=self.history_act, outcomes=self.history_fb, 
                                 context_lenght=self.gap, dim_out=len(self.all_outcomes),
                                 tokenizer=self.tokenizer)
        
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        time_train = time.time()
        
        for i in range(self.nbEpochs):
            self.model.train()
            steps = 0
            train_acc = 0
            training_loss = []
            for tmp, (x,t) in enumerate(data_loader):
                x = x.to(self.device)
                t = t.to(self.device)
                bs = t.shape[0]
                h = torch.zeros(self.model.num_layers, bs, self.model.hidden_size, device=self.device)
                cell = torch.zeros(self.model.num_layers, bs, self.model.hidden_size, device=self.device)

                pred, h, cell = self.model(x, h, cell)

                loss = self.loss_fn(pred[:, -1, :], t)
                training_loss.append(loss.item())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_acc += sum((pred[:, -1, :].argmax(1) == t).cpu().numpy())
                steps += bs
            
            self.acc_train.append(train_acc / steps)
            if self.data_loader_test is not None:
                self.model.eval()
                steps = 0
                test_acc = 0
                loss_test = []
                
                for text, label in self.data_loader_test:
                    text = text.to(self.device)
                    label = label.to(self.device)
                    bs = label.shape[0]

                    # Initialize hidden and memory states
                    hidden = torch.zeros(self.model.num_layers, bs, self.model.hidden_size, device=self.device)
                    memory = torch.zeros(self.model.num_layers, bs, self.model.hidden_size, device=self.device)
                    
                    # Forward pass through the model
                    pred, hidden, memory = self.model(text, hidden, memory)
                    
                    for i in range(bs):
                        self.visu_val.iloc[steps + i] = [str(self.tokenizer.decode(text[i].cpu().tolist())), 
                                                         float(torch.nn.functional.softmax(pred[i, -1, :], dim=-1).max().item()), 
                                                         int(pred[i, -1, :].argmax().item() == label[i])]

                    # Calculate the loss
                    loss = self.loss_fn(pred[:, -1, :], label)
                    loss_test.append(loss.item())

                    # Calculate test accuracy
                    test_acc +=  sum((pred[:, -1, :].argmax(1) == label).cpu().numpy())
                    steps += bs
                    
                loss_test.append(loss_test)
                self.acc_test.append(test_acc / steps)
                # print(f"Validation time: {time.time() - time_val_epoch}")    
            self.loss_train.append(training_loss)
            # If acc is 100% we stop the training
            if self.acc_train[-1] >= 0.99:
                for _ in range(i, self.nbEpochs):
                    self.acc_train.append(self.acc_train[-1])
                    if self.data_loader_test is not None:
                        self.loss_test.append(self.loss_test[-1])
                break
            
        print(f"Training time: {time.time() - time_train}")
        self.time_train.append(time.time() - time_train)
        
    def predict(self, action):
        """
        Predict the feedback of the action, use the last gap actions/outcomes
        """        
        x = []
        for i in range(-(self.gap - 1) // 2, 0, 1):
            x.append(self.history_act[i])
            x.append(self.history_fb[i])
        x.append(action)
        seq_to_pred = self.tokenizer.encode(x)
        # On simule un batch de taille 1
        seq_to_pred = torch.tensor([seq_to_pred], device=self.device)
        h = torch.zeros(self.model.num_layers, 1, self.model.hidden_size, device=self.device)
        cell = torch.zeros(self.model.num_layers, 1, self.model.hidden_size, device=self.device)
        probs, _, _ = self.model(seq_to_pred, h, cell)
        
        pred_feedback = torch.argmax(probs[:, -1, :]).item()
        pred_feedback = self.tokenizer.decode(pred_feedback)
        
        return pred_feedback
    
    def fill_valance_explo(self, max_depth:int, seq_predi:list, valence_succes_pred:float):
        max_depth -= 1
        inter_max, value = max(self.valence.items(), key=lambda y: y[1])
        for _ in range(max_depth):
            seq_predi += [inter_max.getAction(), inter_max.getOutcome()]
            valence_succes_pred += value
        self.visu_explo.iloc[self.current_index_explo] = [str(seq_predi), valence_succes_pred]
        self.current_index_explo += 1
        if valence_succes_pred > self.valence_explo:
            self.seq_explo = seq_predi
            self.valence_explo = valence_succes_pred
    
    def recursif_expective_valance(self, context:list, max_depth:int, seuil:float=0.5, proba:float = 1,
                                    seq_predi:list = [], valence_pred:float = 0, valence_succes_pred:float = 0):
        """
        Create the list of proposed sequences
        """
        max_depth -= 1
        self.model.eval()
        
        for act in self.all_act:
            new_seq = seq_predi + [act]
            seq_to_predict = context + [self.tokenizer.encode(act)]
            sub_list = subfinder(self.history_inter, seq_to_predict)
            
            if sub_list == []:
                inter_max, value = max([(inter(act, out), 
                    self.valence[inter(act, out)]) for out in self.all_outcomes], 
                    key=lambda y: y[1])
            
                new_seq += [inter_max.getOutcome()]
                tmp_value = valence_succes_pred + value
                self.visu_explo.iloc[self.current_index_explo] = [str(new_seq), valence_succes_pred]
                self.current_index_explo += 1
                if tmp_value > self.valence_explo:
                    self.seq_explo = new_seq
                    self.valence_explo = valence_succes_pred
                    
                self.fill_valance_explo(max_depth, new_seq, tmp_value)
                
                continue
            
            seq_to_predict = torch.tensor([seq_to_predict], dtype=torch.int).to(self.device)
            
            hidden = torch.zeros(self.model.num_layers, 1, self.model.hidden_size, device=self.device)
            memory = torch.zeros(self.model.num_layers, 1, self.model.hidden_size, device=self.device)
            
            x, _, _ = self.model(seq_to_predict, hidden, memory)
            x = x[0, -1, :]
            probs = torch.nn.functional.softmax(x, dim=0).tolist()
            
            expected_valence = valence_pred
            for i, out in enumerate(self.all_outcomes):
                tmp_proba = probs[i] * proba
                expected_valence += float(np.round(self.valence[inter(act, out)] * tmp_proba, decimals=4))
                
            for i, out in enumerate(self.all_outcomes):
                visu_val = None
                tmp_new_seq = new_seq + [out]
                tmp_proba = probs[i] * proba
                # If the probability is above a threshold
                sucess_valence = self.valence[inter(act, out)] + valence_succes_pred
                if tmp_proba > seuil:
                    visu_val = expected_valence
                    # If the max_depth is not reached 
                    if max_depth > 0: 
                        # Recursively look for longer sequences
                        new_context = context + self.tokenizer.encode([act, out])
                        self.recursif_expective_valance(context=new_context[2:], max_depth=max_depth, seuil=seuil, 
                            proba=tmp_proba, seq_predi=tmp_new_seq.copy(), valence_pred=expected_valence, valence_succes_pred=sucess_valence)
                    else:
                        self.prealloc_df.iloc[self.current_index] = [str(tmp_new_seq), visu_val, tmp_new_seq[0], tmp_proba, sucess_valence]
                        self.current_index += 1
                        
    def expective_valance(self, verbose:bool=False):
        """
        Permet de calculer l'expective valance d'une séquence d'interaction

        Args:
            max_depth (int): _description_
            seuil (float, optional): _description_. Defaults to 0.2.
            verbose (bool, optional): _description_. Defaults to False.
        """
        
        x = []
        for i in range(-(self.gap - 1) // 2, 0, 1):
            x.append(self.history_act[i])
            x.append(self.history_fb[i])
        seq_to_pred = self.tokenizer.encode(x)
        self.prealloc_df[:] = np.empty((len(self.prealloc_df), 5))
        self.prealloc_df["valence"] = -np.inf
        self.current_index = 0
        self.seq_explo = []
        self.valence_explo = -np.inf
        self.visu_explo[:] = np.empty((len(self.visu_explo), 2))
        self.current_index_explo = 0
        return self.recursif_expective_valance(context=seq_to_pred,
                                            max_depth=self.maxDepth,
                                            proba=1, seq_predi=[],
                                            seuil=self.seuil)
        
    def decide(self):
        if self.seq_to_exe and len(self.seq_to_exe) > 1:
            out = self.seq_to_exe.pop(0)
            if out == self.history_fb[-1]:
                self.predictExplor.append(self.predictExplor[-1])
                act = self.seq_to_exe.pop(0)
                return act
            else:
                self.force_fit = True
        self.seq_to_exe = []        
        
        time_compute_expective_val = time.time()
        self.expective_valance()
        print(f"Time to compute expective valance: {time.time() - time_compute_expective_val}")
        self.time_expected_val.append(time.time() - time_compute_expective_val)
        # Keep row with probability between 0.4 and 0.6
        # compute_df = self.prealloc_df[(self.prealloc_df["probability"] > 0.3) & (self.prealloc_df["probability"] < 0.7)]
        # if len(compute_df) == 0:
        self.seq_to_exe = self.prealloc_df.sort_values(by="valence", ascending=False).iloc[0].proposition
        expected_val = self.prealloc_df.sort_values(by="valence", ascending=False).iloc[0].valence
        # else:
            # print("renforce ...")
            # self.seq_to_exe = compute_df.sort_values(by="val_sucess", ascending=False).iloc[0].proposition
            # self.force_fit = True
        print(f"expected valence : {expected_val:.2f} valence explo : {self.valence_explo:.2f} model predict : {self.seq_to_exe}, explo want : {self.seq_explo}")
        if self.seq_to_exe is not None and expected_val > self.valence_explo:
            self.seq_to_exe = eval(self.seq_to_exe)
            print("\033[0;35m after compute ... \033[0m", self.seq_to_exe)
            self.predictExplor.append(1)
        else:
            print("\033[0;36m explo ... \033[0m", self.seq_explo)
            self.seq_to_exe = self.seq_explo
            self.force_fit = True
            self.predictExplor.append(0)
        act = self.seq_to_exe.pop(0)
        return act
    
    def action(self, real_outcome, verbose=False):
        """
        La fonction action permet à l'agent de choisir une action en fonction de l'outcome réel.
        Cette fonction entraine le modèle a prévoir les outcomes futurs en fonction des actions passées.

        Args:
            real_outcome : L'outcome réel suite à l'action de l'agent
            verbose : Affiche les informations sur l'entrainement ou non
        """
        # La première étape est de sauvegarder l'outcome réel
        self.history_fb.append(real_outcome)
        self.history_inter.append(self.tokenizer.encode(real_outcome))
        good_pred:bool = self.outcome_prediction == real_outcome
        if verbose :
            print(f"\033[0;31m Action: {self.action_choice} \033[0m, Prediction: {self.outcome_prediction}, Outcome: {real_outcome}, \033[0;31m Satisfaction: {good_pred} \033[0m")
        
        # Ensuite nous regardons si nous devons entrainer le modèle
        # not(explore) and 
        if (not(good_pred) or self.force_fit)and len(self.history_fb) + len(self.history_fb) > self.gap:
            self.fit()
            self.force_fit = False
            
        elif len(self.history_fb) + len(self.history_fb) > self.gap:
            for _ in range(self.nbEpochs):
                self.acc_train.append(self.acc_train[-1])
                if self.data_loader_test is not None:
                    self.loss_test.append(self.loss_test[-1])

        # Nous devons maintenant choisir une action
        if len(self.history_fb) + len(self.history_fb) > self.gap:
            self.action_choice = self.decide()
            self.outcome_prediction = self.predict(self.action_choice)
        else:
            inter_max, value = max(self.valence.items(), key=lambda y: y[1])
            self.action_choice = inter_max.getAction()
        # self.action_choice = np.random.choice(self.all_act)
        self.history_act.append(self.action_choice)
        self.history_inter.append(self.tokenizer.encode(self.action_choice))
        
        return self.action_choice, self.outcome_prediction
              
def runAgentLSTM(environenment:env, agent:AgentLSTM, path ="imgToGif",  verbose:bool=False):
    history_good_pred = []
    hisrory_val = []
    history_good_inter = []
    history_bad_inter = []
    info = {
        "pourcent_by_10": [],
        "by_10_good_inter": [],
        "by_10_bad_inter": [],
        "mean_val": []
    }
    for i in tqdm(range(100)):
        action, predi = agent.action(outcome, verbose=True)
        outcome = environenment.outcome(action)
        history_good_pred.append(outcome == predi)
        info["history_good_inter"].append((action == 'forward' and outcome == 'empty'))
        info["history_bad_inter"].append((action == 'forward' and outcome == 'wall'))
        hisrory_val.append(agent.valence[inter(action, outcome)])
        info["pourcent_by_10"].append(sum(history_good_pred[-10:]) if len(history_good_pred) >= 10 else 0)
        info["by_10_good_inter"].append(sum(history_good_inter[-10:]) if len(history_good_inter) >= 10 else 0)
        info["by_10_bad_inter"].append(sum(history_bad_inter[-10:]) if len(history_bad_inter) >= 10 else 0)
        info["mean_val"].append(np.mean(hisrory_val[-10:]) / 10 if len(hisrory_val) >= 10 else 0)
        environenment.save_world(path=path)
        
    return info
