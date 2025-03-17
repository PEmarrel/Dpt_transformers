from environnement.Env import Environnement
from environnement.GridWord import gridWord
from environnement.SmallLoop import SmallLoop

from agent.AgentLstmV1 import AgentLstmV1
from agent.interactions.SimpleInteraction import SimpleInteraction as inter
from agent.model.RNN import LSTM
from agent.tools.plot import *

import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

env = gridWord(x=1, y=1, theta=0, world=np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 0, 1],
        [1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
    ]))

valence = {
    inter('forward', 'empty') : 10,
    inter('forward', 'wall') : -100,
    inter('turn_left', 'empty') : -41,
    inter('turn_left', 'wall') : -100,
    inter('turn_right', 'empty') : -41,
    inter('turn_right', 'wall') : -100,
    inter('feel_front', 'wall') : -25,
    inter('feel_front', 'empty') : -22,
    inter('feel_right', 'wall') : -25,
    inter('feel_right', 'empty') : -22,
    inter('feel_left', 'wall') : -25,
    inter('feel_left', 'empty') : -22,
}

hidden_size = 16
num_layers = 1
len_vocab = len(env.get_outcomes() + env.get_actions())

# Create the LSTM classifier model
lstm_classifier = LSTM(num_emb=len_vocab, output_size=2,
                       num_layers=num_layers, hidden_size=hidden_size, dropout=0.5).to(device)

optimizer = torch.optim.Adam(lstm_classifier.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

agent = AgentLstmV1(valence=valence, model=lstm_classifier, optimizer=optimizer, loss_fn=loss_func,
    gap=7, maxDepth=5, seuil=0.3, nb_epochs=20,
    data_val=None)

outcome = env.outcome(agent.action_choice)

history_good = []
history_good_inter = []
history_bad_inter = []
hisrory_val = []
pourcent_by_10  = []
by_10_good_inter  = []
by_10_bad_inter  = []
mean_val = []

time_start = time.time()
for i in tqdm(range(100)):
    action, predi = agent.action(outcome, verbose=True)
    outcome = env.outcome(action)
    history_good.append(outcome == predi)
    history_good_inter.append((action == 'forward' and outcome == 'empty'))
    history_bad_inter.append((action == 'forward' and outcome == 'wall'))
    hisrory_val.append(valence[inter(action, outcome)])
    pourcent_by_10.append(sum(history_good[-10:]) if len(history_good) >= 10 else 0)
    by_10_good_inter.append(sum(history_good_inter[-10:]) if len(history_good_inter) >= 10 else 0)
    by_10_bad_inter.append(sum(history_bad_inter[-10:]) if len(history_bad_inter) >= 10 else 0)
    mean_val.append(np.mean(hisrory_val[-10:]) / 10 if len(hisrory_val) >= 10 else 0)
    env.save_world(path="../imgToGif")
time_end = time.time()
print("time : ", time_end - time_start)
    
save_monitor_life_agent(agent.predictExplor, by_10_bad_inter, by_10_good_inter, pourcent_by_10, mean_val, path="test02.png")
save_time_compute(agent.time_train, path="../test01/time_compute.png")
save_evolued_loss(agent.loss_train, path="../test01/evolued_loss.png")
save_evolued_acc(agent.acc_train, path="../test01/evolued_acc.png")
