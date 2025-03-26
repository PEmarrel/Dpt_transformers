import torch
import torch.nn as nn

class FeedbackPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h, _):
        if h is None:
            h = torch.zeros(1, x.size(0), 16)
        out, h = self.rnn(x, h)
        return self.fc(out[:, -1, :]), h, None

# Inspired by https://github.com/LukeDitria/pytorch_tutorials.git
    
class LSTM_Classifeur(nn.Module):
    def __init__(self, num_emb, output_size, num_layers=1, hidden_size=128, dropout=0.5):
        super(LSTM_Classifeur, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_emb = num_emb
        self.output_size = output_size

        # Create an embedding layer to convert token indices to dense vectors
        self.embedding = nn.Embedding(num_emb, hidden_size)
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        
        # Define the output fully connected layer
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden_in, mem_in):
        # Convert token indices to dense vectors
        input_embs = self.embedding(input_seq)

        # Pass the embeddings through the LSTM layer
        output, (hidden_out, mem_out) = self.lstm(input_embs, (hidden_in, mem_in))
                
        # Pass the LSTM output through the fully connected layer to get the final output
        return self.fc_out(output), hidden_out, mem_out
    
class LSTM_GenText(nn.Module):
    def __init__(self, num_emb, num_layers=1, emb_size=128, hidden_size=128, dropout=0.25):
        super(LSTM_GenText, self).__init__()
        
        self.embedding = nn.Embedding(num_emb, emb_size)

        self.mlp_emb = nn.Sequential(nn.Linear(emb_size, emb_size),
                                     nn.LayerNorm(emb_size),
                                     nn.ELU(),
                                     nn.Linear(emb_size, emb_size))
        
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, dropout=dropout)

        self.mlp_out = nn.Sequential(nn.Linear(hidden_size, hidden_size//2),
                                     nn.LayerNorm(hidden_size//2),
                                     nn.ELU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_size//2, num_emb))
        
    def forward(self, input_seq, hidden_in, mem_in):
        input_embs = self.embedding(input_seq)
        input_embs = self.mlp_emb(input_embs)
                
        output, (hidden_out, mem_out) = self.lstm(input_embs, (hidden_in, mem_in))
                
        return self.mlp_out(output), hidden_out, mem_out


def train_LSTM(model, train_loader, optimizer, loss_func, nb_epochs, validate_loader):
    device = model.fc_out.weight.device

    for epoch in range(nb_epochs):
        model.train()
        train_acc = 0
        test_acc = 0
        total_loss = 0
        steps = 0
        for i, (inputs, targets) in enumerate(train_loader):
            bs = targets.shape[0]

            # Initialize hidden and memory states
            hidden = torch.zeros(model.num_layers, bs, model.hidden_size, device=device)
            memory = torch.zeros(model.num_layers, bs, model.hidden_size, device=device)
            
            pred, _, _ = model(inputs, hidden, memory)
            
            loss = loss_func(pred[:, -1, :], targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            train_acc += (pred[:, -1, :].argmax(1) == targets).sum()
            steps += bs
            
        if validate_loader is not None:
            model.eval()
            with torch.no_grad():
                for inputs, targets in validate_loader:
                    bs = targets.shape[0]
                    hidden = torch.zeros(model.num_layers, bs, model.hidden_size, device=device)
                    memory = torch.zeros(model.num_layers, bs, model.hidden_size, device=device)
                    pred, _, _ = model(inputs, hidden, memory)
                    test_acc += (torch.argmax(pred[:, -1, :], dim=1) == targets).sum().item()
            test_acc /= len(validate_loader.dataset)
            
    return train_acc, test_acc
