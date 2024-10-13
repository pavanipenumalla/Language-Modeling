from NNLM_utils import CreateDataset
import torch
import torch.nn as nn
import numpy as np
import random


class LSTMData(CreateDataset):
    def __init__(self, file_path, embedding_file, cut_off_freq=1):
        super().__init__(file_path, embedding_file, cut_off_freq)
        self.word_to_idx['<pad>'] = len(self.word_to_idx)

    def create_data(self, sentences):
        X = []
        Y = []
        pad_len = max([len(sentence) for sentence in sentences])
        for i in range(len(sentences)):
            sentences[i] = sentences[i] + ['<pad>'] * (pad_len - len(sentences[i]))

        for sentence in sentences:
            X.append([self.embeddings[word] for word in sentence[:-1]])     # X is the embedding of the first n-1 words
            y = [self.word_to_idx[word] for word in sentence[1:]]           # y is the index of the next word
            y_one_hot = torch.zeros(len(y), len(self.word_to_idx))          # y_one_hot is the one-hot encoding of y
            for i in range(len(y)):                                                        
                y_one_hot[i][y[i]] = 1 
            Y.append(y_one_hot)

        X = np.array(X)
        Y = np.array(Y)
        X = torch.tensor(X).float()
        Y = torch.tensor(Y).float()
        return X, Y
    
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Set input_dim of fc to hidden_dim

    def forward(self, x):
        h_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        c_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)

        output, (_, _) = self.lstm(x, (h_0, c_0))
        output = self.fc(output)  # Apply FC layer to the LSTM output
        return output

    
def train(model, train_data, val_data, num_epochs, criterion, optimizer, batch_size, device, dataobj):
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        total_samples = 0
        val_loss = 0
        random.shuffle(train_data)
        for i in range(0, len(train_data), batch_size):
            optimizer.zero_grad()
            X, Y = dataobj.create_data(train_data[i:i+batch_size])
            X = X.to(device)
            Y = Y.to(device)
            output = model(X)
            output = output.view(-1, output.shape[2])
            Y = Y.view(-1, Y.shape[2])
            loss = criterion(output, Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total_samples+=1
        
        train_loss/=total_samples
        total_samples = 0

        model.eval()
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                X, Y = dataobj.create_data(val_data[i:i+batch_size])
                X = X.to(device)
                Y = Y.to(device)
                output = model(X)
                output = output.view(-1, output.shape[2])
                Y = Y.view(-1, Y.shape[2])
                loss = criterion(output, Y)
                val_loss += loss.item()
                total_samples+=1
            val_loss/=total_samples
        print(f'Epoch: {epoch+1}, Training Loss: {train_loss}, Validation Loss: {val_loss}')

    return model
        

def test(model, test_data, criterion, batch_size, device, dataobj):
    test_loss = 0
    model = model.to(device)
    model.eval()
    total_samples = 0

    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            X, Y = dataobj.create_data(test_data[i:i+batch_size])
            X = X.to(device)
            Y = Y.to(device)
            output = model(X)
            output = output.view(-1, output.shape[2])
            Y = Y.view(-1, Y.shape[2])
            loss = criterion(output, Y)
            test_loss += loss.item()
            total_samples+=1
        test_loss /= total_samples
    return test_loss
 
def get_perplexity(model, sentences, dataobj, criterion, file_path, device):
    model = model.to(device)
    perplexity = 0
    average_perplexity = 0
    model.eval()

    with torch.no_grad():
        for sentence in sentences:
            X, Y = dataobj.create_data([sentence])
            X = X.to(device)
            Y = Y.to(device)
            output = model(X)
            output = output.view(-1, output.shape[2])
            Y = Y.view(-1, Y.shape[2])
            loss = criterion(output, Y)
            perplexity = torch.exp(loss)
            average_perplexity += perplexity.item()
            with open(file_path, 'a') as f:
                f.write(' '.join(sentence) + ' ' + str(perplexity.item()) + '\n')
        average_perplexity /= len(sentences)
        with open(file_path, 'a') as f:
            f.write(f'Average Perplexity: {average_perplexity}\n')
    return average_perplexity
     
        
    




