from preprocess_utils import tokenize, get_glove_embeddings, split_data, add_unks, create_vocab, add_unks_val
import numpy as np
import torch
import torch.nn as nn
import random

class CreateDataset():
    def __init__(self,file_path,embedding_file,cut_off_freq=1):
        self.sentences = tokenize(file_path)
        self.embeddings = get_glove_embeddings(embedding_file)
        self.cut_off_freq = cut_off_freq
        self.train_sentences , self.val_sentences , self.test_sentences = split_data(self.sentences,0.7,0.1)
        self.train_sentences = add_unks(self.train_sentences,self.embeddings,self.cut_off_freq)
        self.word_to_idx , self.idx_to_word = create_vocab(self.train_sentences)
        self.val_sentences, self.test_sentences = add_unks_val(self.val_sentences,self.test_sentences,self.word_to_idx)
        
    def create_data(self,sentences):
        X = []
        Y = []
        for sentence in sentences:
            for i in range(5,len(sentence)):
                X.append([self.embeddings[word] for word in sentence[i-5:i]])
                X[-1] = np.array(X[-1]).flatten()
                Y.append(np.zeros(len(self.word_to_idx)))
                Y[-1][self.word_to_idx[sentence[i]]] = 1
        X = np.array(X)
        Y = np.array(Y)
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X,Y
    
    
class NNLM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size, activation):
        super(NNLM,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    
def train(model,train_data,val_data,num_epochs,criterion,optimizer,batch_size,device,dataobj):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        total_samples = 0
        val_loss = 0
        random.shuffle(train_data)
        for i in range(0,len(train_data),batch_size):
            optimizer.zero_grad()
            X,Y = dataobj.create_data(train_data[i:i+batch_size])
            X = X.to(device)
            Y = Y.to(device)
            output = model(X.float())
            loss = criterion(output,Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total_samples+=1
        train_loss /= total_samples
        train_losses.append(train_loss)

        model.eval()
        total_samples = 0
        with torch.no_grad():
            for i in range(0,len(val_data),batch_size):
                X,Y = dataobj.create_data(val_data[i:i+batch_size])
                X = X.to(device)
                Y = Y.to(device)
                output = model(X.float())
                loss = criterion(output,Y)
                val_loss += loss.item()
                total_samples+=1
            val_loss /= total_samples
            val_losses.append(val_loss)

        print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')
    return model,train_losses,val_losses

def test(model,test_data,criterion,batch_size,device,dataobj):
    model.eval()
    test_loss = 0
    total_samples = 0

    with torch.no_grad():
        for i in range(0,len(test_data),batch_size):
            X,Y = dataobj.create_data(test_data[i:i+batch_size])
            X = X.to(device)
            Y = Y.to(device)
            output = model(X.float())
            loss = criterion(output,Y)
            test_loss += loss.item()
            total_samples+=1
        test_loss /= total_samples
    return test_loss

def get_perplexity(model, sentences, embedding , word2idx, criterion, file_path, device):
    model = model.to(device)
    perplexity = 0
    average_perplexity = 0
    model.eval()
    
    with torch.no_grad():
        for sentence in sentences:
            X = []
            Y = []
            for i in range(5,len(sentence)):
                X.append([embedding[word] for word in sentence[i-5:i]])
                X[-1] = np.array(X[-1]).flatten()
                Y.append(np.zeros(len(word2idx)))
                Y[-1][word2idx[sentence[i]]] = 1
            X = np.array(X)
            Y = np.array(Y)
            X = torch.tensor(X)
            Y = torch.tensor(Y)
            X = X.to(device)
            Y = Y.to(device)
            output = model(X.float())
            loss = criterion(output,Y)
            perplexity = torch.exp(loss)
            average_perplexity += perplexity.item()
            sentence = sentence[1:-1]
            with open(file_path,'a') as f:
                f.write(f' '.join(sentence) + f' Perplexity: {perplexity.item()}\n')
        average_perplexity /= len(sentences)
        with open(file_path,'a') as f:
            f.write(f'Average Perplexity is {average_perplexity}\n')
    return average_perplexity



        


                
            
