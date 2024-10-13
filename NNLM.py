import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from NNLM_utils import CreateDataset,NNLM,train,test,get_perplexity
import warnings
warnings.filterwarnings("ignore")

dataset = CreateDataset('Auguste_Maquet.txt', 'glove.6B.100d.txt')
train_sentences, val_sentences, test_sentences = dataset.train_sentences, dataset.val_sentences, dataset.test_sentences

batch_size = 64
input_dim = 500
hidden_dim = 300
output_dim = len(dataset.word_to_idx)
num_epochs = 7
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

activation = nn.Tanh()
model = NNLM(input_dim, hidden_dim, output_dim, activation).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss().to(device)

trained_model,train_losses,val_losses = train(model, train_sentences, val_sentences, num_epochs, criterion, optimizer,  batch_size, device, dataset)

torch.save(trained_model, 'NNLM.pt')
# loading the trained model
# trained_model = torch.load('NNLM.pt')

test_loss = test(trained_model, test_sentences, criterion,batch_size, device, dataset)
print(f'Test Loss: {test_loss}')

get_perplexity(trained_model, test_sentences, dataset.embeddings, dataset.word_to_idx, criterion, "2021101133-LM1-test-perplexity.txt", device)
get_perplexity(trained_model, val_sentences, dataset.embeddings, dataset.word_to_idx, criterion, "2021101133-LM1-val-perplexity.txt", device)
get_perplexity(trained_model, train_sentences, dataset.embeddings, dataset.word_to_idx, criterion, "2021101133-LM1-train-perplexity.txt", device)


