from LSTM_utils import LSTMData, LSTM, train, test, get_perplexity
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import warnings
warnings.filterwarnings('ignore')

 
dataset = LSTMData('Auguste_Maquet.txt', 'glove.6B.100d.txt')
train_sentences, val_sentences, test_sentences = dataset.train_sentences, dataset.val_sentences, dataset.test_sentences
print("Train Sentences: ", len(train_sentences))
print("Validation Sentences: ", len(val_sentences))
print("Test Sentences: ", len(test_sentences))

batch_size = 32
input_dim = 100
hidden_dim = 300
output_dim = len(dataset.word_to_idx)
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTM(input_dim, hidden_dim, 1, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss().to(device)

trained_model = train(model, train_sentences, val_sentences, num_epochs, criterion, optimizer, batch_size, device, dataset)
torch.save(trained_model, 'LSTM.pt')
test_loss = test(trained_model, test_sentences,criterion, batch_size, device, dataset)
print("Test Loss: ", test_loss)

get_perplexity(trained_model, train_sentences, dataset, criterion, "2021101133-LM2-train-perplexity.txt", device)
get_perplexity(trained_model, val_sentences, dataset,criterion, "2021101133-LM2-val-perplexity.txt", device)
get_perplexity(trained_model, test_sentences, dataset , criterion, "2021101133-LM2-test-perplexity.txt", device)