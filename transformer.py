from transformer_utils import TransformerData, Transformer_Decoder, train_decoder, test_decoder, get_perplexity
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import warnings
warnings.filterwarnings('ignore')

dataset = TransformerData('Auguste_Maquet.txt', 'glove.6B.100d.txt')
train_sentences, val_sentences, test_sentences = dataset.train_sentences, dataset.val_sentences, dataset.test_sentences
print("Train Sentences: ", len(train_sentences))
print("Validation Sentences: ", len(val_sentences))
print("Test Sentences: ", len(test_sentences))

batch_size = 32
input_dim = 100
output_dim = len(dataset.word_to_idx)
num_epochs = 5
n_heads = 4
ff_dim = 300
n_layers = 1
dropout = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Transformer_Decoder(input_dim, output_dim, dataset.max_len, n_heads, ff_dim, n_layers, dropout)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss().to(device)

trained_model = train_decoder(model, train_sentences, val_sentences, num_epochs, criterion, optimizer, batch_size, device, dataset)
torch.save(trained_model, 'Transformer.pt')

test_loss = test_decoder(trained_model, test_sentences, criterion, batch_size, device, dataset)
print("Test Loss: ", test_loss)

get_perplexity(trained_model, train_sentences, dataset, criterion, "2021101133-LM3-train-perplexity.txt", device)
get_perplexity(trained_model, val_sentences, dataset, criterion, "2021101133-LM3-val-perplexity.txt", device)
get_perplexity(trained_model, test_sentences, dataset, criterion, "2021101133-LM3-test-perplexity.txt", device)
