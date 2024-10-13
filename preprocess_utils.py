import re
import contractions
import nltk
import numpy as np
import random


def tokenize(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    text = ' '.join(lines)
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(' +', ' ', text)
    sentences = nltk.sent_tokenize(text)
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    tokens = [[token for token in sentence if any(c.isalnum() for c in token)] for sentence in tokens]  
    tokens = [sentence for sentence in tokens if len(sentence) > 5]
    tokens = [['<s>'] + sentence + ['</s>'] for sentence in tokens]
    return tokens

def split_data(tokens, train_ratio, val_ratio):
    random.shuffle(tokens)
    train_size = int(train_ratio * len(tokens))
    val_size = int(val_ratio * len(tokens))
    train_tokens = tokens[:train_size]
    val_tokens = tokens[train_size:train_size + val_size]
    test_tokens = tokens[train_size + val_size:]
    return train_tokens, val_tokens, test_tokens

def add_unks(tokens, embeddings, cut_off_freq):
    word_freq = {}
    for sentence in tokens:
        for word in sentence:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    for i in range(len(tokens)):
        for j in range(len(tokens[i])):
            if word_freq[tokens[i][j]] < cut_off_freq or tokens[i][j] not in embeddings:
                tokens[i][j] = '<unk>'
    return tokens

def create_vocab(tokens):
    word_to_idx = {}
    idx_to_word = {}
    i = 0
    for sentence in tokens:
        for word in sentence:
            if word not in word_to_idx:
                word_to_idx[word] = i
                idx_to_word[i] = word
                i += 1
    return word_to_idx, idx_to_word

def add_unks_val(val_tokens, test_tokens, word_to_idx):
    for i in range(len(val_tokens)):
        for j in range(len(val_tokens[i])):
            if val_tokens[i][j] not in word_to_idx:
                val_tokens[i][j] = '<unk>'
    for i in range(len(test_tokens)):
        for j in range(len(test_tokens[i])):
            if test_tokens[i][j] not in word_to_idx:
                test_tokens[i][j] = '<unk>'
    return val_tokens, test_tokens

def get_glove_embeddings(glove_file):
    embeddings = {}
    with open(glove_file, 'r') as file:
        for line in file:
            parts = line.split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            embeddings[word] = vector
    embeddings['<unk>'] = np.ones(100)
    embeddings['<s>'] = np.zeros(100)
    embeddings['</s>'] = np.zeros(100)
    embeddings['<pad>'] = np.zeros(100)
    return embeddings

if __name__ == '__main__':
    tokens = tokenize('Auguste_Maquet.txt')
    print(tokens[:7])
    embeddings = get_glove_embeddings()
    print(len(embeddings))
    print(embeddings['the'])







