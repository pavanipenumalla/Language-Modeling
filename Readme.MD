# Advanced NLP | Assignment 1  

This implements three different types of language models using PyTorch:

1. **Neural Network-based Language Model** (5-gram context)
2. **RNN-based Language Model** using LSTM
3. **Transformer Decoder-based Language Model**

Each model is trained and evaluated on the **Auguste_Maquet** corpus, using **GloVe 100d** pre-trained embeddings. The performance of the models is measured using perplexity scores on both training and test sets.

---

### Neural Network-based Language Model (5-gram Context)

This section implements a Neural Network-based Language Model (NNLM) using a 5-gram context to predict the next word in a sentence.

#### Key Files

- **[NNLM.py](./NNLM.py)**: Main script that trains and tests the Neural Network language model.
- **[NNLM_utils.py](./NNLM_utils.py)**: Utility functions for data preprocessing, model training, and evaluation.
- **[preprocess_utils.py](./preprocess_utils.py)**: Helper functions for tokenization, splitting data, handling unknown words, and loading GloVe embeddings.

#### Training the Model

To train the NNLM on the preprocessed data, run:

```bash
python NNLM.py
```

#### Model Evaluation

The model is evaluated on both validation and test datasets. Perplexity is computed for each sentence in the test, validation, and training sets. The perplexity scores are saved in separate text files for each dataset:

- `2021101133-LM1-train-perplexity.txt`
- `2021101133-LM1-val-perplexity.txt`
- `2021101133-LM1-test-perplexity.txt`

---

### RNN-based Language Model using LSTM

This section implements a Recurrent Neural Network (RNN) based Language Model using Long Short-Term Memory (LSTM) cells to predict the next word in a sentence.

#### Key Files

- **[LSTM.py](./LSTM.py)**: Main script that trains and tests the LSTM language model.
- **[LSTM_utils.py](./LSTM_utils.py)**: Utility functions for data preprocessing, model training, and evaluation.

#### Training the Model

To train the LSTM model run:

```bash
python LSTM.py
```

#### Model Evaluation

Train, test and validation perplexity scores are computed and saved in separate text files:

- `2021101133-LM2-train-perplexity.txt`
- `2021101133-LM2-val-perplexity.txt`
- `2021101133-LM2-test-perplexity.txt`

---

### Transformer Decoder-based Language Model

This section implements a Transformer Decoder-based Language Model to predict the next word in a sentence.

#### Key Files

- **[transformer.py](./transformer.py)**: Main script that trains and tests the Transformer Decoder language model.
- **[transformer_utils.py](./transformer_utils.py)**: Utility functions for data preprocessing, model training, and evaluation.

#### Training the Model

To train the Transformer Decoder model, run:

```bash
python transformer.py
```

#### Model Evaluation

Perplexity scores are computed for the training, validation, and test datasets and saved in separate text files:

- `2021101133-LM3-train-perplexity.txt`
- `2021101133-LM3-val-perplexity.txt`
- `2021101133-LM3-test-perplexity.txt`

---
All the saved models can be viewed in the following link: [Drive](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/aditya_pavani_students_iiit_ac_in/EguDiQnrGCtEgWqtJym-_lUBytlUr1dYZHhETzSXGslt0w?e=DnNmEa)


 

 

 
