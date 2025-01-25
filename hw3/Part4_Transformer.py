#!/usr/bin/env python
# coding: utf-8

# $$
# \newcommand{\mat}[1]{\boldsymbol {#1}}
# \newcommand{\mattr}[1]{\boldsymbol {#1}^\top}
# \newcommand{\matinv}[1]{\boldsymbol {#1}^{-1}}
# \newcommand{\vec}[1]{\boldsymbol {#1}}
# \newcommand{\vectr}[1]{\boldsymbol {#1}^\top}
# \newcommand{\rvar}[1]{\mathrm {#1}}
# \newcommand{\rvec}[1]{\boldsymbol{\mathrm{#1}}}
# \newcommand{\diag}{\mathop{\mathrm {diag}}}
# \newcommand{\set}[1]{\mathbb {#1}}
# \newcommand{\norm}[1]{\left\lVert#1\right\rVert}
# \newcommand{\pderiv}[2]{\frac{\partial #1}{\partial #2}}
# \newcommand{\bb}[1]{\boldsymbol{#1}}
# $$
# # Part 3: Transformer
# <a id=part3></a>

# In this part we will implement a variation of the attention mechanism named the 'sliding window attention'. Next, we will create a transformer encoder with the sliding-window attention implementation, and we will train the encoder for sentiment analysis.

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
import unittest
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch.optim as optim
from tqdm import tqdm
import os


# In[2]:


test = unittest.TestCase()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using device:', device)


# ## Reminder: scaled dot product attention
# <a id=part3_1></a>

# In class, you saw that the scaled dot product attention is defined as:
# 
# $$
# \begin{align}
# \mat{B} &= \frac{1}{\sqrt{d}} \mat{Q}\mattr{K}  \ \in\set{R}^{m\times n} \\
# \mat{A} &= softmax({\mat{B}},{\mathrm{dim}=1}), \in\set{R}^{m\times n} \\
# \mat{Y} &= \mat{A}\mat{V} \ \in\set{R}^{m\times d_v}.
# \end{align}
# $$
# 
# where `K`,`Q` and `V` for the self attention came as projections of the same input sequnce
# 
# $$
# \begin{align*}
# \vec{q}_{i} &= \mat{W}_{xq}\vec{x}_{i} &
# \vec{k}_{i} &= \mat{W}_{xk}\vec{x}_{i} &
# \vec{v}_{i} &= \mat{W}_{xv}\vec{x}_{i} 
# \end{align*}
# $$
# 
# If you feel the attention mechanism doesn't quite sit right, we recommend you go over lecture and tutorial notes before proceeding. 
# 
# We are now going to introduce a slight variation of the scaled dot product attention.

# ## Sliding window attention
# <a id=part3_2></a>

# The scaled dot product attention computes the dot product between **every** pair of key and query vectors. Therefore, the computation complexity is $O(n^2)$ where $n$ is the sequence length.
# 
# In order to obtain a computational complexity that grows linearly with the sequnce length, the authors of 'Longformer: The Long-Document Transformer https://arxiv.org/pdf/2004.05150.pdf' proposed the 'sliding window attention' which is a variation of the scaled dot product attention. 
# 
# In this variation, instead of computing the dot product for every pair of key and query vectors, the dot product is only computed for keys that are in a certain 'window' around the query vector. 
# 
# For example, if the keys and queries are embeddings of words in the sentence "CS is more prestigious than EE", and the window size is 2, then for the query corresponding to the word 'is' we will only compute a dot product with the keys that are at most ${window\_size}\over{2}$$ = $${2}\over{2}$$=1$ to the left and to the right. Meaning the keys that correspond to the workds 'CS', 'is' and 'more'.
# 
# Formally, the intermediate calculation of the normalized dot product can be written as: 
# 
# $$
# \mathrm{b}(q, k, w) 
# =
# \begin{cases}
#     q⋅k^T\over{\sqrt{d_k}} & \mathrm{if} \;d(q,k) ≤ {{w}\over{2}} \\
#     -\infty & \mathrm{otherwise}
# \end{cases}.
# $$
# 
# Where $b(\cdot,\cdot,\cdot)$ is the intermediate result function (used to construct a matrix $\mat{B}$ on which we perform the softmax), $q$ is the query vector, $k$ is the key vector, $w$ is the sliding window size, and $d(\cdot,\cdot)$ is the distance function between the positions of the tokens corresponding to the key and query vectors.
# 
# **Note**: The distance function $d(\cdot,\cdot)$ is **Not** cyclical. Meaning that that in the example above when searching for the words at distance 1 from the word 'CS', we **don't** return cyclically from the right and count the word EE.
# 
# The result of this operation can be visualized like this: (green corresponds to computing the scaled dot product, and white to a no-op or $-∞$).
# 
# <img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-31_at_7.27.29_PM.png" width="400"/>
# 
# 
# 
# 
# 

# **TODO**: Implement the sliding_window_attention function in hw3/transformer.py

# In[3]:


from hw3.transformer import sliding_window_attention


## test sliding-window attention
num_heads = 3
batch_size = 2
seq_len = 5
embed_dim = 3
window_size = 2

## test without extra dimension for heads
x = torch.arange(seq_len*embed_dim).reshape(seq_len,embed_dim).repeat(batch_size,1).reshape(batch_size, seq_len, -1).float()

values, attention = sliding_window_attention(x, x, x,window_size)

gt_values = torch.load(os.path.join('test_tensors','values_tensor_0_heads.pt'))


test.assertTrue(torch.all(values == gt_values), f'the tensors differ in dims [B,row,col]:{torch.stack(torch.where(values != gt_values),dim=0)}')

gt_attention = torch.load(os.path.join('test_tensors','attention_tensor_0_heads.pt'))
test.assertTrue(torch.all(attention == gt_attention), f'the tensors differ in dims [B,row,col]:{torch.stack(torch.where(attention != gt_attention),dim=0)}')


## test with extra dimension for heads
x = torch.arange(seq_len*embed_dim).reshape(seq_len,embed_dim).repeat(batch_size, num_heads, 1).reshape(batch_size, num_heads, seq_len, -1).float()

values, attention = sliding_window_attention(x, x, x,window_size)

gt_values = torch.load(os.path.join('test_tensors','values_tensor_3_heads.pt'))
test.assertTrue(torch.all(values == gt_values), f'the tensors differ in dims [B,num_heads,row,col]:{torch.stack(torch.where(values != gt_values),dim=0)}')


gt_attention = torch.load(os.path.join('test_tensors','attention_tensor_3_heads.pt'))
test.assertTrue(torch.all(attention == gt_attention), f'the tensors differ in dims [B,num_heads,row,col]:{torch.stack(torch.where(attention != gt_attention),dim=0)}')


# ## Multihead Sliding window attention
# <a id=part3_2></a>

# As you've seen in class, the transformer model uses a Multi-head attention module. We will use the same implementation you've seen in the tutorial, aside from the attention mechanism itslef, which will be swapped with the sliding-window attention you implemented.
# 

# **TODO**: Insert the call to the sliding-window attention mechanism in the forward of MultiHeadAttention in hw3/transformer.py 

# ## Sentiment analysis
# <a id=part3_3></a>

# We will now go on to tackling the task of sentiment analysis which is the process of analyzing text to determine if the emotional tone of the message is positive or negative (many times a neutral class is also used, but this won't be the case in the data we will be working with).
# 
# 
# 
# 

# ### IMBD hugging face dataset
# <a id=part3_3_1></a>

# Hugging Face is a popular open-source library and platform that provides state-of-the-art tools and resources for natural language processing (NLP) tasks. It has gained immense popularity within the NLP community due to its user-friendly interfaces, powerful pre-trained models, and a vibrant community that actively contributes to its development. 
# 
# Hugging Face provides a wide array of tools and utilities, which we will leverage as well. The Hugging Face Transformers library, built on top of PyTorch and TensorFlow, offers a simple yet powerful API for working with Transformer-based models (such as Distil-BERT). It enables users to easily load, fine-tune, and evaluate models, as well as generate text using these models.
# 
# Furthermore, Hugging Face offers the Hugging Face Datasets library, which provides access to a vast collection of publicly available datasets for NLP. These datasets can be conveniently downloaded and used for training and evaluation purposes.
# 
# You are encouraged to visit their site and see other uses: https://huggingface.co/

# In[4]:


import numpy as np
import pandas as pd
import sys
import pathlib
import urllib
import shutil
import re

import matplotlib.pyplot as plt

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[5]:


from datasets import DatasetDict
from datasets import load_dataset, concatenate_datasets


# First, we load the dataset using Hugging Face's `datasets` library.
# 
# Feel free to look around at the full array of datasets that they offer.
# 
# https://huggingface.co/docs/datasets/index
# 
# We will load the full training and test sets in addition to a small toy subset of the training set.
# 

# In[6]:


dataset = load_dataset('imdb', split=['train', 'test', 'train[12480:12520]'])


# In[7]:


print(dataset)


# We see that it returned a list of 3 labeled datasets, the first two of size 25,000, and the third of size 40.
# We will use these as `train` and `test` datasets for training the model, and the `toy` dataset for a sanity check. 
# These Datasets are wrapped in a `Dataset` class.
# 
# We now wrap the dataset into a `DatasetDict` class, which contains helpful methods to use for working with the data.   
# https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.DatasetDict

# In[8]:


#wrap it in a DatasetDict to enable methods such as map and format
dataset = DatasetDict({'train': dataset[0], 'val': dataset[1], 'toy': dataset[2]})


# In[9]:


dataset


# We can now access the datasets in the Dict as we would a dictionary.
# Let's print a few training samples

# In[10]:


print(dataset['train'])

for i in range(4):
    print(f'TRAINING SAMPLE {i}:') 
    print(dataset['train'][i]['text'])
    label = dataset['train'][i]['label']
    print(f'Label {i}: {label}')
    print('\n')


# We should check the label distirbution:

# In[11]:


def label_cnt(type):
    ds = dataset[type]
    size = len(ds)
    cnt= 0 
    for smp in ds:
        cnt += smp['label']
    print(f'negative samples in {type} dataset: {size - cnt}')
    print(f'positive samples in {type} dataset: {cnt}')
    
label_cnt('train')
label_cnt('val')
label_cnt('toy')


# ### __Import the tokenizer for the dataset__

# Let’s tokenize the texts into individual word tokens using the tokenizer implementation inherited from the pre-trained model class.  
# With Hugging Face you will always find a tokenizer associated with each model. If you are not doing research or experiments on tokenizers it’s always preferable to use the standard tokenizers.  
# 
# 

# In[12]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print("Tokenizer input max length:", tokenizer.model_max_length)
print("Tokenizer vocabulary size:", tokenizer.vocab_size)


# Let's create helper functions to tokenize the text. Notice the arguments sent to the tokenizer.  
# __Padding__ is a strategy for ensuring tensors are rectangular by adding a special padding token to shorter sentences.   
# On the other hand , sometimes a sequence may be too long for a model to handle. In this case, you’ll need to __truncate__ the sequence to a shorter length.

# In[13]:


def tokenize_text(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

def tokenize_dataset(dataset):
    dataset_tokenized = dataset.map(tokenize_text, batched=True, batch_size =None)
    return dataset_tokenized

dataset_tokenized = tokenize_dataset(dataset)


# In[14]:


# we would like to work with pytorch so we can manually fine-tune
dataset_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])


# In[15]:


# no need to parrarelize in this assignment
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ### __Setting up the dataloaders and dataset__

# We will now set up the dataloaders for efficient batching and loading of the data.  
# By now, you are familiar with the Class methods that are needed to create a working Dataloader.
# 

# In[16]:


from torch.utils.data import DataLoader, Dataset


# In[17]:


class IMDBDataset(Dataset):
    def __init__(self, dataset):
        self.ds = dataset

    def __getitem__(self, index):
        return self.ds[index]

    def __len__(self):
        return self.ds.num_rows


# In[18]:


train_dataset = IMDBDataset(dataset_tokenized['train'])
val_dataset = IMDBDataset(dataset_tokenized['val'])
toy_dataset = IMDBDataset(dataset_tokenized['toy'])


# In[19]:


dl_train,dl_val, dl_toy = [ 
    DataLoader(
    dataset=train_dataset,
    batch_size=12,
    shuffle=True, 
    num_workers=0
),
DataLoader(
    dataset=val_dataset,
    batch_size=12,
    shuffle=True, 
    num_workers=0
),
DataLoader(
    dataset=toy_dataset,
    batch_size=4,
    num_workers=0
)]


# ### Transformer Encoder
# <a id=part3_3_2></a>

# The model we will use for the task at hand, is the encoder of the transformer proposed in the seminal paper 'Attention Is All You Need'.
# 
# The encoder is composed of positional encoding, and then multiple blocks which compute multi-head attention, layer normalization and a feed forward network as described in the diagram below.
# 
# 

# <img src="imgs/transformer_encoder.png" alt="Alternative text" />

# We provided you with implemetations for the positional encoding and the position-wise feed forward MLP in hw3/transformer.py. 
# 
# Feel free to read through the implementations to make sure you understand what they do.

# **TODO**: To begin with, complete the transformer EncoderLayer in hw3/transformer.py

# In[20]:


from hw3.transformer import EncoderLayer
# set torch seed for reproducibility
torch.manual_seed(0)
layer = EncoderLayer(embed_dim=16, hidden_dim=16, num_heads=4, window_size=4, dropout=0.1)

# load x and y
x = torch.load(os.path.join('test_tensors','encoder_layer_input.pt'))
y = torch.load(os.path.join('test_tensors','encoder_layer_output.pt'))
padding_mask = torch.ones(2, 10)
padding_mask[:, 5:] = 0

# forward pass
out = layer(x, padding_mask)
test.assertTrue(torch.allclose(out, y, atol=1e-6), 'output of encoder layer is incorrect')


# In order to classify a sentence using the encoder, we need to somehow summarize the output of the last encoder layer (which will include an output for each token in the tokenized input sentence). 
# 
# There are several options for doing this. We will use the output of the special token [CLS] appended to the beginning of each sentence by the bert tokenizer we are using.
# 
# Let's see an example of the first tokens in a sentence after tokenization:

# In[21]:


tokenizer.convert_ids_to_tokens(dataset_tokenized['train'][0]['input_ids'])[:10]


# 
# 
# **TODO**: Now it's time to put it all together. Complete the implementaion of 'Encoder' in hw3/transformer.py

# In[41]:


from hw3.transformer import Encoder

# set torch seed for reproducibility
torch.manual_seed(0)
encoder = Encoder(vocab_size=100, embed_dim=16, num_heads=4, num_layers=3, 
                  hidden_dim=16, max_seq_length=64, window_size=4, dropout=0.1)


# load x and y
x = torch.load(os.path.join('test_tensors','encoder_input.pt'))
y = torch.load(os.path.join('test_tensors','encoder_output.pt'))
# x = torch.randint(0, 100, (2, 64)).long()

padding_mask = torch.ones(2, 64)
padding_mask[:, 50:] = 0

# forward pass
out = encoder(x, padding_mask)
test.assertTrue(torch.allclose(out, y, atol=1e-6), 'output of encoder layer is incorrect')


# ### Training the encoder
# <a id=part3_3_3></a>

# We will now proceed to train the model. 
# 
# **TODO**: Complete the implementation of TransformerEncoderTrainer in hw3/training.py

# #### Training on a toy dataset

# To begin with, we will train on a small toy dataset of 40 samples. This will serve as a sanity check to make sure nothing is buggy.
# 
# **TODO**: choose the hyperparameters in hw3.answers part3_transformer_encoder_hyperparams.

# In[116]:


from hw3.answers import part4_transformer_encoder_hyperparams

params = part4_transformer_encoder_hyperparams()
print(params)
embed_dim = params['embed_dim'] 
num_heads = params['num_heads']
num_layers = params['num_layers']
hidden_dim = params['hidden_dim']
window_size = params['window_size']
dropout = params['droupout']
lr = params['lr']

vocab_size = tokenizer.vocab_size
max_seq_length = tokenizer.model_max_length

max_batches_per_epoch = None
N_EPOCHS = 20


# In[119]:


toy_model = Encoder(vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_seq_length, window_size, dropout=dropout).to(device)
toy_optimizer = optim.Adam(toy_model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()


# In[120]:


# fit your model
import pickle
if not os.path.exists('toy_transfomer_encoder.pt'):
    # overfit
    from hw3.training import TransformerEncoderTrainer
    toy_trainer = TransformerEncoderTrainer(toy_model, criterion, toy_optimizer)
    # set max batches per epoch
    _ = toy_trainer.fit(dl_toy, dl_toy, N_EPOCHS, checkpoints='toy_transfomer_encoder', max_batches=max_batches_per_epoch)

    

toy_saved_state = torch.load('toy_transfomer_encoder.pt')
toy_best_acc = toy_saved_state['best_acc']
toy_model.load_state_dict(toy_saved_state['model_state']) 



# In[109]:


test.assertTrue(toy_best_acc >= 95)


# #### Training on all data

# Congratulations! You are now ready to train your sentiment analysis classifier!
# 

# In[121]:


max_batches_per_epoch = 500
N_EPOCHS = 4


# In[122]:


model = Encoder(vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_seq_length, window_size, dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)


# In[112]:


# fit your model
import pickle
if not os.path.exists('trained_transfomer_encoder.pt'):
    from hw3.training import TransformerEncoderTrainer
    trainer = TransformerEncoderTrainer(model, criterion, optimizer)
    # set max batches per epoch
    _ = trainer.fit(dl_train, dl_val, N_EPOCHS, checkpoints='trained_transfomer_encoder', max_batches=max_batches_per_epoch)
    

saved_state = torch.load('trained_transfomer_encoder.pt')
best_acc = saved_state['best_acc']
model.load_state_dict(saved_state['model_state']) 
    

    


# In[113]:


test.assertTrue(best_acc >= 65)


# Run the follwing cells to see an example of the model output:

# In[ ]:


rand_index = torch.randint(len(dataset_tokenized['val']), (1,))
rand_index


# In[ ]:


sample = dataset['val'][rand_index]
sample['text']


# In[ ]:


tokenized_sample = dataset_tokenized['val'][rand_index]
tokenized_sample
input_ids = tokenized_sample['input_ids'].to(device)
label = tokenized_sample['label'].to(device)
attention_mask = tokenized_sample['attention_mask'].to(float).to(device)

print('label', label.shape)
print('attention_mask', attention_mask.shape)
prediction = model.predict(input_ids, attention_mask).squeeze(0)

print('label: {}, prediction: {}'.format(label, prediction))


# In the next part you wil see how to fine-tune a pretrained model for the same task.

# In[ ]:


from cs236781.answers import display_answer
import hw3.answers


# ## Questions

# Fill your answers in hw3.answers.part3_q1 and hw3.answers.part3_q2 

# ### Question 1

# Explain why stacking encoder layers that use the sliding-window attention results in a broader context in the final layer.
# Hint: Think what happens when stacking CNN layers.
# 

# In[ ]:


display_answer(hw3.answers.part3_q1)


# ### Question 2

# Propose a variation of the attention pattern such that the computational complexity stays similar to that of the sliding-window attention O(nw), but the attention is computed on a more global context.
# Note: There is no single correct answer to this, feel free to read the paper that proposed the sliding-window. Any solution that makes sense will be considered correct.

# In[ ]:


display_answer(hw3.answers.part3_q2)


# In[ ]:




