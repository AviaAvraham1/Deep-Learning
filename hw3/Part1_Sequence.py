#!/usr/bin/env python
# coding: utf-8

# $$ ...
# (LaTeX macros)
# $$

# Part 1: Sequence Models
# <a id=part1></a>

# In this part we will learn about working with text sequences using recurrent neural networks.
# We'll go from a raw text file all the way to a fully trained GRU-RNN model and generate works of art!

# In[1]:

import unittest
import os
import sys
import pathlib
import urllib
import shutil
import re

import numpy as np
import torch
import matplotlib.pyplot as plt

# Remove IPython-specific lines for standalone execution
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


test = unittest.TestCase()
plt.rcParams.update({'font.size': 12})
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# ## Text generation with a char-level RNN
# <a id=part1_1></a>

# ### Obtaining the corpus
# <a id=part1_2></a>

# Let's begin by downloading a corpus containing all the works of William Shakespeare.
# Since he was very prolific, this corpus is fairly large and will provide us with enough data for
# obtaining impressive results.

# In[3]:


CORPUS_URL = 'https://github.com/cedricdeboom/character-level-rnn-datasets/raw/master/datasets/shakespeare.txt'
DATA_DIR = pathlib.Path.home().joinpath('.pytorch-datasets')

def download_corpus(out_path=DATA_DIR, url=CORPUS_URL, force=False):
    pathlib.Path(out_path).mkdir(exist_ok=True)
    out_filename = os.path.join(out_path, os.path.basename(url))
    
    if os.path.isfile(out_filename) and not force:
        print(f'Corpus file {out_filename} exists, skipping download.')
    else:
        print(f'Downloading {url}...')
        with urllib.request.urlopen(url) as response, open(out_filename, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print(f'Saved to {out_filename}.')
    return out_filename
    
corpus_path = download_corpus()


# Load the text into memory and print a snippet:

# In[4]:


with open(corpus_path, 'r', encoding='utf-8') as f:
    corpus = f.read()

print(f'Corpus length: {len(corpus)} chars')
print(corpus[7:1234])


# ### Data Preprocessing
# <a id=part1_3></a>

# The first thing we'll need is to map from each unique character in the corpus to an index that will represent it in our learning process.
# 
# **TODO**: Implement the `char_maps()` function in the `hw3/charnn.py` module.

# In[5]:


import hw3.charnn as charnn

char_to_idx, idx_to_char = charnn.char_maps(corpus)
print(char_to_idx)

test.assertEqual(len(char_to_idx), len(idx_to_char))
test.assertSequenceEqual(list(char_to_idx.keys()), list(idx_to_char.values()))
test.assertSequenceEqual(list(char_to_idx.values()), list(idx_to_char.keys()))


# Seems we have some strange characters in the corpus that are very rare and are probably due to mistakes.
# To reduce the length of each tensor we'll need to later represent our chars, it's best to remove them.
# 
# **TODO**: Implement the `remove_chars()` function in the `hw3/charnn.py` module.

# In[6]:


corpus, n_removed = charnn.remove_chars(corpus, ['}','$','_','<','\ufeff'])
print(f'Removed {n_removed} chars')

# After removing the chars, re-create the mappings
char_to_idx, idx_to_char = charnn.char_maps(corpus)


# The next thing we need is an **embedding** of the chracters.
# An embedding is a representation of each token from the sequence as a tensor.
# For a char-level RNN, our tokens will be chars and we can thus use the simplest possible embedding: encode each char as a **one-hot** tensor. In other words, each char will be represented
# as a tensor whos length is the total number of unique chars (`V`) which contains all zeros except at the index
# corresponding to that specific char.
# 
# **TODO**: Implement the functions `chars_to_onehot()` and `onehot_to_chars()` in the `hw3/charnn.py` module.

# In[7]:


# Wrap the actual embedding functions for calling convenience
def embed(text):
    return charnn.chars_to_onehot(text, char_to_idx)

def unembed(embedding):
    return charnn.onehot_to_chars(embedding, idx_to_char)

text_snippet = corpus[3104:3148]
print(text_snippet)
print(embed(text_snippet[0:3]))

test.assertEqual(text_snippet, unembed(embed(text_snippet)))
test.assertEqual(embed(text_snippet).dtype, torch.int8)


# ### Dataset Creation
# <a id=part1_4></a>

# We wish to train our model to generate text by constantly predicting what the next char should be based on the past.
# To that end we'll need to train our recurrent network in a way similar to a classification task. At each timestep, we input a char and set the expected output (label) to be the next char in the original sequence.
# 
# We will split our corpus into shorter sequences of length `S` chars (see question below).
# Each **sample** we provide our model with will therefore be a tensor of shape `(S,V)` where `V` is the embedding dimension. Our model will operate sequentially on each char in the sequence.
# For each sample, we'll also need a **label**. This is simply another sequence, shifted by one char so that the label of each char is the next char in the corpus.

# **TODO**: Implement the `chars_to_labelled_samples()` function in the `hw3/charnn.py` module.

# In[8]:


# Create dataset of sequences
seq_len = 64
vocab_len = len(char_to_idx)

# Create labelled samples
samples, labels = charnn.chars_to_labelled_samples(corpus, char_to_idx, seq_len, device)
print(f'samples shape: {samples.shape}')
print(f'labels shape: {labels.shape}')

# Test shapes
num_samples = (len(corpus) - 1) // seq_len
test.assertEqual(samples.shape, (num_samples, seq_len, vocab_len))
test.assertEqual(labels.shape, (num_samples, seq_len))

# Test content
for _ in range(1000):
    # random sample
    i = np.random.randint(num_samples, size=(1,))[0]
    # Compare to corpus
    test.assertEqual(unembed(samples[i]), corpus[i*seq_len:(i+1)*seq_len], msg=f"content mismatch in sample {i}")
    # Compare to labels
    sample_text = unembed(samples[i])
    label_text = str.join('', [idx_to_char[j.item()] for j in labels[i]])
    test.assertEqual(sample_text[1:], label_text[0:-1], msg=f"label mismatch in sample {i}")


# Let's print a few consecutive samples. You should see that the text continues between them.

# In[9]:


import re
import random

i = random.randrange(num_samples-5)
for i in range(i, i+5):
    test.assertEqual(len(samples[i]), seq_len)
    s = re.sub(r'\s+', ' ', unembed(samples[i])).strip()
    print(f'sample [{i}]:\n\t{s}')


# As usual, instead of feeding one sample at a time into our model's forward we'll work with **batches** of samples. This means that at every timestep, our model will operate on a batch of chars that are from **different sequences**.
# Effectively this will allow us to parallelize training our model by dong matrix-matrix multiplications
# instead of matrix-vector during the forward pass.

# An important nuance is that we need the batches to be **contiguous**, i.e. sample $k$ in batch $j$ should continue sample $k$ from batch $j-1$.
# The following figure illustrates this:
# 
# <img src="imgs/rnn-batching.png"/>
# 
# If we naïvely take consecutive samples into batches, e.g. `[0,1,...,B-1]`, `[B,B+1,...,2B-1]` and so on, we won't have contiguous
# sequences at the same index between adjacent batches.
# 
# To accomplish this we need to tell our `DataLoader` which samples to combine together into one batch.
# We do this by implementing a custom PyTorch `Sampler`, and providing it to our `DataLoader`.

# **TODO**: Implement the `SequenceBatchSampler` class in the `hw3/charnn.py` module.

# In[10]:


from hw3.charnn import SequenceBatchSampler

sampler = SequenceBatchSampler(dataset=range(32), batch_size=10)
sampler_idx = list(sampler)
print('sampler_idx =\n', sampler_idx)

# Test the Sampler
test.assertEqual(len(sampler_idx), 30)
batch_idx = np.array(sampler_idx).reshape(-1, 10)
for k in range(10):
    test.assertEqual(np.diff(batch_idx[:, k], n=2).item(), 0)


# Even though we're working with sequences, we can still use the standard PyTorch `Dataset`/`DataLoader` combo.
# For the dataset we can use a built-in class, `TensorDataset` to return tuples of `(sample, label)`
# from the `samples` and `labels` tensors we created above.
# The `DataLoader` will be provided with our custom `Sampler` so that it generates appropriate batches.

# In[11]:


import torch.utils.data

# Create DataLoader returning batches of samples.
batch_size = 32

ds_corpus = torch.utils.data.TensorDataset(samples, labels)
sampler_corpus = SequenceBatchSampler(ds_corpus, batch_size)
dl_corpus = torch.utils.data.DataLoader(ds_corpus, batch_size=batch_size, sampler=sampler_corpus, shuffle=False)


# Let's see what that gives us:

# In[12]:


print(f'num batches: {len(dl_corpus)}')

x0, y0 = next(iter(dl_corpus))
print(f'shape of a batch of samples: {x0.shape}')
print(f'shape of a batch of labels: {y0.shape}')


# Now lets look at the same sample index from multiple batches taken from our corpus.

# In[13]:


# Check that sentences in in same index of different batches complete each other.
k = random.randrange(batch_size)
for j, (X, y) in enumerate(dl_corpus,):
    print(f'=== batch {j}, sample {k} ({X[k].shape}): ===')
    s = re.sub(r'\s+', ' ', unembed(X[k])).strip()
    print(f'\t{s}')
    if j==4: break


# ### Model Implementation
# <a id=part1_5></a>

# Finally, our data set is ready so we can focus on our model.
# 
# We'll implement here is a multilayer gated recurrent unit (GRU) model, with dropout.
# This model is a type of RNN which performs similar to the well-known LSTM model,
# but it's somewhat easier to train because it has less parameters.
# We'll modify the regular GRU slightly by applying dropout to
# the hidden states passed between layers of the model.
# 
# The model accepts an input $\mat{X}\in\set{R}^{S\times V}$ containing a sequence of embedded chars.
# It returns an output $\mat{Y}\in\set{R}^{S\times V}$ of predictions for the next char and the final hidden state
# $\mat{H}\in\set{R}^{L\times H}$. Here $S$ is the sequence length, $V$ is the vocabulary size (number of unique chars), $L$ is the number of layers in the model and $H$ is the hidden dimension.

# Mathematically, the model's forward function at layer $k\in[1,L]$ and timestep $t\in[1,S]$ can be described as
# 
# $$
# \begin{align}
# \vec{z_t}^{[k]} &= \sigma\left(\vec{x}^{[k]}_t {\mattr{W}_{\mathrm{xz}}}^{[k]} +
#     \vec{h}_{t-1}^{[k]} {\mattr{W}_{\mathrm{hz}}}^{[k]} + \vec{b}_{\mathrm{z}}^{[k]}\right) \\
# \vec{r_t}^{[k]} &= \sigma\left(\vec{x}^{[k]}_t {\mattr{W}_{\mathrm{xr}}}^{[k]} +
#     \vec{h}_{t-1}^{[k]} {\mattr{W}_{\mathrm{hr}}}^{[k]} + \vec{b}_{\mathrm{r}}^{[k]}\right) \\
# \vec{g_t}^{[k]} &= \tanh\left(\vec{x}^{[k]}_t {\mattr{W}_{\mathrm{xg}}}^{[k]} +
#     (\vec{r_t}^{[k]}\odot\vec{h}_{t-1}^{[k]}) {\mattr{W}_{\mathrm{hg}}}^{[k]} + \vec{b}_{\mathrm{g}}^{[k]}\right) \\
# \vec{h_t}^{[k]} &= \vec{z}^{[k]}_t \odot \vec{h}^{[k]}_{t-1} + \left(1-\vec{z}^{[k]}_t\right)\odot \vec{g_t}^{[k]}
# \end{align}
# $$

# The input to each layer is,
# $$
# \mat{X}^{[k]} =
# \begin{bmatrix}
#     {\vec{x}_1}^{[k]} \\ \vdots \\ {\vec{x}_S}^{[k]}
# \end{bmatrix} 
# =
# \begin{cases}
#     \mat{X} & \mathrm{if} ~k = 1~ \\
#     \mathrm{dropout}_p \left(
#     \begin{bmatrix}
#         {\vec{h}_1}^{[k-1]} \\ \vdots \\ {\vec{h}_S}^{[k-1]}
#     \end{bmatrix} \right) & \mathrm{if} ~1 < k \leq L+1~
# \end{cases}.
# $$

# The output of the entire model is then,
# $$
# \mat{Y} = \mat{X}^{[L+1]} {\mattr{W}_{\mathrm{hy}}} + \mat{B}_{\mathrm{y}}
# $$

# and the final hidden state is
# $$
# \mat{H} = 
# \begin{bmatrix}
#     {\vec{h}_S}^{[1]} \\ \vdots \\ {\vec{h}_S}^{[L]}
# \end{bmatrix}.
# $$

# Notes:
# - $t\in[1,S]$ is the timestep, i.e. the current position within the sequence of each sample.
# - $\vec{x}_t^{[k]}$ is the input of layer $k$ at timestep $t$, respectively.
# - The outputs of the **last layer** $\vec{y}_t^{[L]}$, are the predicted next characters for every input char.
#   These are similar to class scores in classification tasks.
# - The hidden states at the **last timestep**, $\vec{h}_S^{[k]}$, are the final hidden state returned from the model.
# - $\sigma(\cdot)$ is the sigmoid function, i.e. $\sigma(\vec{z}) = 1/(1+e^{-\vec{z}})$ which returns values in $(0,1)$.
# - $\tanh(\cdot)$ is the hyperbolic tangent, i.e. $\tanh(\vec{z}) = (e^{2\vec{z}}-1)/(e^{2\vec{z}}+1)$ which returns values in $(-1,1)$.
# - $\vec{h_t}^{[k]}$ is the hidden state of layer $k$ at time $t$. This can be thought of as the memory of that layer.
# - $\vec{g_t}^{[k]}$ is the candidate hidden state for time $t+1$.
# - $\vec{z_t}^{[k]}$ is known as the update gate. It combines the previous state with the input to determine how much the current state will be combined with the new candidate state. For example, if $\vec{z_t}^{[k]}=\vec{1}$ then the current input has no effect on the output.
# - $\vec{r_t}^{[k]}$ is known as the reset gate. It combines the previous state with the input to determine how much of the previous state will affect the current state candidate. For example if $\vec{r_t}^{[k]}=\vec{0}$ the previous state has no effect on the current candidate state.

# Here's a graphical representation of the GRU's forward pass at each timestep. The $\vec{\tilde{h}}$ in the image is our $\vec{g}$ (candidate next state).
# 
# <img src="imgs/gru_cell.png" width="400"/>
# 
# You can see how the reset and update gates allow the model to completely ignore it's previous state, completely ignore it's input, or any mixture of those states (since the gates are actually continuous and between $(0,1)$).

# Here's a graphical representation of the entire model.
# You can ignore the $c_t^{[k]}$ (cell state) variables (which are relevant for LSTM models).
# Our model has only the hidden state, $h_t^{[k]}$. Also notice that we added dropout between layers (i.e., on the up arrows).
# 
# <img src="imgs/lstm_model.png" />
# 
# The purple tensors are inputs (a sequence and initial hidden state per layer), and the green tensors are outputs (another sequence and final hidden state per layer). Each blue block implements the above forward equations.
# Blocks that are on the same vertical level are at the same layer, and therefore share parameters.

# **TODO**:implement `MultilayerGRU` class in the `hw3/charnn.py` module.
# 
# 
# Notes:
# - We use **batches** now.
#   The math is identical to the above, but all the tensors will have an extra batch 
#   dimension as their first dimension.
# - Read the diagram above, try to understand all the dimentions.
# 

# In[14]:


in_dim = vocab_len
h_dim = 256
n_layers = 3
model = charnn.MultilayerGRU(in_dim, h_dim, out_dim=in_dim, n_layers=n_layers)
model = model.to(device)
print(model)

# Test forward pass
y, h = model(x0.to(dtype=torch.float, device=device))
print(f'y.shape={y.shape}')
print(f'h.shape={h.shape}')

test.assertEqual(y.shape, (batch_size, seq_len, vocab_len))
test.assertEqual(h.shape, (batch_size, n_layers, h_dim))
test.assertEqual(len(list(model.parameters())), 9 * n_layers + 2) 


# ### Generating text by sampling
# <a id=part1_6></a>

# Now that we have a model, we can implement **text generation** based on it.
# The idea is simple:
# At each timestep our model receives one char $x_t$ from the input sequence and outputs scores $y_t$
# for what the next char should be.
# We'll convert these scores into a probability over each of the possible chars.
# In other words, for each input char $x_t$ we create a probability distribution for the next char
# conditioned on the current one and the state of the model (representing all previous inputs):
# $$p(x_{t+1}|x_t, \vec{h}_t).$$
# 
# Once we have such a distribution, we'll sample a char from it.
# This will be the first char of our generated sequence.
# Now we can feed this new char into the model, create another distribution, sample the next char and so on.
# Note that it's crucial to propagate the hidden state when sampling.

# The important point however is how to create the distribution from the scores.
# One way, as we saw in previous ML tasks, is to use the softmax function.
# However, a drawback of softmax is that it can generate very diffuse (more uniform) distributions if the score values are very similar. When sampling, we would prefer to control the distributions and make them less uniform to increase the chance of sampling the char(s) with the highest scores compared to the others.
# 
# To control the variance of the distribution, a common trick is to add a hyperparameter $T$, known as the 
# *temperature* to the softmax function. The class scores are simply scaled by $T$ before softmax is applied:
# $$
# \mathrm{softmax}_T(\vec{y}) = \frac{e^{\vec{y}/T}}{\sum_k e^{y_k/T}}
# $$
# 
# A low $T$ will result in less uniform distributions and vice-versa.

# **TODO**: Implement the `hot_softmax()` function in the `hw3/charnn.py` module.

# In[15]:


scores = y[0,0,:].detach()
_, ax = plt.subplots(figsize=(15,5))

for t in reversed([0.3, 0.5, 1.0, 100]):
    ax.plot(charnn.hot_softmax(scores, temperature=t).cpu().numpy(), label=f'T={t}')
ax.set_xlabel('$x_{t+1}$')
ax.set_ylabel('$p(x_{t+1}|x_t)$')
ax.legend()

uniform_proba = 1/len(char_to_idx)
uniform_diff = torch.abs(charnn.hot_softmax(scores, temperature=100) - uniform_proba)
test.assertTrue(torch.all(uniform_diff < 1e-4))


# **TODO**: Implement the `generate_from_model()` function in the `hw3/charnn.py` module.

# In[16]:


for _ in range(3):
    text = charnn.generate_from_model(model, "foobar", 50, (char_to_idx, idx_to_char), T=0.5)
    print(text)
    test.assertEqual(len(text), 50)


# ### Training
# <a id=part1_7></a>

# To train this model, we'll calculate the loss at each time step by comparing the predicted char to
# the actual char from our label. We can use cross entropy since per char it's similar to a classification problem.
# We'll then sum the losses over the sequence and back-propagate the gradients though time.
# Notice that the back-propagation algorithm will "visit" each layer's parameter tensors multiple times,
# so we'll accumulate gradients in parameters of the blocks. Luckily `autograd` will handle this part for us.

# As usual, the first step of training will be to try and **overfit** a large model (many parameters) to a tiny dataset.
# Again, this is to ensure the model and training code are implemented correctly, i.e. that the model can learn.
# 
# For a generative model such as this, overfitting is slightly trickier than for classification.
# What we'll aim to do is to get our model to **memorize** a specific sequence of chars, so that when given the first
# char in the sequence it will immediately spit out the rest of the sequence verbatim.
# 
# Let's create a tiny dataset to memorize.

# In[17]:


# Pick a tiny subset of the dataset
subset_start, subset_end = 1001, 1005
ds_corpus_ss = torch.utils.data.Subset(ds_corpus, range(subset_start, subset_end))
batch_size_ss = 1
sampler_ss = SequenceBatchSampler(ds_corpus_ss, batch_size=batch_size_ss)
dl_corpus_ss = torch.utils.data.DataLoader(ds_corpus_ss, batch_size_ss, sampler=sampler_ss, shuffle=False)

# Convert subset to text
subset_text = ''
for i in range(subset_end - subset_start):
    subset_text += unembed(ds_corpus_ss[i][0])
print(f'Text to "memorize":\n\n{subset_text}')


# Now let's implement the first part of our training code.
# 
# **TODO**: Implement the `train_epoch()` and `train_batch()` methods of the `RNNTrainer` class in the `hw3/training.py` module. 
# You must think about how to correctly handle the hidden state of the model between batches and epochs for this specific task (i.e. text generation).

# In[18]:


import torch.nn as nn
import torch.optim as optim
from hw3.training import RNNTrainer

torch.manual_seed(42)

lr = 0.01
num_epochs = 500

in_dim = vocab_len
h_dim = 128
n_layers = 2
loss_fn = nn.CrossEntropyLoss()
model = charnn.MultilayerGRU(in_dim, h_dim, out_dim=in_dim, n_layers=n_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
trainer = RNNTrainer(model, loss_fn, optimizer, device)

for epoch in range(num_epochs):
    epoch_result = trainer.train_epoch(dl_corpus_ss, verbose=False)
    
    # Every X epochs, we'll generate a sequence starting from the first char in the first sequence
    # to visualize how/if/what the model is learning.
    if epoch == 0 or (epoch+1) % 25 == 0:
        avg_loss = np.mean(epoch_result.losses)
        accuracy = np.mean(epoch_result.accuracy)
        print(f'\nEpoch #{epoch+1}: Avg. loss = {avg_loss:.3f}, Accuracy = {accuracy:.2f}%')
        
        generated_sequence = charnn.generate_from_model(model, subset_text[0],
                                                        seq_len*(subset_end-subset_start),
                                                        (char_to_idx,idx_to_char), T=0.1)
        
        # Stop if we've successfully memorized the small dataset.
        print(generated_sequence)
        if generated_sequence == subset_text:
            break

# Test successful overfitting
test.assertGreater(epoch_result.accuracy, 99)
test.assertEqual(generated_sequence, subset_text)


# OK, so training works - we can memorize a short sequence.
# We'll now train a much larger model on our large dataset. You'll need a GPU for this part.
# 
# First, lets set up our dataset and models for training.
# We'll split our corpus into 90% train and 10% test-set.
# Also, we'll use a learning-rate scheduler to control the learning rate during training.

# **TODO**: Set the hyperparameters in the `part1_rnn_hyperparams()` function of the `hw3/answers.py` module.

# In[19]:


from hw3.answers import part1_rnn_hyperparams

hp = part1_rnn_hyperparams()
print('hyperparams:\n', hp)

### Dataset definition
vocab_len = len(char_to_idx)
batch_size = hp['batch_size']
seq_len = hp['seq_len']
train_test_ratio = 0.9
num_samples = (len(corpus) - 1) // seq_len
num_train = int(train_test_ratio * num_samples)

samples, labels = charnn.chars_to_labelled_samples(corpus, char_to_idx, seq_len, device)

ds_train = torch.utils.data.TensorDataset(samples[:num_train], labels[:num_train])
sampler_train = SequenceBatchSampler(ds_train, batch_size)
dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False, sampler=sampler_train, drop_last=True)

ds_test = torch.utils.data.TensorDataset(samples[num_train:], labels[num_train:])
sampler_test = SequenceBatchSampler(ds_test, batch_size)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size, shuffle=False, sampler=sampler_test, drop_last=True)

print(f'Train: {len(dl_train):3d} batches, {len(dl_train)*batch_size*seq_len:7d} chars')
print(f'Test:  {len(dl_test):3d} batches, {len(dl_test)*batch_size*seq_len:7d} chars')

### Training definition
in_dim = out_dim = vocab_len
checkpoint_file = 'checkpoints/rnn'
num_epochs = 50
early_stopping = 5

model = charnn.MultilayerGRU(in_dim, hp['h_dim'], out_dim, hp['n_layers'], hp['dropout'])
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=hp['learn_rate'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=hp['lr_sched_factor'], patience=hp['lr_sched_patience'], verbose=True
)
trainer = RNNTrainer(model, loss_fn, optimizer, device)


# The code blocks below will train the model and save checkpoints containing the training state and the best model parameters to a file. This allows you to stop training and resume it later from where you left.
# 
# Note that you can use the `main.py` script provided within the assignment folder to run this notebook from the command line as if it were a python script by using the `run-nb` subcommand. This allows you to train your model using this notebook without starting jupyter. You can combine this with `srun` or `sbatch` to run the notebook with a GPU on the course servers.

# **TODO**:
# - Implement the `fit()` method of the `Trainer` class. You can reuse the relevant implementation parts from HW2, but make sure to implement early stopping and checkpoints.
# - Implement the `test_epoch()` and `test_batch()` methods of the `RNNTrainer` class in the `hw3/training.py` module.
# - Run the following block to train.
# - When training is done and you're satisfied with the model's outputs, rename the checkpoint file to `checkpoints/rnn_final.pt`.
#   This will cause the block to skip training and instead load your saved model when running the homework submission script.
#   Note that your submission zip file will not include the checkpoint file. This is OK.

# In[20]:


from cs236781.plot import plot_fit

def post_epoch_fn(epoch, train_res, test_res, verbose):
    # Update learning rate
    scheduler.step(test_res.accuracy)
    # Sample from model to show progress
    if verbose:
        start_seq = "ACT I."
        generated_sequence = charnn.generate_from_model(
            model, start_seq, 100, (char_to_idx,idx_to_char), T=0.5
        )
        print(generated_sequence)

# Train, unless final checkpoint is found
checkpoint_file_final = f'{checkpoint_file}_final.pt'
if os.path.isfile(checkpoint_file_final):
    print(f'*** Loading final checkpoint file {checkpoint_file_final} instead of training')
    saved_state = torch.load(checkpoint_file_final, map_location=device)
    model.load_state_dict(saved_state['model_state'])
else:
    try:
        # Print pre-training sampling
        print(charnn.generate_from_model(model, "ACT I.", 100, (char_to_idx,idx_to_char), T=0.5))

        fit_res = trainer.fit(dl_train, dl_test, num_epochs, max_batches=None,
                              post_epoch_fn=post_epoch_fn, early_stopping=early_stopping,
                              checkpoints=checkpoint_file, print_every=1)
        
        fig, axes = plot_fit(fit_res)
    except KeyboardInterrupt as e:
        print('\n *** Training interrupted by user')


# ### Generating a work of art
# <a id=part1_8></a>

# Armed with our fully trained model, let's generate the next Hamlet! You should experiment with modifying the sampling temperature and see what happens.
# 
# The text you generate should “look” like a Shakespeare play:
# old-style English words and sentence structure, directions for the actors
# (like “Exit/Enter”), sections (Act I/Scene III) etc.
# There will be no coherent plot of course, but it should at least seem like
# a Shakespearean play when not looking too closely.
# If this is not what you see, go back, debug and/or and re-train.
# 
# **TODO**: Specify the generation parameters in the `part1_generation_params()` function within the `hw3/answers.py` module.

# In[1]:


# delete later: (using it to provide a start sequence):
print(corpus[0:30])


# In[ ]:


from hw3.answers import part1_generation_params

start_seq, temperature = part1_generation_params()

generated_sequence = charnn.generate_from_model(
    model, start_seq, 10000, (char_to_idx,idx_to_char), T=temperature
)

print(generated_sequence)


# ## Questions
# <a id=part1_9></a>

# **TODO** Answer the following questions. Write your answers in the appropriate variables in the module `hw3/answers.py`.

# In[ ]:


from cs236781.answers import display_answer
import hw3.answers


# ### Question 1
# Why do we split the corpus into sequences instead of training on the whole text?

# In[ ]:


display_answer(hw3.answers.part1_q1)


# ### Question 2
# How is it possible that the generated text clearly shows memory longer than the sequence length?

# In[ ]:


display_answer(hw3.answers.part1_q2)


# ### Question 3
# Why are we not shuffling the order of batches when training?

# In[ ]:


display_answer(hw3.answers.part1_q3)


# ### Question 4
# 1. Why do we lower the temperature for sampling (compared to the default of $1.0$)?
# 2. What happens when the temperature is very high and why?
# 3. What happens when the temperature is very low and why?

# In[ ]:


display_answer(hw3.answers.part1_q4)

