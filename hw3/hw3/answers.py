r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=64,
        seq_len=100,
        h_dim=512,
        n_layers=3,
        dropout=0.3,
        learn_rate=0.001,
        lr_sched_factor=0.1,
        lr_sched_patience=3,
    )
    # Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    # ========================
    return hypers


def part1_generation_params():
    start_seq = "ACT I. SCENE 1."
    temperature = 0.01
    # Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    # ========================
    return start_seq, temperature


part1_q1 = r"""
We split the corpus into sequences because if we train for the whole text, we'll have to 
keep a lot of memory for long text (to save all the hidden-states in order to use later
in the backpropagation), as studied in class. So by splitting it to chunks, we can run forward and backward
through chunks of sequences, and thus only need to keep the hidden-states of the last layer
in each chunk. Also, splitting to chunks allows to speed up training using parallelism.
"""

part1_q2 = r"""
The model's memory is longer than the sequence length since the hidden state are carried over between sequences
and so they serve as a sort of long-term memory that is maintained throughout the entire training text.
Also, the GRU architecture is built to allow retaining important information over longer periods and forgetting
irrelevant information.

"""

part1_q3 = r"""
We don't shuffle the order of batches when training because if we did, we'll lose important information
regarding the position of that sequence in the text. We don't actually want to refer to each such sequence
as an independant sample, but we want to preserve their order to be able to learn their sequential dependencies.
(The importance of the order between batches is reflected in the value of the hidden state, which retains information
about previous sequences and influences the prediction of subsequent ones.)

"""

part1_q4 = r"""
1. When sampling, we would prefer to control the distributions and make them less uniform to increase the chance of
    sampling the char(s) with the highest scores compared to the others.
2. When the temperature is very high, the softmax which takes the scores and returns probablities, 
    will return probabilities that are very close to uniform distribution, meaning the predictions of the models
    are similar to randomly picking a char from the vocabulary each time, as if we didn't learn anything from the 
    training data.
3. When the temperature is very low, the softmax will exaggerate the differences between the different scores, i.e 
    sharpen the probability distribution, making the model highly confident about the most likely characters while
    practically ignoring the ones with the lower scores. So lower number of characters likely to be predicted, but 
    these are predicted with higher probability.


"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = "https://github.com/AviaAvraham1/TempDatasets/raw/refs/heads/main/George_W_Bush2.zip"


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 96
    hypers["h_dim"] = 512
    hypers["z_dim"] = 64
    hypers["x_sigma2"] = 0.1
    hypers["learn_rate"] = 0.0002
    hypers["betas"] = 0 #(a,b)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


"""

part2_q2 = r"""
**Your answer:**


"""

part2_q3 = r"""
**Your answer:**



"""

part2_q4 = r"""
**Your answer:**

"""

# Part 3 answers
def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======

    # ========================
    return hypers

part3_q1 = r"""
**Your answer:**


"""

part3_q2 = r"""
**Your answer:**


"""

part3_q3 = r"""
**Your answer:**



"""



PART3_CUSTOM_DATA_URL = None


def part4_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**


"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


part4_q3= r"""
**Your answer:**


"""

part4_q4 = r"""
**Your answer:**


"""

part4_q5 = r"""
**Your answer:**


"""


# ==============
