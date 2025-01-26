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
    # Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 96
    hypers["h_dim"] = 512
    hypers["z_dim"] = 64
    hypers["x_sigma2"] = 0.1
    hypers["learn_rate"] = 0.0002
    hypers["betas"] = (0.9, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
Looking at the loss function of the VAE, we can see that the reconstruction loss is the sum of the squared differences
between the input and the output of the decoder, divided by the variance of the input: the $\sigma^2$ hyperparameter.
This means that when $\sigma^2$ is low, the VAE is pushed to reconstruct the input more precisely, as is more heacily 
pentalize high differences. In practice this cal also lead to overfitting, since the model try to memorize the input
rateher then learn more general latent features.
On the other hand, when $\sigma^2$ is high, the VAE is less strict about the reconstruction, which may allow
the model to learn more robust, higher level features, but can also lead to underfitting. 


"""

part2_q2 = r"""
**Your answer:**
1. The reconstruction loss's purpose is to ensure that the model can reconstruct the input data well from the latent space.
The KL divergence loss's purpose is to ensure that the latent space distribution is close to the prior distribution.

2. When the KL loss term is low, the latent space distribution is close to the prior distribution. 
So for example, if the prior distribution is a standard normal distribution, the KL loss will 
push the latent space to also be a standard normal distribution.

3. When the latent space distribution is close to the prior distribution, it ensures the latent space is smooth and continuous 
like the prior distribution, making it possible to sample meaningful latent variables during generation.
In addition, it also prevents overfitting by encoding into a more generalizable latent space, and it allows thes 
interpolation and generation of new data points, because the latent space is similar to the prior distrubution.

"""

part2_q3 = r"""
**Your answer:**
We maximize the evidence distribution $p(X)$ because in generative modeling we want our model to assign high
probability to the observed data. Intuitively, the better the model “explains” or “fits” the data, the larger
$p(X)$ becomes. Hence, finding parameters that maximize $p(X)$ yields a model that best represents or generates
the data we observe.
(In practice we maximize an approximation of the evidence distribution (the ELBO), since the true evidence
distribution is intractable to compute.)

"""

part2_q4 = r"""
**Your answer:**
We let the encoder output the log of the latent-space variance, i.e. $\log \sigma_{\alpha}^2$, instead of $\sigma_{\alpha}^2$ 
directly in order to ensure that the resulting variance is always positive, and to improve numerical stability 
by avoiding extremely large or small values that might arise if the network output without the log.

"""

# Part 3 answers
def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=64,  # Batch size for training
        h_dim=128,      # Hidden dimension for generator/discriminator
        z_dim=100,      # Latent dimension for noise vector
        x_sigma2=1.0,   # Data variance (assuming normalized images)
        learn_rate=2e-4,  # Learning rate for Adam optimizer
        betas=(0.5, 0.999),  # Betas for Adam optimizer (momentum parameters)
    )

    # TODO: UNDERSTAND IF WE SHOULD LEAVE IT AS IT IS
    hypers["discriminator_optimizer"] = {
    'type':'Adam',
    'lr':0.0002,
    'betas':(0.5, 0.999)
    }
    hypers["generator_optimizer"] = {
        'type':'Adam',
        'lr':0.0002,
        'betas':(0.5, 0.999)
    }
    hypers["data_label"] = 0
    hypers["label_noise"] = 0.15
    # ========================
    return hypers

part3_q1 = r"""
Explain in detail why during training we sometimes need to maintain gradients when sampling from the GAN,
and other times we don't. When are they maintained and why? When are they discarded and why?
**Your answer:**
We maintain gradients when sampling from the GAN when we want the generator to learn from the discrimnator's feedback,
meaning that if the discriminator is able to distinguish between real and fake samples, the generator should learn from
this feedback and adjust its weights accordingly, this is done by backpropagating the gradients from the discriminator
to the generator.
On the other hand, we discard gradients when sampling from the GAN when we only want to generate new samples
without training the generator, for example for inference or evaluation.


"""

part3_q2 = r"""
**Your answer:**
1. When training a GAN to generate images, should we decide to stop training solely based on the fact that the Generator loss is below some threshold? Why or why not?
No, when only the generator loss is below some threshold, we might still need to continue training.
This is because this situation might happen even when the generator produces poor quality images, but the discriminator is even weaker, 
failing to classify even them and real data. 
A weak Discriminator would incorrectly classify the Generator's outputs, leading to a low Generator loss 
without reflecting actual progress in image quality

2. What does it mean if the discriminator loss remains at a constant value while the generator loss decreases?
It means that the discriminator is no longer improving and that it reached a "plateau" in its learning, meaning it can't improve anymore.
When the generator loss decreases, it means that the generator is still learning and improving, producing higher quality samples which fool the Discriminator.

"""

part3_q3 = r"""
**Your answer:**



"""



PART3_CUSTOM_DATA_URL = "https://github.com/AviaAvraham1/TempDatasets/raw/refs/heads/main/George_W_Bush2.zip"


def part4_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim=256,       # Moderate size to balance capacity and stability
        num_heads=4,         # Ensure embed_dim % num_heads == 0
        num_layers=6,        # Reduced depth for faster convergence
        hidden_dim=512,      # Keep the feed-forward network expressive
        window_size=32,       # Moderate window size for both local and global context
        droupout=0.1,        # Introduce dropout to regularize
        lr=1e-4,             # Lower learning rate to stabilize training
    )

    # Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers




part4_q1 = r"""
**Your answer:**
When stacking encoder layers that use the sliding-window attention, the final layer has a broader context due to the 
receptive field expansion with each layer.
For example, when window size is 2, in the first layer, each token attends to itself and one token to the left and right.
But in the second layer, the representation of each token already incorporates information from its immediate neighbors,
so if each token in the first layer attends to 3 tokens and the second layer allows a token to indirectly attend to neighbors
of its neighbors, it's effectively covering a window of up to 5 tokens. And so on until the final layer, which has the 
widest context.
"""

part4_q2 = r"""
**Your answer:**
We suggest a slight modification to the sliding window attention mechanism that intends to make the context more global:
we still define a window size, but instead of simply attending to the tokens within the window, we also define a stride parameter, 
which determines the number of tokens to skip between each token in the window. The number of tokens to attend to will still be
limited by the window size (hence the complexity remains O(nw)), but the stride will allow the model to attend to tokens that are
further apart from each other, and thus increase the global context of the attention mechanism. Also, since the stride remains
constant, it will allow coverage of the entire sequence over all tokens.

"""


part5_q1 = r"""
**Your answer:**
Both our fine-tuned models performed better than the trained from scratch model from the previous part 
(both fine-tuned model gained accuracy of over 80%, while the model from the previous part gained accuracy of
a little below 70%).
It likely happened because using the method of fine-tuning allowed the model to adapt its pre-trained knowledge
entirely to the task, and focus on optimizing both general and task-specific representations, while training from
scratch requires learning everything from the ground up, including basic language understanding, which is less efficient.
This won't always necessarily be the case on any downstream task. For example, if the downstream task domain
significantly differs from the pre-training data, fine-tuning may not be as effective.

"""

part5_q2 = r"""
**Your answer:**
The results of freezing the last layers and fine-tuning only the last layers will likely be worse than fine-tuning
the freezing the internal ones and fine-tuning the last ones. This is because the internal layers are responsible 
for general features extraction, like capturing syntax and semantics, while the last layers are the ones that 
are closer to the output space and are designed to produce task-specific representations, thus freezing them will
significantly limit the model's ability to adapt to the new task, which is the basic idea behind fine-tuning.

"""


part5_q3= r"""
**Your answer:**
To use BERT for machine translation, we'd need to make changes to the model architecture and pre-training.
First, BERT is a bidirectional encoder only model, but machine translation requires an encoder-decoder architecture -
the encoder will process the input tokens from the source language ($x_t$), and the decoder will generate the 
output tokens ($y_t$), using previously generated tokens as input.

A change in the pre-training (or fine tune) is required, since we'd need to train the model on two languages in parallel, 
in order to teach the model how to translate between them.
"""

part5_q4 = r"""
**Your answer:**
RNNs are a good choice for tasks with strong temporal dependencies, where the current output heavily relies
on previous inputs, such as speech recognition or real-time sensor data. They process sequences step-by-step
in order, making them a good fit for data with a clear progression over time. 
Additionally, RNNs are more memory-efficient than Transformers when working with very long sequences, as they
don’t require computing attention over all input tokens at once. This makes them a better choice for limited
resources or variable-length data.

"""

part5_q5 = r"""
**Your answer:**
NSP is a pre-trained task in BERT where the model is trained to predict whether two sentences are consecutive or 
sampled independently from the dataset.
The prediction occurs at the CLS token, a special token added at the start of the input sequence, which serves
as a summary representation of the entire input pair for classification tasks.
The loss is the binary cross-entropy loss, where the labels indicate whether the two sentences are consecutive or not.
While the NSP can help BERT understand sentence relationships, which is useful for tasks like sentence prediction
and answering questions, it may not be crucial because language modeling already captures much of this implicitly.
We may say that its importance depends on how much sentence-level reasoning the downstream tasks require.
"""


# ==============
