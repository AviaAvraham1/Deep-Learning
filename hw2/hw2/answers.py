from .layers import CrossEntropyLoss
r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**

1. A. the shape of this tensor is (64,512,64,1024) - we derive every entrance in the matrix according to every
    entrance in the other matrix and X is of size (64, 1024) and Y is of size (64, 512).
   B. yes, the Jacobian matrix is sparse since each y_i,j is a function of only one x_i,j 
   (the ith sample in the batch), so we get a diagonal matrix: diag(J_i), where J_i is the
    Jacobian matrix of y_i with respect to x_i.
   C. There is no need to materialize the whole Jacobian matrix, instead of calculating by the 
   chain-rule: dL/dx = dL/dy * dy/dx, we can use the fact that the Jacobian matrix is diagonal matrix where
   each element is the Jacobian matrix of y_i with respect to x_i, i.e W^T (the derivative of x*W^T+b w.r.t x)
   and compute dL/dx = dL/dy * W^T (since dL/dx_i = dL/dy_i * W^T for each i, this is equvilent to multiplying
   directly by W^T).

2. A. (64,512,512,1024).
   B. yes, similarly to the previous question, the Jacobian matrix is sparse since each y_i,j depends on only
   the j-th row of W. so the Jacobian matrix is a block-diagonal matrix where each block is a matrix of size 512x1024.
   C. we can use the same trick as before, instead of calculating the whole Jacobian matrix, we can calculate
   dL/dW = dL/dy * X (since X is the derivative of X*W^T+b w.r.t W).

   
"""

part1_q2 = r"""
**Your answer:**

No, it is not required to use back-propagation in order to train neural networks with descent-based optimization.
The back-propagation algorithm is a method to calculate the gradient of the loss function with respect to the weights
of the network, but actually there are other methods to compute gardients to use in gradient descent optimization.
For example, we learned in the Automatic Differentiation tutorial, that we can use numerical diffrentiation
(the explicit method to compute gradients, as we learned in Calculus) but as we said, it is inexact and very expensive.
Also, that way we will need to create a gardient function for each paramater, and recalculating it upon every change in the
network, which is not very scalable.

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 1.0 
    lr = 0.01
    reg = 0.0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    lr_vanilla = 0.00009 
    reg = 0.000002
    wstd = 0.001
    lr_momentum = 0.00001
    lr_rmsprop = 0.00001
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.00009
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1. Yes, the graphs match our expectations. 
When dropout is not used, the model overfits the training data, and thus the test loss is significantly higher
than the training loss. Also, we can see that the training loss is lower compared to when droput is used.
In general, it happens because we get more expressiveness when the network is fully connected - 
without dropout we use all of our network nuerons all the time, as opposed to when we use dropout.

2. When droput=0.8, we turned off most of the neurons, and the network is less expressive, and thus
the loss is high, both in training and test.
When dropout=0.4 we get a good balance between expressiveness and generalization, and thus we see nice
results in the training loss and similar ones in the test loss.  
"""

part2_q2 = r"""
When training a model with the cross-entropy loss function, is it possible for the test loss to **increase** for a few epochs while the test accuracy also **increases**?

If it's possible explain how, if it's not explain why not.
**Your answer:**
Yes, it is possible for the test loss to increase for a few epochs while the test accuracy also increases.
The cross entopy loss also pentalizes correct predictions when they are not given with full confidence.
Therefore, it is theoretically possible that the model will predict in a certain epoch more correct samples
and thus the accuracy will increase (since accuracy simply measures the fraction of correct predictions),
but the loss will increase because the model is not confident in its predictions, and thus the overall loss
will increase.

"""

part2_q3 = r"""

**Your answer:**
1. Gradient descent is an optimiztaion method used to minimize the loss function of the model by iteratively
computing the gradient of the loss function with respect to the model's parameters and updating the parameters
in the opposite direction of the gradient, which is meant to show us the direction of the steepest descent of
the function at each point we are at.
Backpropagation is an algorithm used to compute the gradient of the loss function with respect to the model's
parameters.
So while gradient descent is the optimization method, aimed to take us to the minimum of the loss,
backpropagation is the algorithm used to compute the slope at each point (gradients).

2. In GD, we go over all of our data samples in each iteration to compute the gradient that will determine how we'll
update the parameters in that iteration in order to descend. However, in SGD, we randomly pick one sample and go in 
the direction of its gradient.

    In GD, we update: $ \vec{\theta} \leftarrow \vec{\theta} - \eta \nabla L(\theta) $

    In SGD, we update for some sample i: $\vec{\theta} \leftarrow \vec{\theta} - \eta \nabla L_i(\theta)$

3. 
In the practice of deep learning, SGD has some advatages over GD:
* **Fit in memory** - in some cases, espcially with large datasets, we might not be able to fit all of the data
in our machine's memory, which is needed in order to calculate the gradient step, in SGD, we only have to load
one sample which does fit in memory.
* **Faster** - even if we could fit all of the data in memory, iterating over all of it *in every iteration* is
expensive computationally and takes a long time. Working with SGD, however, is much faster, espcially in large
datasets, since we have less computations to do.
* **Higher success rate** - As we discussed in class, empirically, SGD is often more successful when compared
 to GD, because of considerations of noise, convergance rate and overfitting, for example

4. 
1. Yes, this approach will produce a gradient equivalent to GD:
$$L_{\text{batches}} = \sum_{\text{all\_batches}} L_{b} = \sum_{\text{all\_batches}} \sum_{\text{batch\_size}} L(x_{b,i}) = \sum_{\text{all\_samples}} L(x_{j}) = L_{\text{GD}}$$

    as we can see, the loss function of the whole dataset is the sum of the loss functions of all the batches.
    
2. What's likely happened here is that some of the intermediate results may have accumulated over time in the
compute of the gradient and thus even though we made sure that each batch fits in memory, at some point the
accumulated results together with the current batch got too big to fit in memory and therefore we got an error.
"""

part2_q4 = r"""
**Your answer:**
1.
Our derivative is $$f'(x) = f_n'\left(f_{n-1}\left(f_{n-2}\left(\ldots f_1(x) \ldots\right)\right)\right) \cdot f_{n-1}'\left(f_{n-2}\left(\ldots f_1(x) \ldots\right)\right) \cdots f_1'(x)$$

A.
We can calculate the gradient using forward mode AD as follows:

```python
function compute_gradient(f, x_0):
    # init 
    last_grad <- 1
    last_val <- x_0

    for i from 1 to n:
        curr_grad <- last_grad * f[i].derivative(last_val)
        curr_val <- f[i](last_val)
        last_grad <- curr_grad
        last_val <- curr_val

    return last_grad
```

**The memory complexity is O(1)**

At the end of the run we will have the gradient of f, by the definition shown above and the implementation
of compute_gradient. It's trivial to see that the memory complexity is O(1), because we always work with
up to 4 variables every time.

B.

We can calculate the gradient using backward mode AD as follows:

```python
function compute_gradient_reduced_memory(f, x_0):
    # init
    values <- array of size n
    values[0] <- x_0

    for i from 1 to n:
        values[i] <- f[i](values[i-1])

    last_grad <- 1
    
    # iterate from the end and calculate based on the previous value
    for i from n-1 to 0:
        curr_grad <- last_grad * f[i+1].derivative(values[i])
        last_grad <- curr_grad

    return last_grad
```

**The memory complexity is O(n), but we reduced the memory needed by half - from 2n to n**

As we can see in the loop, the grdaients are calcluted in reverse mode, meaning we start from the left side
and go right in the formula above. The need to save the values of the functions costs us O(n) memory, as opposed to the forward
mode which only needs O(1) memory, since we can use the last value to calculate the current one. 
Yet, we reduced the memory complexity by half, from 2n to n since we still kept at each iteration only the
the last gradient computed, instead of all of them.


2. Yes, these techniques can be generalized for arbitrary computational graphs, but up to a certain point.
In the common case where each layer is composed of multiple nodes, we will need to save all the gradients
of each node in the last layer at a time (instead of just one- in the case where each layer has width of 1)
and regarding the values: in forward mode we will need to save value for each node in the last layer 
(instead of just one) and in forward mode the values of all the nodes (instead of one for each layer).

3. As we've said, using these techniques, we can reduce memory usage both using forward and backward mode AD.
This is especially useful in deep architectures, where we have many layers, thus the need to save values
only for the last layer can reduce the memory usage significantly.
We may utilize this freed memory to improve the performance of the networks for example by adding layers or
widening them by adding more nodes in each layer.
"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**
1. High optimization error, also known as the training error, is an error that occur when a model does
not predict the training data well. This means that the model is not learning the patterns in the data
effectively, which could be an indication of underfitting, of a too simplistic model, or of poor training
in general, because the model failed to predict correctly even the data it was trained on.

To reduce this error, we have several methods:
* testing different optimization algorithms - for example, vanilla SGD,  momentum SGD, and RMSprop we've seen in class.
* tweak the model's hyperparameters, such as learning rate, regularization, batch size, etc. This could
be achieved by performing grid-search and cross-validation over the hyperparameters.
* use a larger dataset - can also improve our training, espcially if we don't have enough data in the beginning

2. Generalization error is the difference between the population loss (the expected loss over the true data
distribution) and the empirical loss (the average loss over the training or test dataset). Since we can't actually 
compute this value, we estimate it by measuring the difference between the training loss and the test loss.
High generalization error means that the model is not generalizing well, i.e. it is not able to predict well on
unseen data, and can indicate overfitting of the model on the training set.

In order to reduce generalization error, we can use techniques meant to reduce overfitting by reducing the 
expressiveness of the model or adding regularization to it. For example, we can use dropout, L1 or L2 regularization,

3. Approximation error is the difference between the population loss of the best possible model in the hypothesis
 class and the population loss with our selected model. This error reflects how well the chosen hypothesis class can approximate the
 true data distribution, and thus a high approximation error may indicate that the model class is not complex enough.

Therefore, the way to reduce this error is to increase the complexity of the model, for example by adding more layers, 
adding more neurons to each layer. Another way would be to increase each neuron's receptive field by adding convolutional
layers, pooling layers or adding fully connected layers that will enhance the robustness of the model and improve its
approximation capabilities, since each feature map can "learn" more about the data.

"""

part3_q2 = r"""
**Your answer:**

The most typical case where a binary classifier will produce a high rate of false positives is in the case 
where it returns a positive prediction for every sample. This will result in a false-postive for each of the
samples that are actually negative, and thus a high rate of false positives.

Similarily, a binary classifier that classifies all samples as negative will have a high rate of false negatives.
"""

part3_q3 = r"""
**Your answer:**

1. In this case, we'd prefer lower rates of false positive, in the cost of higher rate of false negatives,
since in the case of false positive (i.e healthy person misclassified as sick), the cost of the tests is very
high, despite the disease not being lethal, while in the case of false negative (i.e sick person misclassified 
as healthy) the person will likely be diagnosed despite this prediction due to the appearance of the non-lethal symptoms.
Therefore, in the ROC curve, we'd prefer to be higher on the curve, meaning we have higher precision, i.e less false
positives, again because the diagnosis is expensive.


2. In this case, we'd prefer lower rates of false negatives, even in the cost of higher rate of false positives, 
since misclassifying a sick person as healthy puts their life at risk due to the fact that there are no apperant 
symptoms, and the disease being lethal, while misclassifying a healthy person as sick only costs the tests.
Therefore, in the ROC curve, we'd prefer to be lower on the curve, meaning we have higher recall, i.e less
false negatives.

"""


part3_q4 = r"""
**Your answer:**

MLP is not a good choice for processing sequential data, because it doesn't bring into account the order
of the data or the connections between the different data points.
Specifically, regarding a model that predicts the sentiment of a sentence, it will likely fail to predict
it based on its predictions on each word separately, since it is crucial to consider each word in the context
of the entire sentence.
For example, a model that tries to average on the sentiment on ech word in order to provide prediction 
for the entire sentence will likely predict that the sentence "I've had better days" is positive, since
it has the word 'better', and it doesn't contain any negative words, while in fact, the sentence is negative.

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr = 0.001
    weight_decay = 0.001
    momentum = 0.99
    loss_fn = CrossEntropyLoss() # torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**
(1) Let's calculate the number of parameters for each example, including bias:
    (a) Regular: 
        We have two 3x3 convulutions, each one is: 3x3x256x256 + 256 = 590,080
        so a total of 2*590,080 = 1,180,160 ~ 1.18M parameters
    
    (b) Bottleneck:
        We have a 1x1 convolution, a 3x3 convolution and another 1x1 convolution, overall we have:
        1x1x256x64 + 64 = 16,384 + 64 = 16,448
        3x3x64x64 + 64 = 36,864 + 64 = 36,928 
        1x1x64x256 + 256 = 16,384 + 256 = 16,640
        so overall we have a total of 16,448 + 36,928  + 16,640 = 70,016 ~ 70K parameters
    
    The bottleneck design uses about 17x fewer parameters.

(2) Now we'll asses the number of FLOPs for each example:
    In the regular block, we have two 3x3 convolutions operating on large 256-channel feature maps,
    which results in very high amount of FLOPs due to the need to convolve these large 3x3 filters
    across all 256 input and output channels twice.
    However, in the bottleneck block, we start with a 1x1 projection from 256 to 64 channels, perform a
    3x3 convolution on the smaller 64-channel space, and finally end with a 1x1 projection back up to 256 channels.
    As we've seen in the previous question, the number of parameters in the bottleneck block is significantly
    lower than in the regular block, making it much more efficent also in terms of FLOPs.
    Overall, since the bottleneck design uses dimensionality reduction before making the 3x3 convolution to reduce
    computational cost while maintaining model expressiveness, it is more efficient in terms of the number of
    floating point operations compared to the regular block which doesn't have this dimension reduction.

(3) In terms of ability to combine the input:
    (1) Spatially (i.e within the feature-map):
        In the regular case we have two 3x3 convolutions, which actually results in a 5x5 convolution,
        which can capture more spatial information in the feature map compared to each one of the bottleneck case 
        in which we only have a 3x3 convolution in total.
    (2) Across feature maps:
        Even though both examples have some ability to combine the input across feature maps, since in both cases
        output feature map is a linear combination of all input feature maps in the kernel window, the regular case
        architecture maintains the number of feature-maps across layers, while the bottleneck case reduces the
        number of feature maps in the bottleneck layer which can limit the ability to combine the input across
        feature maps due to the information loss.
"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
(1) The model didn't detect most of the objects correctly, and also most of the detections were given with low confidence.
    In the first picture, it both misclassified the upper dolphin and failed to draw the rectangle around the two 
    lower dolphines correctly (identified them as one person + surfboard).
    In the second image it failed to detect the cat as an object, and it also predicted 2 out of the 3 dogs as cats, with only one dog identified correctly.
    It only predicted with confidence higher than 0.65 the lower two-dolphins wrongly as person with 0.9 confidence.
(2) In the first image the model didn't recognize the dolphin, which means it might not have seen enough dolphines in the
    training set, so to resolve the issue we can add more dolphines to the training set, same deal with the dogs.
    Another issue the model has was to seperate overlapping objects (both with the dolphins and the dogs&cats), so to
    resolve this issue we can add more cluttered images with overlapping objects to the training set. Also, we can try training the 
    model on smaller segments of the image and allow the model to focus on the objects separately while training.
"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""