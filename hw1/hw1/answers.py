r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1. False -  In-sample error is the error rate of a model when it is evaluated on the same data it was trained on. 
            Thus, the test set, which is independent of the training set, can't help with evaluating this error.
2. False -  Some splits might not be as useful as others. 
            For example, if one of the features in the dataset is gender, and we split it so that the train-set
            contains only males, and the test set only females, then if this feature is statistaclly significant to
            the prediction, we should expect worse results than the ones we'll get using a balanced split
            that represents the distribution of the original dataset, which is usually achieved via random sampling.          
3. True -   The test set is kept completely separate to provide an unbiased evaluation of the final model, while
            cross validation is a process used for training and hyper parameters tuning of the model.
            Therefore, we should not use the test set for this purpose, if we would, we would have better results
            that don't indicate the quality of our model.
4. True -   In cross validation, each fold acts as a validation set once, while the other folds in the batch are
            used for training. Since the model was not trained on this fold, it can be used to estimate how the
            model is likely to generalize to unseen data, i.e., its generalization error.

                
"""

part1_q2 = r"""
Our friend's approach is not justified.
It is important to keep a separated dataset for each concern.
Our friend used the test set in the process of tuning the model, and thus harmed its validity
in serving the purpose of evaluating the final model and its ability to generalize to unseen
data (since the data is no longer unseen).
In addition, he should not have used the training set for the purpose of tuning lambda, since the
hyperparameters tuning and the model paramaters tuning are two different processes that can interfere
with each other, and also this may cause in overfitting to the training data.
There is a reason why the best practice is considered to be use of a separated validation set for the
purpose of hyperparameters tuning.

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
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
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
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
