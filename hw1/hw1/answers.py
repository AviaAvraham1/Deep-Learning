r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1. False -  In-sample error is the error rate of a model when it is evaluated on the same data it was trained on. 
            Thus, the test set, which is independent of the training set, can't help with evaluating this error.
            For example, it is possible that the test error will be very high, but the training error
            (the in-sample error) will be very low or even 0, if we overfitted to the training set,
              therefore, the test set in no way helps us estimate this error.
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
**Your answer:**
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
If delta is not zero, the model will require separation between the score for the correct class and the scores
for incorrect classes, and thus will be more strict in its classification requirements.
As a result, it will become harder to satisfy loss conditions and we may find that it converges more slowly,  
but as an advantage, we may find that the model generalizes better (compared to when delta=0) since we reduced 
sensitivity to small variations in the input data by forcing the decision boundary to push further away from the
training samples.

"""

part2_q2 = r"""
**Your answer:**
The images visualize the learned weights of each class's model but as we can see, for each class the
visualization looks roughly like the digit itself, represented by the class. We can infer that what 
the model actually learns is the general shape of the digit (represented by the bright areas),
i.e where it expects to find the white pixels in images that belong to that class.

As for the errors, we often get those for samples that can be considered as outliars - samples where the digit
looks distinctively different compared to how the digit usually looks like (its "avrerage look"), and in some 
cases can be confused with another digit.
For example, in the first row, the misclassifed 5 can be identified as a 6, like the model predicted, because it visually resembles it.
In addition, classification errors may rise due to overlap in areas between classes, or noise in the learned weights
For example, in the third line, the 4 is misclassifed as 6, and the areas in which the 4 is in are contained within the 6.
 
"""

part2_q3 = r"""
**Your answer:**
1. Based on the graph of the training set loss, we'd say that the learning rate we chose 
(via trial and error) is good.
If the learning rate was too high (which it was initially during our trial and error), we'd expect to see noise \ spikes in the
curve, meaning that we took a too big step each time and thus had to go back and forth.
If the learning rate was too low, we'd see that the graph is decreasing very slowly, and it'd require 
a larger number of epochs to converge to a good and low enough loss value, more than the number of epochs we use.

2. Slightly overfitted to the training set
We see a fairly low rate of loss, and high rate of accuracy, which means we're not in an underfit scenario.
We see a difference between the curve of our train and the curve of the validation set, in favor of the train set,
which means we're in an overfit scenerio (since we can't generlaize that well to unseen data).
However, that difference isn't too big, so it isn't highly overfitted too.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

The ideal pattern to see in residuals plot is that the all the dots are as scattered
as poosible around the 0 line, since it means that the distance between the predicted y and the actual y is small.
Also, regarding the dotted red lines, we would wish to see most of the dots between these two lines, that kind 
of represents the error range we are willing to accept (in this case they represents the standard deviation which 
is used to sort of quantifiy the typical size of the error).

If we compare the top-5 features with the final plot after CV, we can see that we indeed managed to get the dots 
more scattered around the 0 and we can also see much less dots outside the red dotted lines.

"""

part3_q2 = r"""
**Your answer:**

1. Yes, it is still a linear regression model, since in "linear" we refer to the linearity
of the model in its parameters, not the features, and this is something we havn't changed 
(we still predict by computing: $y = \mathbf{w}^T \mathbf{x} + b$)

2. No, while adding non-linear features does give as the ability to fit non-linear functions, 
there are still some non-linear functions that we cannot fit, since we are limited to fitting 
functions that can be represented as a linear combination of the non-linear features we have added.
    
    (for example, if we are limited to polynomial feature mapping, we cannot fit a sinus functtion, 
    and with any feature mapping we can't fit a non-computational function such as the function that
    calculates $\overline{\text{HP}}$)

3. Adding non-linear features allows the model to fit more complex patterns in the data,
as it can now fit non-linear decision boundaries, such as circles and graph of different polynimals.
When non-linear features are added, the decision boundary becomes linear **in the extended feature space**,
meaning it is a hyperplane but is becoming non-linear in the original feature space, that is because it's
determined according to features which are non-linear in the original features.

"""

part3_q3 = r"""
**Your answer:**

1. Lambda has a "non-liner effect": small changes in its values can have a large impact when lambda
is small but changes in larger lambda values may have smaller effect. The logarithmic spacing between
the values serves well this effect and also allows for a more effective exploration of a wide range of values.

For example, we'd want to check values like (0.001, 0.01, 0.1, 1, 10, 100, 1000), because they are close in their
small values and farther in their large ones.
However, if we used linear in that range, like (0.001, 100, 200, ... 1000), we'd miss out on most of the values
the logaritmic method check, and there might be a large differnce between those small values

2. The total number of times the model was fitted to data during cross-validation can be calculated by
considering the number of combinations of hyperparameters and the number of folds in cross-validation:
degree_range: 3
lambda_range: 20
k_folds: 3
total of times the model was fitted: 3*20*3 = 180.

"""

# ==============

# ==============
