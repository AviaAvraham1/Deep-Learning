import abc
import torch
from hw1.transforms import BiasTrick


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        N, C = x_scores.shape
        correct_class_scores = x_scores[range(N), y]
        M = x_scores - correct_class_scores[:, None] + self.delta
        M[range(N), y] = 0
        hinge_loss = (M > 0) * M 
        loss = hinge_loss.sum() / N

        # Save what you need for gradient calculation in self.grad_ctx
        self.grad_ctx = {
            "margins": hinge_loss,
            "x": x,
            "y": y,
            "x_scores": x_scores,
        }

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        margins = self.grad_ctx["margins"]
        x = self.grad_ctx["x"]
        #x_with_bias = BiasTrick()(x)
        
        y = self.grad_ctx["y"]
        N, C = margins.shape
        G = (margins > 0).float()
        G[range(N), y] = -G.sum(dim=1)
        grad = torch.matmul(x.T, G) / N
        
        return grad
