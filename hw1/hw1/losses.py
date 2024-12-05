import abc
import torch


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

        # Create a matrix M where M[i,j] is the margin-loss for sample i and class j
        margins = x_scores - x_scores[torch.arange(x_scores.shape[0]), y].unsqueeze(1) + self.delta
        margins[torch.arange(x_scores.shape[0]), y] = 0  # Set margin to 0 for the true class

        # Calculate the hinge loss
        loss = torch.max(torch.relu(margins), dim=1)[0].mean()

        # Save what you need for gradient calculation in self.grad_ctx
        self.grad_ctx['x_scores'] = x_scores
        self.grad_ctx['y'] = y
        self.grad_ctx['margins'] = margins

        return loss


        N, C = x_scores.shape
        actual_label_scores = x_scores[torch.arange(N), y].unsqueeze(1) 
        margins = x_scores - actual_label_scores + self.delta  
        margins[torch.arange(N), y] = 0 
        hinge_loss = torch.clamp(margins, min=0)
        loss = hinge_loss.sum() / N
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

        grad = None
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

        return grad
