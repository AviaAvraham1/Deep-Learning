import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss
from hw1.transforms import BiasTrick


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.
        self.weights = torch.normal(
            mean=0.0,
            std=weight_std,
            # n_features conatins bias
            size=(n_features, n_classes)
        )

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """
        #x_with_bias = BiasTrick()(x)

        class_scores = torch.matmul(x, self.weights)
        y_pred = torch.argmax(class_scores, dim=1)

        return y_pred, class_scores


    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.
        correct_predictions = (y == y_pred).sum().item()
        total_predictions = y.size(0) 
        acc = correct_predictions / total_predictions

        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0
            
            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            total_samples = 0

            for x_batch, y_batch in dl_train:
                # forward 
                y_pred, scores = self.predict(x_batch)
                loss = loss_fn.loss(x_batch, y_batch, scores, y_pred)

                # calculate regularization
                reg_term = weight_decay * (self.weights ** 2).sum() / 2
                loss += reg_term

                # loss with regularization
                loss_fn.grad_ctx["x"] = x_batch
                grads = loss_fn.grad() + weight_decay * self.weights
                self.weights -= learn_rate * grads  # gradient step

                # update stuff
                total_correct += (y_pred == y_batch).sum().item() # item converts from Tensor to python element
                average_loss += loss.item()
                total_samples += y_batch.shape[0]

            average_loss /= len(dl_train)
            train_res.accuracy.append(total_correct / total_samples * 100)
            train_res.loss.append(average_loss)

            # validation
            total_correct = 0
            average_loss = 0
            total_samples = 0

            for x_batch, y_batch in dl_valid:
                # calc loss
                y_pred, scores = self.predict(x_batch)
                loss = loss_fn.loss(x_batch, y_batch, scores, y_pred)
                
                # regularization
                reg_term = weight_decay * (self.weights ** 2).sum() / 2
                loss += reg_term

                # update more stuff
                total_correct += (y_pred == y_batch).sum().item()
                average_loss += loss.item()
                total_samples += y_batch.shape[0]

            average_loss /= len(dl_valid)
            valid_res.accuracy.append(total_correct / total_samples * 100)
            valid_res.loss.append(average_loss)

            print(".", end="")

        print("")
        return train_res, valid_res
            

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).
        weights_no_bias = self.weights[1:, :] if has_bias else self.weights
        expected_features = img_shape[0] * img_shape[1] * img_shape[2]
        assert weights_no_bias.shape[0] == expected_features, (
            f"Expected {expected_features} features, got {weights_no_bias.shape[0]}"
        )
        n_classes = weights_no_bias.shape[1]
        w_images = weights_no_bias.T.reshape(n_classes, *img_shape)

        return w_images


def hyperparams():
    hp = dict(weight_std=0.01, learn_rate=0.01, weight_decay=0.0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    # ========================

    return hp
