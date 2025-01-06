import functools
from typing import Callable, Tuple

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import auto_diff as ad


def logistic_regression(X: ad.Node, W: ad.Node, b: ad.Node) -> ad.Node:
    """Construct the computational graph of a logistic regression model.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, in_features), denoting the input data.
    W: ad.Node
        A node in shape (in_features, num_classes), denoting the the weight
        in logistic regression.
    b: ad.Node
        A node in shape (num_classes,), denoting the bias term in
        logistic regression.

    Returns
    -------
    logits: ad.Node
        The logits predicted for the batch of input.
        When evaluating, it should have shape (batch_size, num_classes).
    """
    """TODO: Your code here"""


def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Construct the computational graph of average softmax loss over
    a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        logits for the batch of instances.

    y_one_hot: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        one-hot encoding of the ground truth label for the batch of instances.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax loss over the batch.
        When evaluating, it should be a zero-rank array (i.e., shape is `()`).

    Note
    ----
    1. In this homework, you do not have to implement a numerically
    stable version of softmax loss.
    2. You may find that in other machine learning frameworks, the
    softmax loss function usually does not take the batch size as input.
    Try to think about why our softmax loss may need the batch size.
    """
    """TODO: Your code here"""


def sgd_epoch(
    f_run_model: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ],
    X: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    batch_size: int,
    lr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run an epoch of SGD for the logistic regression model
    on training data with regard to the given mini-batch size
    and learning rate.

    Parameters
    ----------
    f_run_model: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]
        The function to run the forward and backward computation
        at the same time for logistic regression model.
        It takes the training data, training label, model weight
        and bias as inputs, and returns the logits, loss value,
        weight gradient and bias gradient in order.
        Please check `f_run_model` in the `train_model` function below.

    X: np.ndarray
        The training data in shape (num_examples, in_features).

    y: np.ndarray
        The training labels in shape (num_examples,).

    W: np.ndarray
        The weight of the logistic regression model.

    b: np.ndarray
        The bias of the logistic regression model.

    batch_size: int
        The mini-batch size.

    lr: float
        The learning rate.

    Returns
    -------
    W_updated: np.ndarray
        The model weight after update in this epoch.

    b_updated: np.ndarray
        The model weight after update in this epoch.

    loss: np.ndarray
        The average training loss of this epoch.
    """
    """TODO: Your code here"""