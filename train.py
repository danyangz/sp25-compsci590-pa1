import auto_diff as ad
from logistic_regression import *

def train_model():
    """Train a logistic regression model with handwritten digit dataset.

    Note
    ----
    Your implementation should NOT make changes to this function.
    """
    # - Set up the training settings.
    num_epochs = 100
    batch_size = 50
    lr = 0.05

    # - Define the forward graph.
    x = ad.Variable(name="x")
    W = ad.Variable(name="W")
    b = ad.Variable(name="b")
    y_predict = logistic_regression(x, W, b)
    # - Construct the backward graph.
    y_groundtruth = ad.Variable(name="y")
    loss = softmax_loss(y_predict, y_groundtruth, batch_size)
    grad_W, grad_b = ad.gradients(loss, nodes=[W, b])
    # - Create the evaluator.
    evaluator = ad.Evaluator([y_predict, loss, grad_W, grad_b])
    test_evaluator = ad.Evaluator([y_predict])

    # - Load the dataset.
    #   Take 80% of data for training, and 20% for testing.
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=0
    )
    num_classes = 10
    in_features = functools.reduce(lambda x1, x2: x1 * x2, digits.images[0].shape, 1)

    # - Initialize model weights.
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    W_val = np.random.uniform(-stdv, stdv, (in_features, num_classes))
    b_val = np.random.uniform(-stdv, stdv, (num_classes,))

    def f_run_model(X_val, y_val, W_val, b_val):
        """The function to compute the forward and backward graph.
        It returns the logits, loss, and gradients for model weights.
        """
        z_val, loss_val, grad_W_val, grad_b_val = evaluator.run(
            input_values={x: X_val, y_groundtruth: y_val, W: W_val, b: b_val}
        )
        return z_val, loss_val, grad_W_val, grad_b_val

    def f_eval_model(X_val, W_val, b_val):
        """The function to compute the forward graph only and returns the prediction."""
        logits = test_evaluator.run({x: X_val, W: W_val, b: b_val})
        return np.argmax(logits[0], axis=1)

    # - Train the model.
    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        W_val, b_val, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, W_val, b_val, batch_size, lr
        )

        # - Evaluate the model on the test data.
        predict_label = f_eval_model(X_test, W_val, b_val)
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label == y_test)}, "
            f"loss = {loss_val}"
        )

    # Return the final test accuracy.
    predict_label = f_eval_model(X_test, W_val, b_val)
    return np.mean(predict_label == y_test)


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")
