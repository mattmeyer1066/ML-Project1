import numpy as np

# ============================================================
# Helper functions
# ============================================================


def applyW(tx, w):
    """
    Applies the weight vector to the input data.

    Args:
        tx: Input data matrix, shape (N, D)
        w: Weight vector, shape (D,)

    Returns:
        Predictions, shape (N,)
    """
    return tx @ w


def compute_error(y, pred):
    """
    Computes residual errors.

    Args:
        y: True target values, shape (N,)
        pred: Predicted values, shape (N,)

    Returns:
        Residuals y - pred
    """
    return y - pred


def compute_mse(y, tx, w):
    """
    Computes the mean squared error:
        (1 / 2N) * sum((y - Xw)^2)

    Args:
        y: Target values, shape (N,)
        tx: Input data matrix, shape (N, D)
        w: Weight vector, shape (D,)

    Returns:
        Mean squared error
    """
    pred = applyW(tx, w)
    error = compute_error(y, pred)

    return 0.5 * np.mean(error**2)


def sigmoid(t):
    """
    Applies the sigmoid function element-wise.

    Args:
        t: Scalar or NumPy array

    Returns:
        Sigmoid of t
    """
    return 1 / (1 + np.exp(-t))


def compute_logistic_loss(y, tx, w):
    """
    Computes logistic loss without regularization.

    Args:
        y: Target values in {0, 1}, shape (N,)
        tx: Input data matrix, shape (N, D)
        w: Weight vector, shape (D,)

    Returns:
        Logistic loss
    """
    z = applyW(tx, w)
    pred = sigmoid(z)

    # Avoid log(0)
    eps = 1e-15
    pred = np.clip(pred, eps, 1 - eps)

    return -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))


def formatOutput(w, loss):
    """
    Formats output according to project requirements.

    Args:
        w: Weight vector
        loss: Final loss value

    Returns:
        Tuple:
            w as a 1D NumPy array
            loss as a float
    """
    w = np.asarray(w).reshape(-1)
    loss = np.asarray(loss)

    return w, loss


# ============================================================
# 1. Mean Squared Error - Gradient Descent
# ============================================================


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Performs linear regression using gradient descent.
    """

    def compute_mse_gradient(y, tx, w):
        pred = applyW(tx, w)
        error = compute_error(y, pred)

        return -(tx.T @ error) / len(y)

    w = np.asarray(initial_w).copy().reshape(-1)

    for i in range(max_iters):
        gradient = compute_mse_gradient(y, tx, w)
        w = w - gamma * gradient

    loss = compute_mse(y, tx, w)

    return formatOutput(w, loss)


# ============================================================
# 2. Mean Squared Error - Stochastic Gradient Descent
# ============================================================


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Performs linear regression using stochastic gradient descent.
    Uses one randomly selected datapoint per iteration.
    """

    def compute_sgd_gradient(y_n, tx_n, w):
        pred = tx_n @ w
        error = y_n - pred

        return -tx_n * error

    w = np.asarray(initial_w).copy().reshape(-1)

    n = len(y)

    for i in range(max_iters):
        # Randomly select one datapoint
        index = np.random.randint(n)

        tx_n = tx[index]
        y_n = y[index]

        gradient = compute_sgd_gradient(y_n, tx_n, w)

        w = w - gamma * gradient

    # Final loss is computed on the full dataset
    loss = compute_mse(y, tx, w)

    return formatOutput(w, loss)


# ============================================================
# 3. Least Squares
# ============================================================


def least_squares(y, tx):
    """
    Computes linear regression using the normal equations.
    """

    return ridge_regression(y, tx, 0)


# ============================================================
# 4. Ridge Regression
# ============================================================


def ridge_regression(y, tx, lambda_):
    """
    Computes ridge regression using the normal equations.
    """

    n = len(y)
    d = tx.shape[1]

    # Identity matrix of size D x D
    identity = np.eye(d)

    # Left-hand side:
    # X^T X + 2N lambda I
    a = tx.T @ tx + 2 * n * lambda_ * identity

    # Right-hand side:
    # X^T y
    b = tx.T @ y

    # Solve ridge normal equation
    w = np.linalg.solve(a, b)

    # Return only the MSE, without regularization penalty
    loss = compute_mse(y, tx, w)

    return formatOutput(w, loss)


# ============================================================
# 5. Logistic Regression
# ============================================================


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Performs logistic regression using gradient descent.
    """

    return reg_logistic_regression(y, tx, 0, initial_w, max_iters, gamma)


# ============================================================
# 6. Regularized Logistic Regression
# ============================================================


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Performs regularized logistic regression using gradient descent.
    """

    def compute_reg_logistic_gradient(y, tx, w):
        pred = sigmoid(applyW(tx, w))

        # Gradient of logistic loss
        logistic_gradient = (tx.T @ (pred - y)) / len(y)

        # Gradient of lambda * ||w||^2
        regularization_gradient = 2 * lambda_ * w

        return logistic_gradient + regularization_gradient

    w = np.asarray(initial_w).copy().reshape(-1)

    for i in range(max_iters):
        gradient = compute_reg_logistic_gradient(y, tx, w)

        w = w - gamma * gradient

    # Return logistic loss only,
    # without regularization penalty
    loss = compute_logistic_loss(y, tx, w)

    return formatOutput(w, loss)
