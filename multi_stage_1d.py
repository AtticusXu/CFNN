import os
import numpy as np
import tensorflow as tf
from typing import Callable, Tuple
import multiprocessing
from util import (
    CFNN,
    DNN,
    ExpPlus,
    MultiStageNN,
    save_results,
    make_plots,
    make_combined_plot,
    make_compare_plot,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def generate_1d_data(
    func: Callable, n_train: int = 3000, n_test: int = 10000, dtype=tf.float64
) -> Tuple[tf.Tensor, ...]:
    """Generate 1D training and test data.

    Args:
        func: The function to approximate
        n_train: Number of training points
        n_test: Number of test points
        dtype: TensorFlow data type to use (default: tf.float64)

    Returns:
        t_train: Training input points
        x_train: Training target values
        t_test: Test input points
        x_test: Test target values
    """
    # Training data
    t_train = np.linspace(-1.1, 1.1, n_train)[:, None]
    t_train = tf.cast(t_train, dtype=dtype)
    x_train = func(t_train)

    # Test data

    t_test = np.linspace(-1, 1, n_test)[:, None]
    t_test = tf.cast(t_test, dtype=dtype)
    x_test = func(t_test)

    return t_train, x_train, t_test, x_test


def f1(t: tf.Tensor) -> tf.Tensor:
    """f1(x) = x"""
    return t


def f2(t: tf.Tensor) -> tf.Tensor:
    """f2(x) = sin(2x+1) + 0.2exp(1.3x)"""
    return tf.sin(2 * t + 1) + 0.2 * tf.exp(1.3 * t)


def f3(t: tf.Tensor) -> tf.Tensor:
    """f3(x) = |sin(πx)|^2"""
    return tf.square(tf.abs(tf.sin(np.pi * t)))


def f4(t: tf.Tensor, m: float = 30.0) -> tf.Tensor:
    """f4(x) = (1-x^2/2)cos(m(x+0.5x^3))"""
    return (1 - t**2 / 2) * tf.cos(m * (t + 0.5 * t**3))


def f5(t: tf.Tensor) -> tf.Tensor:
    """f5(x) = |x|"""
    return tf.abs(t)


def f6(t: tf.Tensor) -> tf.Tensor:
    """f6(x) = sign(x)"""
    return tf.sign(t)


def train_1d_example(
    func: Callable,
    func_name: str,
    nn_type: str = "CFNN",
    n_stages: int = 4,
    hidden_layers: int = 3,
    neurons: int = 30,
    data_type: tf.DType = tf.float64,
    base_dir: str = "./results",
) -> None:
    """Train CFNN on 1D example.

    Args:
        func: Function to approximate
        func_name: Name of the function (for saving results)
        nn_type: Type of neural network to use (CFNN or DNN)
        n_stages: Number of training stages
        hidden_layers: Number of hidden layers
        neurons: Number of neurons per layer
        dtype: TensorFlow data type to use
    """
    # Generate data
    t_train, x_train, t_test, x_test = generate_1d_data(func, dtype=data_type)

    # Define network architecture
    layers = [1] + [neurons] * hidden_layers + [1]

    if n_stages == 1:
        # ablation study for 1 stage training
        iters_adam = [20000]
        iters_lbfgs = [80000]
        fl_ini = ExpPlus(1.0 / 5.0**3, 0.0, data_type)
    else:
        iters_adam = [5000, 5000, 5000, 5000]
        iters_lbfgs = [20000, 20000, 20000, 20000]
        fl_ini = ExpPlus(5.0, 0.0, data_type)

    # Training parameters
    adam_lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01, decay_steps=100, decay_rate=0.97, staircase=True
    )

    # Initialize first stage

    if nn_type == "CFNN":
        nn = CFNN(layers, fl_ini, data_type)
    elif nn_type == "DNN":
        # ablation study for DNN
        nn = DNN(layers, data_type)
    model = MultiStageNN(
        t_train, x_train, nn, tf.reduce_min(t_train), tf.reduce_max(t_train), adam_lr
    )

    # Train first stage
    model.train(iters_adam[0], mode=1)  # Adam
    model.train(iters_lbfgs[0], mode=2)  # L-BFGS

    # Store results
    models = [model]
    preds_train = [model.predict(t_train)]
    preds_test = [model.predict(t_test)]
    losses = [[l.numpy() for l in model.loss]]

    # Train subsequent stages
    for i in range(1, n_stages):
        # Calculate residual
        x_train_new = x_train - sum(preds_train)

        # Initialize new stage
        fl_ini_new = ExpPlus(5.0 / 5**i, 10 * 6 ** (i - 1), data_type)
        if nn_type == "CFNN":
            nn_new = CFNN(layers, fl_ini_new, data_type)
        elif nn_type == "DNN":
            nn_new = DNN(layers, data_type)
        model_new = MultiStageNN(
            t_train,
            x_train_new,
            nn_new,
            tf.reduce_min(t_train),
            tf.reduce_max(t_train),
            adam_lr,
        )

        # Train new stage
        model_new.train(iters_adam[i], mode=1)  # Adam
        model_new.train(iters_lbfgs[i], mode=2)  # L-BFGS

        # Store results
        models.append(model_new)
        preds_train.append(model_new.predict(t_train))
        preds_test.append(model_new.predict(t_test))
        losses.append([l.numpy() for l in model_new.loss])
    # Compute final prediction and error
    final_pred_test = sum(preds_test)
    error_test = tf.norm(x_test - final_pred_test) / tf.norm(x_test)
    print(f"Function {func_name} - test relative L2 Error: {error_test:.2e}")

    final_pred_train = sum(preds_train)
    error_train = tf.norm(x_train - final_pred_train) / tf.norm(x_train)
    print(f"Function {func_name} - train relative L2 Error: {error_train:.2e}")

    # Save results
    save_results(
        func_name,
        t_train,
        x_train,
        t_test,
        x_test,
        preds_train,
        preds_test,
        losses,
        base_dir=base_dir,
    )
    make_plots(
        func_name,
        base_dir=base_dir,
    )
    make_combined_plot(
        func_name,
        base_dir=base_dir,
    )


def main():
    """Run all 1D examples in parallel using multiprocessing."""
    np.random.seed(42)
    tf.random.set_seed(42)

    base_dir = "./results_1d"
    functions = [
        (f1, "f1_linear", "CFNN", 4, 3, 40, tf.float64, base_dir),
        (f2, "f2_sinexp", "CFNN", 4, 3, 40, tf.float64, base_dir),
        (f3, "f3_sinsquare", "CFNN", 4, 3, 40, tf.float64, base_dir),
        (f4, "f4_oscillatory", "CFNN", 4, 3, 40, tf.float64, base_dir),
        (f4, "f4_DNN", "DNN", 4, 3, 40, tf.float64, base_dir),
        (f4, "f4_1stage", "CFNN", 1, 3, 160, tf.float64, base_dir),
        (f5, "f5_absolute", "CFNN", 4, 3, 40, tf.float64, base_dir),
        (f6, "f6_sign", "CFNN", 4, 3, 40, tf.float64, base_dir),
    ]

    # Create process pool and run tasks in parallel
    with multiprocessing.Pool(processes=len(functions)) as pool:
        pool.starmap(train_1d_example, functions)
    make_compare_plot(["f4_DNN", "f4_1stage"], base_dir=base_dir)


if __name__ == "__main__":
    main()
