import os
from scipy.io import savemat, loadmat

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import time


class SinNN:
    def __init__(self, layers, kappa, datatype, acts=0):
        self.layers = layers
        self.num_layers = len(layers) - 1
        self.kappa = kappa
        # Decide activation for the first layer based on 'acts'
        if acts == 0:
            first_activation = tf.tanh
        elif acts == 1:
            first_activation = lambda x: tf.sin(self.kappa * x)

        # List to hold all layer configurations including activation functions
        self.activations = (
            [first_activation] + [tf.tanh] * (self.num_layers - 2) + [tf.identity]
        )  # None for the last layer (linear output)

        self.datatype = datatype
        self.weights, self.biases = self.initialize_NN(layers)

    def initialize_NN(self, layers):
        weights = []
        biases = []

        for l in range(self.num_layers):
            W = self.MPL_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=self.datatype))
            weights.append(W)
            biases.append(b)
        return weights, biases

    def MPL_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(
            tf.random.truncated_normal(
                [in_dim, out_dim], stddev=xavier_stddev, dtype=self.datatype
            )
        )

    def forward(self, X):
        H = X
        for l in range(self.num_layers):
            W = self.weights[l]
            b = self.biases[l]
            H = self.activations[l](tf.add(tf.matmul(H, W), b))

        return H

    @property
    def trainable_variables(self):
        return self.weights + self.biases


class CFNN:
    def __init__(self, layers, fl_ini, datatype):
        self.layers = layers
        self.num_layers = len(layers) - 1
        self.fl_ini = fl_ini
        self.datatype = datatype
        self.weights, self.biases = self.initialize_NN(layers)

        self.activations = [tf.tanh] * (self.num_layers - 2) + [tf.identity]

    def initialize_NN(self, layers):
        weights = []
        biases = []

        weights.append(
            tf.Variable(
                self.fl_ini.sample(sample_shape=(layers[0], layers[1])),
                dtype=self.datatype,
            )
        )
        for l in range(1, self.num_layers):
            W = self.MPL_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=self.datatype))
            weights.append(W)
            biases.append(b)

        return weights, biases

    def MPL_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(
            tf.random.truncated_normal(
                [in_dim, out_dim], stddev=xavier_stddev, dtype=self.datatype
            )
        )

    def forward(self, X):
        A = tf.math.acos(X)
        W = self.weights[0]
        H = tf.cos(tf.matmul(A, W))

        for l in range(1, self.num_layers):
            W = self.weights[l]
            b = self.biases[l - 1]
            H = self.activations[l - 1](tf.add(tf.matmul(H, W), b))

        return H

    def count_trainable_variables(self):
        total_parameters = 0
        for variable in self.trainable_variables:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim
            total_parameters += variable_parameters
        print(total_parameters)

    @property
    def trainable_variables(self):
        return self.weights + self.biases


class DNN:
    def __init__(self, layers, datatype):
        self.layers = layers
        self.num_layers = len(layers) - 1
        self.datatype = datatype
        self.weights, self.biases = self.initialize_NN(layers)

        # All layers use tanh except the last one which uses identity
        self.activations = [tf.tanh] * (self.num_layers - 1) + [tf.identity]

    def initialize_NN(self, layers):
        weights = []
        biases = []

        # Initialize all layers using MPL_init
        for l in range(self.num_layers):
            W = self.MPL_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=self.datatype))
            weights.append(W)
            biases.append(b)

        return weights, biases

    def MPL_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(
            tf.random.truncated_normal(
                [in_dim, out_dim], stddev=xavier_stddev, dtype=self.datatype
            )
        )

    def forward(self, X):
        H = X
        # Standard feedforward for all layers
        for l in range(self.num_layers):
            W = self.weights[l]
            b = self.biases[l]
            H = self.activations[l](tf.add(tf.matmul(H, W), b))

        return H

    def count_trainable_variables(self):
        total_parameters = 0
        for variable in self.trainable_variables:
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim
            total_parameters += variable_parameters
        print(total_parameters)

    @property
    def trainable_variables(self):
        return self.weights + self.biases


class ExpPlus:
    def __init__(self, rate, const=0.0, dtype=tf.float32, seed=42):

        self.dtype = dtype
        self.const = tf.constant(const, dtype=self.dtype)
        self.rate = tf.constant(rate, dtype=self.dtype)
        self.seed = seed

    def sample(self, sample_shape):
        Exp = tfp.distributions.Exponential(self.rate).sample(
            sample_shape, seed=self.seed
        )
        const = tf.constant(self.const, dtype=self.dtype, shape=sample_shape)

        # print(Exp+const)
        return Exp + const


class MultiStageNN:
    def __init__(self, t_u, x_u, NN, lt, ut, adam_lr):
        #### Changed by CY
        self.dim = t_u.shape[-1]
        self.scale = tf.math.sqrt(tf.math.reduce_mean(x_u**2, axis=0))
        x_u2 = x_u / self.scale

        self.x_u = x_u2
        self.NN = NN
        self.train_variables = self.NN.trainable_variables
        self.lt = lt
        self.ut = ut

        self.t_u = 1.998 * (t_u - self.lt) / (self.ut - self.lt) - 0.999
        #### Changed by CY
        self.weights_cheb = (
            1 / tf.sqrt(self.dim - tf.math.reduce_sum(self.t_u**2, axis=-1))
        )[
            :, None
        ]  # Chebyshev weights

        self.loss0 = self.scale**2
        self.lr_schedule = adam_lr
        self.optimizer_Adam = tf.optimizers.Adam(learning_rate=self.lr_schedule)
        self.loss = []

    def neural_net(self, X):
        return self.NN.forward(X)

    # @tf.function
    # def loss_NN(self):
    #     self.x_pred = self.neural_net(self.t_u)
    #     loss = tf.reduce_mean(tf.square(self.x_u - self.x_pred))
    #     return loss
    @tf.function
    def loss_NN(self):
        self.x_pred = self.neural_net(self.t_u)
        #### Changed by CY
        weights = self.weights_cheb
        squared_errors = tf.square(self.x_u - self.x_pred)  # Compute squared errors
        weighted_squared_errors = weights * squared_errors  # Apply weights
        loss = tf.reduce_mean(
            weighted_squared_errors
        )  # Weighted mean of squared errors
        return loss

    """
    Functions used to define ADAM optimizers
    ===============================================================
    """

    # define the function to apply the ADAM optimizer
    def adam_function(self):
        @tf.function
        def f():
            # calculate the loss
            loss_norm = self.loss_NN()
            #### Changed by CY
            loss_value = tf.reduce_sum(loss_norm * self.loss0)
            # store loss value so we can retrieve later
            tf.py_function(f.loss.append, inp=[loss_value], Tout=[])

            # print out iteration & loss
            f.iter.assign_add(1)

            str_iter = tf.strings.as_string([f.iter])
            str_loss = tf.strings.as_string([loss_value], precision=4, scientific=True)

            str_print = tf.strings.join(
                ["Iter: ", str_iter[0], ", loss: ", str_loss[0]]
            )
            tf.cond(
                f.iter % 100 == 0,
                lambda: tf.print(str_print),
                lambda: tf.constant(True),  # return arbitrary for non-printing case
            )
            return loss_norm

        f.iter = tf.Variable(0)
        f.term = []
        f.loss = []
        return f

    def Adam_optimizer(self, nIter):
        varlist = self.train_variables
        func_adam = self.adam_function()
        for it in range(nIter):
            self.optimizer_Adam.minimize(func_adam, varlist)
        return func_adam

    """
    Functions used to define L-BFGS optimizers
    ===============================================================
    """

    # A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    def Lbfgs_function(self, varlist):
        # obtain the shapes of all trainable parameters in the model
        shapes = tf.shape_n(varlist)
        n_tensors = len(shapes)

        # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
        # prepare required information first
        count = 0
        idx = []  # stitch indices
        part = []  # partition indices

        self.start_time = time.time()

        for i, shape in enumerate(shapes):
            n = np.prod(shape)
            idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
            part.extend([i] * n)
            count += n

        part = tf.constant(part)

        def assign_new_model_parameters(params_1d):
            # A function updating the model's parameters with a 1D tf.Tensor.
            # Sub-function under function of class not need to input self

            params = tf.dynamic_partition(params_1d, part, n_tensors)
            for i, (shape, param) in enumerate(zip(shapes, params)):
                varlist[i].assign(tf.reshape(param, shape))

        @tf.function
        def f(params_1d):
            # A function that can be used by tfp.optimizer.lbfgs_minimize.
            # This function is created by function_factory.
            # Sub-function under function of class not need to input self

            # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
            with tf.GradientTape() as tape:
                # update the parameters in the model
                # this step is critical for self-defined function for L-BFGS
                assign_new_model_parameters(params_1d)
                # calculate the loss
                loss_norm = self.loss_NN()
                #### Changed by CY
                loss_value = tf.reduce_sum(loss_norm * self.loss0)

            # calculate gradients and convert to 1D tf.Tensor
            grads = tape.gradient(loss_norm, varlist)
            grads = tf.dynamic_stitch(idx, grads)

            # store loss value so we can retrieve later
            tf.py_function(f.loss.append, inp=[loss_value], Tout=[])

            # print out iteration & loss
            f.iter.assign_add(1)

            str_iter = tf.strings.as_string([f.iter])
            str_loss = tf.strings.as_string([loss_value], precision=4, scientific=True)

            str_print = tf.strings.join(
                ["Iter: ", str_iter[0], ", loss: ", str_loss[0]]
            )
            tf.cond(
                f.iter % 100 == 0,
                lambda: tf.print(str_print),
                lambda: tf.constant(True),  # return arbitrary for non-printing case
            )

            return loss_value, grads

        # store these information as members so we can use them outside the scope
        f.iter = tf.Variable(0)
        f.idx = idx
        f.part = part
        f.shapes = shapes
        f.assign_new_model_parameters = assign_new_model_parameters
        f.loss = []

        return f

    # define the function to apply the L-BFGS optimizer
    def Lbfgs_optimizer(self, nIter, varlist):

        func = self.Lbfgs_function(varlist)

        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(func.idx, varlist)

        max_nIter = tf.cast(nIter / 3, dtype=tf.int32)

        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func,
            initial_position=init_params,
            tolerance=1e-11,
            max_iterations=max_nIter,
        )

        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
        func.assign_new_model_parameters(results.position)

        return func

    """
    Function used for training the model
    ===============================================================
    """

    def train(self, nIter, mode):
        if mode == 1:
            # mode 1: running the Adam optimization
            func_adam = self.Adam_optimizer(nIter)
            self.loss += func_adam.loss
        elif mode == 2:
            # mode 2: running the Lbfgs optimization
            func_bfgs = self.Lbfgs_optimizer(nIter, self.train_variables)
            self.loss += func_bfgs.loss

    # @tf.function
    def predict(self, t):
        t = 1.998 * (t - self.lt) / (self.ut - self.lt) - 0.999
        x_p = self.neural_net(t) * self.scale
        return x_p


def save_results(
    example_name: str,
    t_train: tf.Tensor,
    x_train: tf.Tensor,
    t_test: tf.Tensor,
    x_test: tf.Tensor,
    preds_train: list,
    preds_test: list,
    losses: list,
    base_dir: str = "./results",
) -> None:
    """Save training results to files.

    Args:
        example_name: Name of the example/function
        t_train: Training input points
        x_train: Training target values
        t_test: Test input points
        x_test: Test target values
        preds_train: List of predictions for training data at each stage
        preds_test: List of predictions for test data at each stage
        losses: List of loss values during training
        base_dir: Base directory for saving results
    """

    # Create directory for this example
    save_dir = os.path.join(base_dir, example_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save data as .mat file for MATLAB compatibility
    savemat(
        os.path.join(save_dir, "results.mat"),
        {
            "t_train": t_train.numpy(),
            "x_train": x_train.numpy(),
            "t_test": t_test.numpy(),
            "x_test": x_test.numpy(),
            "preds_train": [p.numpy() for p in preds_train],
            "preds_test": [p.numpy() for p in preds_test],
            "losses": losses,
        },
    )


def make_plots(example_name: str, base_dir: str = "./results"):
    """Create plots from saved training results.

    Args:
        example_name: Name of the example/function
        base_dir: Base directory where results are saved
    """

    # Load the saved results
    results = loadmat(os.path.join(base_dir, example_name, "results.mat"))

    # Extract data
    t_train = results["t_train"].squeeze()
    x_train = results["x_train"].squeeze()
    t_test = results["t_test"].squeeze()
    x_test = results["x_test"].squeeze()
    preds_train = [pred.squeeze() for pred in results["preds_train"]]
    preds_test = [pred.squeeze() for pred in results["preds_test"]]
    losses = results["losses"].squeeze()

    # 1. Loss Plot
    actual_iters = [len(stage_losses.flatten()) for stage_losses in losses]
    print(actual_iters)
    plt.figure(figsize=(12, 6))
    # Flatten and concatenate all loss arrays
    all_losses = np.concatenate([stage_loss.flatten() for stage_loss in losses])
    plt.semilogy(all_losses)

    # Add vertical lines at the end of each stage
    for i in range(len(actual_iters) - 1):  # Don't plot line for last stage
        plt.axvline(x=sum(actual_iters[: i + 1]), color="r", linestyle="--", alpha=0.5)

    # Add stage labels
    for i in range(len(actual_iters)):
        start = 0 if i == 0 else sum(actual_iters[:i])
        end = sum(actual_iters[: i + 1])
        mid_point = start + (end - start) // 2
        plt.text(
            mid_point,
            plt.ylim()[0] + 0.1,
            f"$\\mathrm{{Stage}}~{i}$",
            horizontalalignment="center",
            fontsize=14,
        )

    plt.xlabel("$\\mathrm{Epochs}$", fontsize=14)
    plt.ylabel("$\\mathrm{Loss}$", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, example_name, "loss_history.pdf"), dpi=600)
    plt.close()

    # 2. Training Results Plot
    fig, axes = plt.subplots(5, 1, figsize=(12, 15))
    axes = axes.flatten()

    # Plot each stage's residual and network output
    for i in range(4):
        # Filter and sort points between -1 and 1
        mask = (t_train >= -1) & (t_train <= 1)
        t_plot = t_train[mask]
        sort_idx = np.argsort(t_plot)
        t_plot = t_plot[sort_idx]

        # Calculate residual for this stage
        residual = x_train - sum(preds_train[:i]) if i > 0 else x_train
        residual_plot = residual[mask][sort_idx]
        pred_plot = preds_train[i][mask][sort_idx]

        axes[i].plot(
            t_plot,
            residual_plot,
            "b-",
            linewidth=2,
            label="$f(x)$" if i == 0 else "$E_{%d}(x)$" % i,
        )
        axes[i].plot(
            t_plot,
            pred_plot,
            "r--",
            linewidth=2,
            label=(
                "$N_{\mathtt{CF}}^{(0)}(x)$"
                if i == 0
                else "$\epsilon_{%d}N_{\mathtt{CF}}^{(%d)}(x)$" % (i, i)
            ),
        )
        axes[i].legend(fontsize=12, loc="upper right")
        axes[i].set_xlim([-1, 1])
        axes[i].tick_params(axis="x", labelsize=12)
        axes[i].tick_params(axis="y", labelsize=12)

    # Plot final residual E_4
    mask = (t_train >= -1) & (t_train <= 1)
    t_plot = t_train[mask]
    sort_idx = np.argsort(t_plot)
    t_plot = t_plot[sort_idx]
    final_residual = x_train - sum(preds_train)
    final_residual_plot = final_residual[mask][sort_idx]

    axes[4].plot(t_plot, final_residual_plot, "b-", linewidth=2, label="$E_{4}(x)$")
    axes[4].legend(fontsize=12, loc="upper right")
    axes[4].set_xlim([-1, 1])
    axes[4].tick_params(axis="x", labelsize=12)
    axes[4].tick_params(axis="y", labelsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, example_name, "training_results.pdf"), dpi=600)
    plt.close()

    # 3. Test Results Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    # Final prediction (sum of all stages)
    final_pred_test = sum(preds_test)

    sort_idx = np.argsort(t_test)
    t_test = t_test[sort_idx]
    x_test = x_test[sort_idx]
    final_pred_test = final_pred_test[sort_idx]
    # Plot test results
    ax1.plot(t_test, x_test, "b-", linewidth=2, label="Ground Truth")
    ax1.plot(t_test, final_pred_test, "r--", linewidth=2, label="Prediction")
    ax1.set_xlim([-1.0, 1.0])
    ax1.tick_params(axis="x", labelsize=12)
    ax1.tick_params(axis="y", labelsize=12)
    ax1.legend(fontsize=12, loc="upper right")

    # Plot error
    error = x_test.squeeze() - final_pred_test
    ax2.plot(t_test, error, "k-", linewidth=2, label="Error")
    ax2.set_xlim([-1.0, 1.0])
    ax2.tick_params(axis="x", labelsize=12)
    ax2.tick_params(axis="y", labelsize=12)
    ax2.legend(fontsize=12, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, example_name, "test_results.pdf"), dpi=600)
    plt.close()


def make_combined_plot(example_name: str, base_dir: str = "./results"):
    # Load and extract data (keep existing code)
    results = loadmat(os.path.join(base_dir, example_name, "results.mat"))
    t_train = results["t_train"].squeeze()
    x_train = results["x_train"].squeeze()
    t_test = results["t_test"].squeeze()
    x_test = results["x_test"].squeeze()
    preds_train = [pred.squeeze() for pred in results["preds_train"]]
    preds_test = [pred.squeeze() for pred in results["preds_test"]]
    losses = results["losses"].squeeze()

    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 12))
    gs = plt.GridSpec(3, 2, height_ratios=[1, 0.8, 0.2])

    # 1. Loss Plot (top left)
    ax_loss = fig.add_subplot(gs[0, 0])
    all_losses = np.concatenate([stage_loss.flatten() for stage_loss in losses])
    ax_loss.semilogy(all_losses)

    actual_iters = [len(stage_losses.flatten()) for stage_losses in losses]
    for i in range(len(actual_iters) - 1):
        ax_loss.axvline(
            x=sum(actual_iters[: i + 1]), color="r", linestyle="--", alpha=0.5
        )

    for i in range(len(actual_iters)):
        start = 0 if i == 0 else sum(actual_iters[:i])
        end = sum(actual_iters[: i + 1])
        mid_point = start + (end - start) // 2
        ax_loss.text(
            mid_point,
            ax_loss.get_ylim()[0] + 0.08,
            f"$\\mathrm{{Stage}}~{i}$",
            horizontalalignment="center",
            fontsize=14,
        )

    ax_loss.set_xlabel("$\\mathrm{Epochs}$", fontsize=14)
    ax_loss.set_ylabel("$\\mathrm{Loss}$", fontsize=14)
    ax_loss.tick_params(labelsize=12)
    ax_loss.set_title("Training Loss", fontsize=14)

    # 2. Test Results Plot (bottom left)
    gs_test = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1:, 0])
    ax_test1 = fig.add_subplot(gs_test[0])
    ax_test2 = fig.add_subplot(gs_test[1])

    final_pred_test = sum(preds_test)
    sort_idx = np.argsort(t_test)
    t_test_sorted = t_test[sort_idx]
    x_test_sorted = x_test[sort_idx]
    final_pred_test_sorted = final_pred_test[sort_idx]

    ax_test1.plot(t_test_sorted, x_test_sorted, "b-", linewidth=2, label="Ground Truth")
    ax_test1.plot(
        t_test_sorted, final_pred_test_sorted, "r--", linewidth=2, label="Prediction"
    )
    ax_test1.set_xlim([-1.0, 1.0])
    ax_test1.tick_params(labelsize=12)
    ax_test1.legend(fontsize=12, loc="upper right")
    ax_test1.set_title("Test Result", fontsize=14)

    error = x_test_sorted - final_pred_test_sorted
    ax_test2.plot(t_test_sorted, error, "k-", linewidth=2, label="Error")
    ax_test2.set_xlim([-1.0, 1.0])
    ax_test2.tick_params(labelsize=12)
    ax_test2.legend(fontsize=12, loc="upper right")
    ax_test2.set_title("Test Error", fontsize=14)

    # 3. Training Results Plot (right side)
    gs_train = GridSpecFromSubplotSpec(5, 1, subplot_spec=gs[:, 1])

    for i in range(5):
        ax = fig.add_subplot(gs_train[i])

        mask = (t_train >= -1) & (t_train <= 1)
        t_plot = t_train[mask]
        sort_idx = np.argsort(t_plot)
        t_plot = t_plot[sort_idx]

        if i < 4:
            residual = x_train - sum(preds_train[:i]) if i > 0 else x_train
            residual_plot = residual[mask][sort_idx]
            pred_plot = preds_train[i][mask][sort_idx]

            ax.plot(
                t_plot,
                residual_plot,
                "b-",
                linewidth=2,
                label="$f(x)$" if i == 0 else "$E_{%d}(x)$" % i,
            )
            ax.plot(
                t_plot,
                pred_plot,
                "r--",
                linewidth=2,
                label=(
                    "$N_{\mathtt{CF}}^{(0)}(x)$"
                    if i == 0
                    else "$\mathrm{\epsilon}_{%d}N_{\mathtt{CF}}^{(%d)}(x)$" % (i, i)
                ),
            )
        else:
            # Plot final residual E_4
            residual = x_train - sum(preds_train[:4])
            residual_plot = residual[mask][sort_idx]
            ax.plot(
                t_plot,
                residual_plot,
                "b-",
                linewidth=2,
                label="$E_{4}(x)$",
            )

        ax.legend(fontsize=12, loc="upper right")
        ax.set_xlim([-1, 1])
        ax.tick_params(labelsize=12)
        ax.set_title(f"Training Stage {i}" if i < 4 else "Training Error", fontsize=14)

    plt.tight_layout()
    plt.savefig(
        os.path.join(base_dir, example_name, "combined_results.pdf"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()


def make_compare_plot(example_names: list, base_dir: str = "./results"):
    """Create comparison plots for two different methods.

    Args:
        example_names: List of two example names to compare
        base_dir: Base directory where results are saved
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 2, height_ratios=[1, 0.4, 0.4])

    # Load results for both methods
    results = []
    for name in example_names:
        results.append(loadmat(os.path.join(base_dir, name, "results.mat")))

    # Plot training loss (top row)
    for idx, (name, result) in enumerate(zip(example_names, results)):
        ax = fig.add_subplot(gs[0, idx])

        # Combine all losses
        losses = result["losses"].squeeze()
        all_losses = np.concatenate([stage_loss.flatten() for stage_loss in losses])

        ax.semilogy(all_losses)

        # Add vertical lines for stages if more than one stage
        actual_iters = [len(stage_losses.flatten()) for stage_losses in losses]
        if len(actual_iters) == 4:
            for i in range(len(actual_iters) - 1):
                ax.axvline(
                    x=sum(actual_iters[: i + 1]), color="r", linestyle="--", alpha=0.5
                )
            for i in range(len(actual_iters)):
                start = 0 if i == 0 else sum(actual_iters[:i])
                end = sum(actual_iters[: i + 1])
                mid_point = start + (end - start) // 2
                ax.text(
                    mid_point,
                    ax.get_ylim()[0] + 0.08,
                    f"$\\mathrm{{Stage}}~{i}$",
                    horizontalalignment="center",
                    fontsize=14,
                )

        ax.set_xlabel("$\\mathrm{Epochs}$", fontsize=14)
        ax.set_ylabel("$\\mathrm{Loss}$", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.set_title("Training Loss", fontsize=14)

    # Plot training results (middle row)
    for idx, (name, result) in enumerate(zip(example_names, results)):
        ax = fig.add_subplot(gs[1, idx])

        t_train = result["t_train"].squeeze()
        x_train = result["x_train"].squeeze()
        preds_train = [pred.squeeze() for pred in result["preds_train"]]

        # Final prediction (sum of all stages)
        final_pred_train = sum(preds_train)

        # Sort for plotting
        mask = (t_train >= -1) & (t_train <= 1)
        t_plot = t_train[mask]
        sort_idx = np.argsort(t_plot)
        t_plot = t_plot[sort_idx]
        x_train_plot = x_train[mask][sort_idx]
        final_pred_train_plot = final_pred_train[mask][sort_idx]

        # Calculate training error
        train_error = x_train_plot - final_pred_train_plot

        ax.plot(t_plot, train_error, "k-", linewidth=2)
        ax.set_xlim([-1.0, 1.0])
        ax.tick_params(labelsize=12)
        ax.set_title("Training Error", fontsize=14)
        ax.set_xlabel("$x$", fontsize=14)
        ax.set_ylabel("Error", fontsize=14)

    # Plot test results (bottom row)
    for idx, (name, result) in enumerate(zip(example_names, results)):
        ax = fig.add_subplot(gs[2, idx])

        t_test = result["t_test"].squeeze()
        x_test = result["x_test"].squeeze()
        preds_test = [pred.squeeze() for pred in result["preds_test"]]

        # Final prediction (sum of all stages)
        final_pred_test = sum(preds_test)

        # Sort for plotting
        sort_idx = np.argsort(t_test)
        t_test = t_test[sort_idx]
        x_test = x_test[sort_idx]
        final_pred_test = final_pred_test[sort_idx]

        # Calculate test error
        test_error = x_test - final_pred_test

        ax.plot(t_test, test_error, "k-", linewidth=2)
        ax.set_xlim([-1.0, 1.0])
        ax.tick_params(labelsize=12)
        ax.set_title("Test Error", fontsize=14)
        ax.set_xlabel("$x$", fontsize=14)
        ax.set_ylabel("Error", fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "comparison.pdf"), dpi=600, bbox_inches="tight")
    plt.close()
