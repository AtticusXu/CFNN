"""
@author: Yongji Wang
"""


import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import time

class SinNN:
    def __init__(self, layers, kappa, datatype, acts=0):
        self.layers = layers
        self.num_layers = len(layers)-1
        self.kappa = kappa
        # Decide activation for the first layer based on 'acts'
        if acts == 0:
            first_activation = tf.tanh
        elif acts == 1:
            first_activation = lambda x: tf.sin(self.kappa * x)

        # List to hold all layer configurations including activation functions
        self.activations = [first_activation] + [tf.tanh] * (self.num_layers - 2) + [tf.identity]  # None for the last layer (linear output)

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
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=self.datatype))

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
        self.num_layers = len(layers)-1
        self.fl_ini = fl_ini
        self.datatype = datatype
        self.weights, self.biases = self.initialize_NN(layers)
        
        self.activations = [tf.tanh] * (self.num_layers - 2) + [tf.identity]
    
    def initialize_NN(self, layers):
        weights = []
        biases = []
        
        weights.append(tf.Variable(self.fl_ini.sample(sample_shape=(layers[0], layers[1])), 
                                   dtype=self.datatype))
        for l in range(1,self.num_layers):
            W = self.MPL_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=self.datatype))
            weights.append(W)
            biases.append(b)
            
        return weights, biases
    
    def MPL_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=self.datatype))

    def forward(self, X):
        A = tf.math.acos(X)
        W = self.weights[0]
        H =tf.cos(tf.matmul(A, W))
        
        for l in range(1,self.num_layers):
            W = self.weights[l]
            b = self.biases[l-1]
            H = self.activations[l-1](tf.add(tf.matmul(H, W), b))

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
        
class ExpPlus:
    def __init__(self, rate, const =0.0, dtype = tf.float32, seed = 42):
        
        self.dtype = dtype
        self.const = tf.constant(const, dtype=self.dtype)
        self.rate = tf.constant(rate, dtype=self.dtype)
        self.seed = seed
        
    def sample(self, sample_shape):
        Exp = tfp.distributions.Exponential(self.rate).sample(sample_shape, seed=self.seed)
        const = tf.constant(self.const, dtype=self.dtype, shape = sample_shape)
        
        # print(Exp+const)
        return Exp+const
        
        
        
class MultiStageNN:
    def __init__(self, t_u, x_u, NN, lt, ut, adam_lr):
        #### Changed by CY
        self.dim = t_u.shape[-1]
        self.scale = tf.math.sqrt(tf.math.reduce_mean(x_u**2,axis=0))
        x_u2 = x_u / self.scale
        
       
        self.x_u = x_u2
        self.NN = NN
        self.train_variables = self.NN.trainable_variables
        self.lt = lt
        self.ut = ut
        
        self.t_u = 1.998 * (t_u - self.lt) / (self.ut - self.lt) - 0.999
        #### Changed by CY
        self.weights_cheb = (1 / tf.sqrt(self.dim - tf.math.reduce_sum(self.t_u**2,axis=-1)))[:,None]  # Chebyshev weights

        self.loss0 = self.scale ** 2
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
        # weights = 1 / tf.sqrt(1 - self.t_u**2)  # Chebyshev weights
        # weights = 1
        squared_errors = tf.square(self.x_u - self.x_pred)  # Compute squared errors
        weighted_squared_errors = weights * squared_errors  # Apply weights
        loss = tf.reduce_mean(weighted_squared_errors)  # Weighted mean of squared errors
        return loss

    '''
    Functions used to define ADAM optimizers
    ===============================================================
    '''

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

            str_print = tf.strings.join(["Iter: ", str_iter[0],
                                         ", loss: ", str_loss[0]])
            tf.cond(
                f.iter % 10 == 0,
                lambda: tf.print(str_print),
                lambda: tf.constant(True)  # return arbitrary for non-printing case
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

    '''
    Functions used to define L-BFGS optimizers
    ===============================================================
    '''

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

            str_print = tf.strings.join(["Iter: ", str_iter[0],
                                         ", loss: ", str_loss[0]])
            tf.cond(
                f.iter % 10 == 0,
                lambda: tf.print(str_print),
                lambda: tf.constant(True)  # return arbitrary for non-printing case
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
            value_and_gradients_function=func, initial_position=init_params,
            tolerance=1e-11, max_iterations=max_nIter)

        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
        func.assign_new_model_parameters(results.position)

        return func

    '''
    Function used for training the model
    ===============================================================
    '''

    def train(self, nIter, idxOpt):
        if idxOpt == 1:
            # mode 1: running the Adam optimization
            func_adam = self.Adam_optimizer(nIter)
            self.loss += func_adam.loss
        elif idxOpt == 2:
            # mode 2: running the Lbfgs optimization
            func_bfgs = self.Lbfgs_optimizer(nIter, self.train_variables)
            self.loss += func_bfgs.loss

    # @tf.function
    def predict(self, t):
        t = 1.998 * (t - self.lt) / (self.ut - self.lt) - 0.999
        x_p = self.neural_net(t) * self.scale
        return x_p
    
if __name__ == "__main__": 
     
    noise = 0.0        

    np.random.seed(234)
    tf.random.set_seed(234)
    
    N_tr = 300
    N_pd = 200
    layers = [1, 20, 20, 20, 1]
    lyscl = [1, 1, 1, 1]
    
    def fun_test(t, mode):
        # customize the function by the user
        # x = t**2 + 0.05*tf.sin(10*np.pi*t)
        x = tf.sin(2*t+1) + 0.2*tf.exp(1.3*t)
        return x
    
    t = np.linspace(-1.05, 1.05, N_tr)[:, None]
    # t = lhs(1, N_tr) * 2 - 1
    t_train = tf.cast(t, dtype=tf.float64)
    x_train = fun_test(t_train, 1)
    
    # Domain bounds
    lt = t.min(0)
    ut = t.max(0)
    
    t_intp = np.linspace(-1, 1, N_pd)[:, None]
    t_intp = tf.cast(t_intp, dtype=tf.float64)
    x_intp = fun_test(t_intp, 0)

    '''
    training the model for the first time
    '''
    # NN = SinNN(layers, 1, tf.float64, acts=0)
    
    fl_ini = ExpPlus(0.5, 0, tf.float64)
    NN = CFNN(layers, fl_ini, tf.float64)
    
    model = MultiStageNN(t_train, x_train, NN, lt, ut)
    model.train(1000, 1)
    model.train(1000, 2)
    x_pred = model.predict(t_intp)

    x_train2 = x_train - model.predict(t_train)    

    error_x = np.linalg.norm(x_intp-x_pred, 2)/np.linalg.norm(x_intp, 2)
    print('Error u: %e' % (error_x))
    
    xmin = x_intp.numpy().min()
    xmax = x_intp.numpy().max()
    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    

    #%%
    fig = plt.figure(figsize=[10, 8], dpi=100)
    
    ax = plt.subplot(211)
    ax.plot(t_intp, x_intp, 'b-', linewidth = 2, label = 'Exact')
    ax.plot(t_intp, x_pred, 'r--', linewidth = 2, label = 'Prediction')
    ax.set_ylabel('$x$', fontsize=15, rotation = 0)
    ax.set_title('Function', fontsize = 10)
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([xmin,xmax])
    
    
    ax1 = plt.subplot(212)
    ax1.plot(t_train, x_train2, 'b.', linewidth=2, label = 'Exact')
    ax1.set_ylabel('$x$', fontsize = 15, rotation = 0)
    ax1.set_title('Residue order 1', fontsize = 10)
    ax1.set_xlim([-1.05, 1.05])

    plt.show()