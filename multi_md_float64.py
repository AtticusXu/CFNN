import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from util import SinNN, CFNN, ExpPlus, MultiStageNN
from scipy.io import savemat
import os
import sys
import pdb
os.chdir(sys.path[0])

TEST_Name = '_Layer'+sys.argv[1]+'_dim'+sys.argv[2]
H_Layer = int(sys.argv[1])
dim = int(sys.argv[2])

np.random.seed(114)
tf.random.set_seed(514)
N_eval = 10000
layersi = [dim,H_Layer,H_Layer,H_Layer,1]
layers = [dim,H_Layer,H_Layer,H_Layer,1]
kappa = 1


# def fun_test(t):
#     # customize the function by the user
#     dim_ = t.shape[-1]
#     sigma = [1.0]*dim_
#     omega = [1.0]*dim_
#     x = 1
#     for i in range(dim_):
#         x *= tf.exp(-sigma[i]**2*((t[:,i:i+1]+1)/2-omega[i])**2)
#     return x

def fun_test(t):
    # customize the function by the user
    dim_ = t.shape[-1]
    x = 0
    for i in range(dim_):
        x += t[:,i:i+1]
    return x


t = np.random.uniform(-1.02, 1.02,[10000,dim])
t_train = tf.cast(t, dtype=tf.float64)
x_train = fun_test(t_train)

# Domain bounds
# lt = t.min(0)
# ut = t.max(0)
lt = tf.reduce_min(t_train,axis=0)
ut = tf.reduce_max(t_train,axis=0)

t_eval = np.random.uniform(-1, 1,[N_eval,dim])
t_eval = tf.cast(t_eval, dtype=tf.float64)
x_eval = fun_test(t_eval)

adam_lr = 0.001

plot_dir = "./plots_CFNN/md_fix_10000_chebweight_"+TEST_Name+"/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
#%%
'''
First stage of training
'''
# acts = 0 indicates selecting tanh as the activation function
fl_ini = ExpPlus(5.0, 0, tf.float64)
NN = CFNN(layersi, fl_ini, tf.float64)
NN.count_trainable_variables()
# pdb.set_trace()

model = MultiStageNN(t_train, x_train, NN, lt, ut, adam_lr)
# start the first stage training
model.train(2000, 1)     # mode 1 use Adam
model.train(10000, 2)    # mode 2 use L-bfgs
#%%
x_pred = model.predict(t_eval)
loss = model.loss_NN().numpy()
loss_his = [l.numpy() for l in model.loss]
loss_last = loss_his[-1]
plt.semilogy(loss_his)
plt.savefig(plot_dir+'loss_1.png', dpi = 600)
plt.close('all')

# plt.plot(t_eval, x_eval - model.predict(t_eval), 'r-', linewidth = 2, label = 'error')
# plt.savefig(plot_dir+'error1.png', dpi = 600)
# plt.show()
#%%
'''
Second stage of training
'''


# calculate the residue for the second stage
x_train2 = x_train - model.predict(t_train)

fl_ini_2 = ExpPlus(1, 0, tf.float64, seed=43)
NN_2 = CFNN(layers, fl_ini_2, tf.float64)

model2 = MultiStageNN(t_train, x_train2, NN_2, lt, ut, adam_lr)

x_pred_btrain = model2.predict(t_train)

# plt.plot(t_train, x_train2, 'b-', linewidth = 2, label = 'target')
# plt.plot(t_train, x_pred_btrain, 'r--', linewidth = 2, label = 'NN')
# plt.legend()
# plt.show()
#%%

# start the second stage training
model2.train(5000, 1)    # mode 1 use Adam
model2.train(20000, 2)   # mode 2 use L-bfgs
x_pred2 = model2.predict(t_eval)
# combining the result from first and second stage
x_p = x_pred + x_pred2


plt.semilogy(model2.loss)
plt.savefig(plot_dir+'loss_2.png', dpi = 600)
plt.close('all')

# plt.plot(t_eval, x_eval-x_p, 'r-', linewidth = 2, label = 'error')
# plt.savefig(plot_dir+'error2.png', dpi = 600)
# plt.show()
#%%
'''
Third stage of training
'''

# calculate the residue for the third stage
x_train3 = x_train - model.predict(t_train) - model2.predict(t_train)

fl_ini_3 = ExpPlus(0.2, 10, tf.float64)
NN_3 = CFNN(layers, fl_ini_3, tf.float64)

model3 = MultiStageNN(t_train, x_train3, NN_3, lt, ut, adam_lr)
# start the third stage training
model3.train(5000, 1)      # mode 1 use Adam
model3.train(30000, 2)     # mode 2 use L-bfgs
x_pred3 = model3.predict(t_eval)
# combining the result from first, second and third stages
x_p2 = x_pred + x_pred2 + x_pred3


plt.semilogy(model3.loss)
plt.savefig(plot_dir+'loss_3.png', dpi = 600)
plt.close('all')

# plt.plot(t_eval, x_eval-x_p2, 'r-', linewidth = 2, label = 'error')
# plt.savefig(plot_dir+'error3.png', dpi = 600)
# plt.show()
#%%
'''
Forth stage of training
'''

adam_lr = tf.keras.optimizers.schedules.ExponentialDecay(
    0.01,
    decay_steps=100,
    decay_rate=0.97,
    staircase=True)
# calculate the residue for the forth stage
x_train4 = x_train - model.predict(t_train) - model2.predict(t_train) - model3.predict(t_train)

fl_ini_4 = ExpPlus(0.04, 50, tf.float64)
NN_4 = CFNN(layers, fl_ini_4, tf.float64)

model4 = MultiStageNN(t_train, x_train4, NN_4, lt, ut, adam_lr)
# start the forth stage training
model4.train(5000, 1)
model4.train(40000, 2)
x_pred4 = model4.predict(t_eval)
# combining the result from all stages
x_p3 = x_pred + x_pred2 + x_pred3 + x_pred4

plt.semilogy(model4.loss)
plt.savefig(plot_dir+'loss_4.png', dpi = 600)
plt.close('all')

# plt.plot(t_eval, x_eval-x_p3, 'r-', linewidth = 2, label = 'error')
# plt.savefig(plot_dir+'error4.png', dpi = 600)
# plt.show()
#%%
# combine the loss of all four stages of training
loss = np.array(model.loss + model2.loss + model3.loss + model4.loss)

residue = x_train4 - model4.predict(t_train)

error_x = np.linalg.norm(x_eval-x_pred, 2)/np.linalg.norm(x_eval, 2)
print('Error u: %e' % (error_x))

error_x2 = np.linalg.norm(x_eval-x_p, 2)/np.linalg.norm(x_eval, 2)
print('Error u: %e' % (error_x2))

error_x3 = np.linalg.norm(x_eval-x_p2, 2)/np.linalg.norm(x_eval, 2)
print('Error u: %e' % (error_x3))

error_x4 = np.linalg.norm(x_eval-x_p3, 2)/np.linalg.norm(x_eval, 2)
print('Error u: %e' % (error_x4))

mdic = {"t": t_eval.numpy(), "x_g": x_eval.numpy(), "x0": x_pred.numpy(),
        "x1": x_pred2.numpy(), "x2": x_pred3.numpy(), 'x3': x_pred4.numpy(),
        "err": residue.numpy(), 'loss': loss}
FileName = plot_dir+'Reg_mNN_1D_64bit.mat'
savemat(FileName, mdic)

#%%

######################################################################
############################# Plotting ###############################
######################################################################
plt.semilogy(loss)
plt.savefig(plot_dir+'loss.png', dpi = 600)
# plt.show()

plt.close('all')

# xmin = x_eval.numpy().min()
# xmax = x_eval.numpy().max()

# fig = plt.figure(figsize=[10, 16], dpi=100)

# ax = plt.subplot(511)
# ax.plot(t_eval, x_eval, 'b-', linewidth = 2, label = 'Exact')
# ax.plot(t_eval, x_p, 'r--', linewidth = 2, label = 'Prediction')
# ax.set_ylabel('$x$', fontsize=15, rotation = 0)
# ax.set_title('Function', fontsize=10)
# ax.set_xlim([-1.05, 1.05])
# ax.set_ylim([xmin,xmax])


# ax1 = plt.subplot(512)
# ax1.plot(t_train, x_train2, 'b.', linewidth=2, label='Exact')
# ax1.plot(t_eval, x_pred2, 'r--', linewidth=2, label='Prediction')
# ax1.set_ylabel('$x$', fontsize=15, rotation=0)
# ax1.set_title('Residue order 1', fontsize=10)
# ax1.set_xlim([-1.05, 1.05])


# ax2 = plt.subplot(513)
# ax2.plot(t_train, x_train3, 'b-', linewidth=2, label='Exact')
# ax2.plot(t_eval, x_pred3, 'r--', linewidth=2, label='Prediction')
# # ax2.plot(t_train, tf.zeros(t_train.shape[0]), 'kx', linewidth = 1, label = 'Exact')
# ax2.set_ylabel('$x$', fontsize = 15, rotation = 0)
# ax2.set_title('Residue order 2', fontsize = 10)
# ax2.set_xlim([-1.05, 1.05])


# ax3 = plt.subplot(514)
# ax3.plot(t_train, x_train4, 'b-', linewidth = 2, label = 'Exact')
# ax3.plot(t_eval, x_pred4, 'r--', linewidth=2, label='Prediction')
# ax3.set_xlabel('$t$', fontsize = 15)
# ax3.set_ylabel('$x$', fontsize = 15, rotation = 0)
# ax3.set_title('Residue order 3', fontsize = 10)
# ax3.set_xlim([-1.05, 1.05])

# ax3 = plt.subplot(515)
# ax3.plot(t_eval, x_eval-x_p3, 'b-', linewidth = 2, label = 'error')
# ax3.set_xlabel('$t$', fontsize = 15)
# ax3.set_ylabel('$x$', fontsize = 15, rotation = 0)
# ax3.set_title('Residue order 4', fontsize = 10)
# ax3.set_xlim([-1.05, 1.05])

# plt.savefig(plot_dir+'error.png', dpi = 600)
# plt.show()