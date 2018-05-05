import numpy as np
import tensorflow as tf
from constants import *

class Model(object):
  def __init__(self, enhance = False, imitate = False, optimize = 0):
    self.enhance = enhance
    self.imitate = imitate
    self.optimize = optimize
    # Define kernel initializers and kernel regularizers.
    self.my_initializer = tf.initializers.random_uniform(-0.05, 0.05)
    self.my_regularizer = tf.contrib.layers.l2_regularizer(C_L2)
    
    # Placeholder for inputs.
    self.Inputs = tf.placeholder(tf.float32, (None, BOARD_SIZE_Y, BOARD_SIZE_X, 4))
    # Compute the feature.
    self.x = self.residual_block(self.Inputs)
    # Compute the value logits.
    self.logits_v = self.value_head(self.x)
    # Apply a tanh to rescale the value to range [-1, 1].
    self.v = tf.tanh(self.logits_v)
    # Compute the policy logits.
    self.logits_p = self.policy_head(self.x)
    # Apply a softmax to normalize the probability distribution.
    self.p = tf.nn.softmax(self.logits_p)
    
    if enhance:
      # Placeholder for feature sequence and sequence length.
      self.Feature_Sequence = tf.placeholder(tf.float32, (None, None, BOARD_SIZE_Y, BOARD_SIZE_X, NUM_FILTERS))
      self.Sequence_Length = tf.placeholder(tf.int32, (None,))
      # Compute the contextual feature.
      self.phi = self.lstm_block(self.Feature_Sequence, self.Sequence_Length)
      # Compute the delta value logits.
      self.delta_logits_v = self.value_calibration_head(self.x, self.phi)
      # Apply a tanh to rescale the value to range [-1, 1].
      self.v_prime = tf.tanh(self.logits_v + self.delta_logits_v)
      # Compute the delta policy logits.
      self.delta_logits_p = self.policy_calibration_head(self.x, self.phi)
      # Apply a softmax to normalize the probability distribution.
      self.p_prime = tf.nn.softmax(self.logits_p + self.delta_logits_p)
    
    if imitate:
      # Compute the imitated contextual feature.
      self.phi_hat = self.imitation_block(self.x)
      # Compute the imitated delta value logits.
      self.delta_logits_v_hat = self.value_calibration_head_hat(self.x, self.phi_hat)
      # Apply a tanh to rescale the value to range [-1, 1].
      self.v_hat = tf.tanh(self.logits_v + self.delta_logits_v_hat)
      # Compute the imitated delta policy logits.
      self.delta_logits_p_hat = self.policy_calibration_head_hat(self.x, self.phi_hat)
      # Apply a softmax to normalize the probability distribution.
      self.p_hat = tf.nn.softmax(self.logits_p + self.delta_logits_p_hat)
    
    if optimize != 0:
      # Placeholder for optimization.
      self.Pi = tf.placeholder(tf.float32, (None, BOARD_SIZE_X*BOARD_SIZE_Y+1))
      self.Z = tf.placeholder(tf.float32, (None, 1))
      self.LR = tf.placeholder(tf.float32, [])
      
      # Define Alpha Zero loss.
      self.value_loss1 = tf.losses.mean_squared_error(self.Z, self.v)
      self.policy_loss1 = tf.losses.softmax_cross_entropy(self.Pi, self.logits_p)
      self.loss1 = self.value_loss1 + self.policy_loss1 + tf.losses.get_regularization_loss()
      
      # Define additional value loss and policy loss.
      with tf.variable_scope("value_calibration_head/conv", reuse = True):
        kernel_v1 = tf.get_variable("kernel1")
        kernel_v2 = tf.get_variable("kernel2")
      with tf.variable_scope("value_calibration_head/dense1", reuse = True):
        kernel_v_dense1 = tf.get_variable("kernel")
      with tf.variable_scope("value_calibration_head/dense2", reuse = True):
        kernel_v_dense2 = tf.get_variable("kernel")
      with tf.variable_scope("policy_calibration_head/conv", reuse = True):
        kernel_p1 = tf.get_variable("kernel1")
        kernel_p2 = tf.get_variable("kernel2")
      with tf.variable_scope("policy_calibration_head/dense", reuse = True):
        kernel_p_dense = tf.get_variable("kernel")
      with tf.variable_scope("lstm_block/rnn/lstm_cell", reuse = True):
        kernel_rnn = tf.get_variable("kernel")
      var_v_batch = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "value_calibration_head/batch")
      var_v_dense1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "value_calibration_head/dense1")
      var_v_dense2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "value_calibration_head/dense2")
      var_p_batch = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "policy_calibration_head/batch")
      var_p_dense = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "policy_calibration_head/dense")
      var_lstm = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "lstm_block")
      regularizer2 = [kernel_v1, kernel_v2, kernel_v_dense1, kernel_v_dense2, kernel_p1, kernel_p2, kernel_p_dense, kernel_rnn]
      self.value_loss2 = tf.losses.mean_squared_error(self.Z, self.v_prime)
      self.policy_loss2 = tf.losses.softmax_cross_entropy(self.Pi, tf.stop_gradient(self.logits_p) + self.delta_logits_p)
      self.regularization_loss2 = sum(C_L2 * 2 * tf.nn.l2_loss(var) for var in regularizer2)
      self.loss2 = self.value_loss2 + self.policy_loss2 + self.regularization_loss2
      trainable2 = [kernel_v1, kernel_v2, kernel_p1, kernel_p2] + var_v_batch + var_v_dense1 + var_v_dense2 + var_p_batch + var_p_dense + var_lstm
      gradients2 = tf.gradients(self.loss2, trainable2)
      
      # Define feature loss.
      regularizer3 = []
      for i in range(NUM_RESBLOCKS_IMIT):
        with tf.variable_scope("imitation_block/Res" + str(i) + "/conv1", reuse = True):
          regularizer3.append(tf.get_variable("kernel"))
        with tf.variable_scope("imitation_block/Res" + str(i) + "/conv2", reuse = True):
          regularizer3.append(tf.get_variable("kernel"))
      var_imit = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "imitation_block")
      self.feature_loss = C_FEATURE * tf.losses.mean_squared_error(tf.stop_gradient(self.phi), self.phi_hat)
      self.regularization_loss3 = sum(C_L2 * 2 * tf.nn.l2_loss(var) for var in regularizer3)
      self.loss3 = self.feature_loss + self.regularization_loss3
      trainable3 = var_imit
      gradients3 = tf.gradients(self.loss3, trainable3)
      
      # Define value hat loss and policy hat loss.
      with tf.variable_scope("value_calibration_head/conv", reuse = True):
        kernel_v3 = tf.get_variable("kernel3")
      with tf.variable_scope("policy_calibration_head/conv", reuse = True):
        kernel_p3 = tf.get_variable("kernel3")
      regularizer4 = [kernel_v3, kernel_p3]
      self.value_loss3 = tf.losses.mean_squared_error(self.Z, self.v_hat)
      self.policy_loss3 = tf.losses.softmax_cross_entropy(self.Pi, tf.stop_gradient(self.logits_p) + self.delta_logits_p_hat)
      self.regularization_loss4 = sum(C_L2 * 2 * tf.nn.l2_loss(var) for var in regularizer4)
      self.loss4 = self.value_loss3 + self.policy_loss3 + self.regularization_loss4
      trainable4 = [kernel_v3, kernel_p3]
      gradients4 = tf.gradients(self.loss4, trainable4)
      
      # Define optimizer.
      optimizer = tf.train.MomentumOptimizer(self.LR, MOMENTUM_OPT)
      # Add dependency to the batch normalization.
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        self.train_op1 = optimizer.minimize(self.loss1)
        self.train_op2 = optimizer.apply_gradients(zip(gradients2, trainable2))
        self.train_op3 = optimizer.apply_gradients(zip(gradients3, trainable3))
        self.train_op4 = optimizer.apply_gradients(zip(gradients4, trainable4))
      self.train_op34 = tf.group(self.train_op3, self.train_op4)
  
  def residual_block(self, inputs):
    is_training = True if self.optimize == 1 else False
    with tf.variable_scope("residual_block"):
      # First is a convolutional block,
      # which contains a convolutional layer, a batch normalization, and a ReLU.
      z = tf.layers.conv2d(inputs, NUM_FILTERS, (3, 3), padding = "same", use_bias = False, kernel_initializer = self.my_initializer, kernel_regularizer = self.my_regularizer, name = "conv", reuse = tf.AUTO_REUSE)
      z_bn = tf.layers.batch_normalization(z, scale = False, training = is_training, name = "batch", reuse = tf.AUTO_REUSE)
      x = tf.nn.relu(z_bn)
      
      # Then follows 9 residual blocks,
      # each of which contains a convolutional layer, a batch normalization, a ReLU,
      # a convolutional layer, a batch normalization, a skip connection, and a ReLU.
      for i in range(NUM_RESBLOCKS):
        with tf.variable_scope("Res" + str(i)):
          z1 = tf.layers.conv2d(x, NUM_FILTERS, (3, 3), padding = "same", use_bias = False, kernel_initializer = self.my_initializer, kernel_regularizer = self.my_regularizer, name = "conv1", reuse = tf.AUTO_REUSE)
          z1_bn = tf.layers.batch_normalization(z1, scale = False, training = is_training, name = "batch1", reuse = tf.AUTO_REUSE)
          x1 = tf.nn.relu(z1_bn)
          z2 = tf.layers.conv2d(x1, NUM_FILTERS, (3, 3), padding = "same", use_bias = False, kernel_initializer = self.my_initializer, kernel_regularizer = self.my_regularizer, name = "conv2", reuse = tf.AUTO_REUSE)
          z2_bn = tf.layers.batch_normalization(z2, scale = False, training = is_training, name = "batch2", reuse = tf.AUTO_REUSE)
          z2_con = tf.add(x, z2_bn)
          x = tf.nn.relu(z2_con)
    
    return x
  
  def lstm_block(self, feature_sequence, sequence_length):
    is_training = True if self.optimize == 2 else False
    with tf.variable_scope("lstm_block") as scope:
      # Define RNN cell and initial state.
      rnn_cell = tf.contrib.rnn.ConvLSTMCell(2, [BOARD_SIZE_Y, BOARD_SIZE_X, NUM_FILTERS], NUM_FILTERS, [3, 3], use_bias = False, initializers = self.my_initializer, name = "lstm_cell")
      initial_state = rnn_cell.zero_state(tf.shape(feature_sequence)[0], dtype=tf.float32)
      
      # Compute the contextual feature via LSTM block, 
      # which contains a LSTM, a batch normalization, and a ReLU.
      phi_output, _ = tf.nn.dynamic_rnn(rnn_cell, feature_sequence, sequence_length, initial_state)
      # Filter the last output.
      id = tf.concat([tf.reshape(tf.range(tf.shape(feature_sequence)[0]), (-1, 1)), tf.reshape(tf.subtract(sequence_length, 1), (-1, 1))], 1)
      phi_output = tf.gather_nd(phi_output, id)
      # Continue with batch normalization.
      phi_bn = tf.layers.batch_normalization(phi_output, scale = False, training = is_training, name = "batch", reuse = tf.AUTO_REUSE)
      phi = tf.nn.relu(phi_bn)
    
    return phi
  
  def imitation_block(self, x):
    is_training = True if self.optimize == 3 else False
    with tf.variable_scope("imitation_block"):
      # Compute the imitated contextual feature via imitation block,
      # which contains a convolutional layer, a batch normalization, a ReLU,
      # a convolutional layer, a batch normalization, a skip connection, and a ReLU.
      for i in range(NUM_RESBLOCKS_IMIT):
        with tf.variable_scope("Res" + str(i)):
          z1 = tf.layers.conv2d(x, NUM_FILTERS, (3, 3), padding = "same", use_bias = False, kernel_initializer = self.my_initializer, name = "conv1", reuse = tf.AUTO_REUSE)
          z1_bn = tf.layers.batch_normalization(z1, scale = False, training = is_training, name = "batch1", reuse = tf.AUTO_REUSE)
          x1 = tf.nn.relu(z1_bn)
          z2 = tf.layers.conv2d(x1, NUM_FILTERS, (3, 3), padding = "same", use_bias = False, kernel_initializer = self.my_initializer, name = "conv2", reuse = tf.AUTO_REUSE)
          z2_bn = tf.layers.batch_normalization(z2, scale = False, training = is_training, name = "batch2", reuse = tf.AUTO_REUSE)
          z2_con = tf.add(x, z2_bn)
          x = tf.nn.relu(z2_con)
      phi_hat = x
    
    return phi_hat
  
  def value_head(self, x):
    is_training = True if self.optimize == 1 else False
    with tf.variable_scope("value_head"):
      # Compute the value logits via value head,
      # which contains a convolutional layer, a batch normalization, a ReLU,
      # a fully connected layer, a ReLU, and a fully connected layer to a scalar.
      z = tf.layers.conv2d(x, NUM_FILTERS_V, (1, 1), padding = "same", use_bias = False, kernel_initializer = self.my_initializer, name = "conv", reuse = tf.AUTO_REUSE)
      z_bn = tf.layers.batch_normalization(z, scale = False, training = is_training, name = "batch", reuse = tf.AUTO_REUSE)
      x1 = tf.nn.relu(z_bn)
      z1 = tf.layers.dense(tf.layers.flatten(x1), NUM_FC_V, kernel_initializer = self.my_initializer, kernel_regularizer = self.my_regularizer, name = "dense1", reuse = tf.AUTO_REUSE)
      x2 = tf.nn.relu(z1)
      logits_v = tf.layers.dense(x2, 1, use_bias = False, kernel_initializer = self.my_initializer, kernel_regularizer = self.my_regularizer, name = "dense2", reuse = tf.AUTO_REUSE)
    
    return logits_v
  
  def value_calibration_head(self, x, phi):
    is_training = True if self.optimize == 2 else False
    with tf.variable_scope("value_calibration_head"):
      # Compute the delta value logits via value calibration head,
      # which contains a convolutional layer, a batch normalization, a ReLU,
      # a fully connected layer, a ReLU, and a fully connected layer to a scalar.
      with tf.variable_scope("conv", reuse = tf.AUTO_REUSE):
        kernel_v1 = tf.get_variable("kernel1", (1, 1, NUM_FILTERS, NUM_FILTERS_V), tf.float32, self.my_initializer)
        kernel_v2 = tf.get_variable("kernel2", (1, 1, NUM_FILTERS, NUM_FILTERS_V), tf.float32, self.my_initializer)
      kernel = tf.concat([kernel_v1, kernel_v2], 2)
      x_phi = tf.concat([x, phi], -1)
      z = tf.nn.conv2d(x_phi, kernel, (1, 1, 1, 1), "SAME")
      z_bn = tf.layers.batch_normalization(z, scale = False, training = is_training, name = "batch", reuse = tf.AUTO_REUSE)
      x1 = tf.nn.relu(z_bn)
      z1 = tf.layers.dense(tf.layers.flatten(x1), NUM_FC_V, kernel_initializer = self.my_initializer, name = "dense1", reuse = tf.AUTO_REUSE)
      x2 = tf.nn.relu(z1)
      delta_logits_v = tf.layers.dense(x2, 1, use_bias = False, kernel_initializer = self.my_initializer, name = "dense2", reuse = tf.AUTO_REUSE)
    
    return delta_logits_v
  
  def value_calibration_head_hat(self, x, phi_hat):
    is_training = False
    with tf.variable_scope("value_calibration_head"):
      # Compute the imitated delta value logits via imitated value calibration head,
      # which contains a convolutional layer, a batch normalization, a ReLU,
      # a fully connected layer, a ReLU, and a fully connected layer to a scalar.
      with tf.variable_scope("conv", reuse = tf.AUTO_REUSE):
        kernel_v1 = tf.get_variable("kernel1", (1, 1, NUM_FILTERS, NUM_FILTERS_V), tf.float32, self.my_initializer)
        kernel_v3 = tf.get_variable("kernel3", (1, 1, NUM_FILTERS, NUM_FILTERS_V), tf.float32, self.my_initializer)
      kernel = tf.concat([kernel_v1, kernel_v3], 2)
      x_phi = tf.concat([x, phi_hat], -1)
      z = tf.nn.conv2d(x_phi, kernel, (1, 1, 1, 1), "SAME")
      z_bn = tf.layers.batch_normalization(z, scale = False, training = is_training, name = "batch", reuse = tf.AUTO_REUSE)
      x1 = tf.nn.relu(z_bn)
      z1 = tf.layers.dense(tf.layers.flatten(x1), NUM_FC_V, kernel_initializer = self.my_initializer, name = "dense1", reuse = tf.AUTO_REUSE)
      x2 = tf.nn.relu(z1)
      delta_logits_v_hat = tf.layers.dense(x2, 1, use_bias = False, kernel_initializer = self.my_initializer, name = "dense2", reuse = tf.AUTO_REUSE)
    
    return delta_logits_v_hat
  
  def policy_head(self, x):
    is_training = True if self.optimize == 1 else False
    with tf.variable_scope("policy_head"):
      # Compute the policy logits via policy head,
      # which contains a convolutional layer, a batch normalization, a ReLU,
      # and a fully connected layer that ouputs a vector of length 65 corresponding to the logit probabilities for each action.
      z = tf.layers.conv2d(x, NUM_FILTERS_P, (1, 1), padding = "same", use_bias = False, kernel_initializer = self.my_initializer, name = "conv", reuse = tf.AUTO_REUSE)
      z_bn = tf.layers.batch_normalization(z, scale = False, training = is_training, name = "batch", reuse = tf.AUTO_REUSE)
      x1 = tf.nn.relu(z_bn)
      logits_p = tf.layers.dense(tf.layers.flatten(x1), BOARD_SIZE_X*BOARD_SIZE_Y+1, use_bias = False, kernel_initializer = self.my_initializer, kernel_regularizer = self.my_regularizer, name = "dense", reuse = tf.AUTO_REUSE)
    
    return logits_p
  
  def policy_calibration_head(self, x, phi):
    is_training = True if self.optimize == 2 else False
    with tf.variable_scope("policy_calibration_head"):
      # Compute the delta policy logits via policy calibration head,
      # which contains a convolutional layer, a batch normalization, a ReLU,
      # and a fully connected layer that ouputs a vector of length 65 corresponding to the logit probabilities for each action.
      with tf.variable_scope("conv", reuse = tf.AUTO_REUSE):
        kernel_p1 = tf.get_variable("kernel1", (1, 1, NUM_FILTERS, NUM_FILTERS_P), tf.float32, self.my_initializer)
        kernel_p2 = tf.get_variable("kernel2", (1, 1, NUM_FILTERS, NUM_FILTERS_P), tf.float32, self.my_initializer)
      kernel = tf.concat([kernel_p1, kernel_p2], 2)
      x_phi = tf.concat([x, phi], -1)
      z = tf.nn.conv2d(x_phi, kernel, (1, 1, 1, 1), "SAME")
      z_bn = tf.layers.batch_normalization(z, scale = False, training = is_training, name = "batch", reuse = tf.AUTO_REUSE)
      x1 = tf.nn.relu(z_bn)
      delta_logits_p = tf.layers.dense(tf.layers.flatten(x1), BOARD_SIZE_X*BOARD_SIZE_Y+1, use_bias = False, kernel_initializer = self.my_initializer, name = "dense", reuse = tf.AUTO_REUSE)
    
    return delta_logits_p
  
  def policy_calibration_head_hat(self, x, phi_hat):
    is_training = False
    with tf.variable_scope("policy_calibration_head"):
      # Compute the imitated delta policy logits via imitated policy calibration head,
      # which contains a convolutional layer, a batch normalization, a ReLU,
      # and a fully connected layer that ouputs a vector of length 65 corresponding to the logit probabilities for each action.
      with tf.variable_scope("conv", reuse = tf.AUTO_REUSE):
        kernel_p1 = tf.get_variable("kernel1", (1, 1, NUM_FILTERS, NUM_FILTERS_P), tf.float32, self.my_initializer)
        kernel_p3 = tf.get_variable("kernel3", (1, 1, NUM_FILTERS, NUM_FILTERS_P), tf.float32, self.my_initializer)
      kernel = tf.concat([kernel_p1, kernel_p3], 2)
      x_phi = tf.concat([x, phi_hat], -1)
      z = tf.nn.conv2d(x_phi, kernel, (1, 1, 1, 1), "SAME")
      z_bn = tf.layers.batch_normalization(z, scale = False, training = is_training, name = "batch", reuse = tf.AUTO_REUSE)
      x1 = tf.nn.relu(z_bn)
      delta_logits_p_hat = tf.layers.dense(tf.layers.flatten(x1), BOARD_SIZE_X*BOARD_SIZE_Y+1, use_bias = False, kernel_initializer = self.my_initializer, name = "dense", reuse = tf.AUTO_REUSE)
    
    return delta_logits_p_hat