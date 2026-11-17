import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################
  h = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
  next_h = np.tanh(h)
  cache = (x, prev_h, Wx, Wh, h)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (N, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  x, prev_h, Wx, Wh, h = cache

  dtanh = dnext_h * (1 - np.tanh(h)**2)

  dx = np.dot(dtanh, Wx.T)
  dprev_h = np.dot(dtanh, Wh.T)
  dWx = np.dot(x.T, dtanh)
  dWh = np.dot(prev_h.T, dtanh)
  db = np.sum(dtanh, axis=0)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, None
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################
  N, T, D = x.shape
  H = h0.shape[1]

  h = np.zeros((N, T, H))

  prev_h = h0
  step_caches = []

  for t in range(T):
    next_h, step_cache = rnn_step_forward(
        x[:, t, :], prev_h, Wx, Wh, b
    )

    h[:, t, :] = next_h
    step_caches.append(step_cache)

    prev_h = next_h

  cache = (x, h0, Wx, Wh, b, step_caches)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
  x, h0, Wx, Wh, b, step_caches = cache

  N, T, D = x.shape
  H = h0.shape[1]

  dx = np.zeros_like(x)
  dWx = np.zeros_like(Wx)
  dWh = np.zeros_like(Wh)
  db = np.zeros_like(b)

  dprev_h = np.zeros_like(h0)

  for t in reversed(range(T)):
    dcurrent_h = dh[:, t, :] + dprev_h

    dx_t, dprev_h, dWx_t, dWh_t, db_t = rnn_step_backward(
        dcurrent_h, step_caches[t]
    )

    dx[:, t, :] = dx_t
    dWx += dWx_t
    dWh += dWh_t
    db += db_t

  dh0 = dprev_h
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  N, D = x.shape
  H = prev_h.shape[1]

  a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b

  ai = a[:, :H]
  af = a[:, H:2*H]
  ao = a[:, 2*H:3*H]
  ag = a[:, 3*H:]

  i = 1 / (1 + np.exp(-ai))
  f = 1 / (1 + np.exp(-af))
  o = 1 / (1 + np.exp(-ao))
  g = np.tanh(ag)

  next_c = f * prev_c + i * g
  next_h = o * np.tanh(next_c)

  cache = (x, prev_h, prev_c, Wx, Wh, b, i, f, o, g, next_c)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  x, prev_h, prev_c, Wx, Wh, b, i, f, o, g, next_c = cache

  tanh_next_c = np.tanh(next_c)

  # Gradient through the hidden state
  do = dnext_h * tanh_next_c
  dtotal_c = dnext_c + dnext_h * o * (1 - tanh_next_c**2)

  # Gradient through the cell state
  df = dtotal_c * prev_c
  dprev_c = dtotal_c * f

  di = dtotal_c * g
  dg = dtotal_c * i

  # Gradients through the gates
  dai = di * i * (1 - i)
  daf = df * f * (1 - f)
  dao = do * o * (1 - o)
  dag = dg * (1 - g**2)

  # Combine gate gradients
  da = np.hstack((dai, daf, dao, dag))

  # Gradients with respect to inputs and weights
  dx = np.dot(da, Wx.T)
  dprev_h = np.dot(da, Wh.T)
  dWx = np.dot(x.T, da)
  dWh = np.dot(prev_h.T, da)
  db = np.sum(da, axis=0)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  N, T, D = x.shape
  H = h0.shape[1]

  h = np.zeros((N, T, H))

  prev_h = h0
  prev_c = np.zeros_like(h0)

  step_caches = []

  for t in range(T):

    next_h, next_c, step_cache = lstm_step_forward(
        x[:, t, :], prev_h, prev_c, Wx, Wh, b
    )

    h[:, t, :] = next_h

    step_caches.append(step_cache)

    prev_h = next_h
    prev_c = next_c

  cache = (x, h0, Wx, Wh, b, step_caches)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  x, h0, Wx, Wh, b, step_caches = cache

  N, T, D = x.shape
  H = h0.shape[1]

  dx = np.zeros_like(x)
  dWx = np.zeros_like(Wx)
  dWh = np.zeros_like(Wh)
  db = np.zeros_like(b)

  dprev_h = np.zeros_like(h0)
  dprev_c = np.zeros_like(h0)

  for t in reversed(range(T)):
    dcurrent_h = dh[:, t, :] + dprev_h

    dx_t, dprev_h, dprev_c, dWx_t, dWh_t, db_t = lstm_step_backward(
        dcurrent_h, dprev_c, step_caches[t]
    )

    dx[:, t, :] = dx_t
    dWx += dWx_t
    dWh += dWh_t
    db += db_t

  dh0 = dprev_h
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db
