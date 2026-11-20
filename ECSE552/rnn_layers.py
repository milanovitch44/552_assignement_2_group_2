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
  ##############################################################################
  # Linear pre-activation: x @ Wx (N, H) + prev_h @ Wh (N, H) + b (H,) -> shape (N, H)
  h = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
  # Non-linear activation squashed into [-1, 1]: shape (N, H)
  next_h = np.tanh(h)
  # Cache inputs, weights, and pre-activation for backward pass
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
  - dWx: Gradients of input-to-hidden weights, of shape (D, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  ##############################################################################
  x, prev_h, Wx, Wh, h = cache

  # Gradient through tanh nonlinearity: dtanh = dnext_h * (1 - tanh(h)^2) -> shape (N, H)
  dtanh = dnext_h * (1 - np.tanh(h)**2)

  # dx = dtanh @ Wx.T: (N, H) @ (H, D) -> (N, D)
  dx = np.dot(dtanh, Wx.T)
  # dprev_h = dtanh @ Wh.T: (N, H) @ (H, H) -> (N, H)
  dprev_h = np.dot(dtanh, Wh.T)
  # dWx = x.T @ dtanh: (D, N) @ (N, H) -> (D, H)
  dWx = np.dot(x.T, dtanh)
  # dWh = prev_h.T @ dtanh: (H, N) @ (N, H) -> (H, H)
  dWh = np.dot(prev_h.T, dtanh)
  # db = sum of dtanh over batch dimension N -> shape (H,)
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
  ##############################################################################
  N, T, D = x.shape
  H = h0.shape[1]

  # Pre-allocate tensor to store hidden states for all T timesteps: shape (N, T, H)
  h = np.zeros((N, T, H))

  prev_h = h0 # Initialize hidden state at t=0 with h0: shape (N, H)
  step_caches = []

  # Iterate forward chronologically through sequence timesteps t = 0 ... T-1
  for t in range(T):
    # Slice current timestep input x[:, t, :] of shape (N, D)
    next_h, step_cache = rnn_step_forward(
        x[:, t, :], prev_h, Wx, Wh, b
    )

    # Store hidden activation into output tensor: shape (N, H)
    h[:, t, :] = next_h
    step_caches.append(step_cache)

    # Pass hidden state to next timestep
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
  ##############################################################################
  x, h0, Wx, Wh, b, step_caches = cache

  N, T, D = x.shape
  H = h0.shape[1]

  # Allocate gradient buffers matching their respective tensor shapes
  dx = np.zeros_like(x)      # (N, T, D)
  dWx = np.zeros_like(Wx)    # (D, H)
  dWh = np.zeros_like(Wh)    # (H, H)
  db = np.zeros_like(b)      # (H,)

  # Gradient flowing from future hidden state; 0 at terminal timestep t=T-1: shape (N, H)
  dprev_h = np.zeros_like(h0)

  # Backpropagate through time (BPTT) in reverse chronological order: t = T-1 down to 0
  for t in reversed(range(T)):
    # Total gradient at timestep t combines direct loss gradient dh[:, t, :] and recurrence from t+1
    dcurrent_h = dh[:, t, :] + dprev_h

    # Step backward through single RNN cell
    dx_t, dprev_h, dWx_t, dWh_t, db_t = rnn_step_backward(
        dcurrent_h, step_caches[t]
    )

    # Store input gradient for timestep t: shape (N, D)
    dx[:, t, :] = dx_t
    # Accumulate shared parameter gradients across all timesteps
    dWx += dWx_t
    dWh += dWh_t
    db += db_t

  # Gradient arriving at initial hidden state h0: shape (N, H)
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
  #############################################################################
  N, D = x.shape
  H = prev_h.shape[1]

  # Joint affine projection for all 4 gates: (N, D) @ (D, 4H) + (N, H) @ (H, 4H) + (4H,) -> shape (N, 4H)
  a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b

  # Split joint activation into 4 gate vectors of size H each
  ai = a[:, :H]        # Input gate pre-activation: shape (N, H)
  af = a[:, H:2*H]     # Forget gate pre-activation: shape (N, H)
  ao = a[:, 2*H:3*H]   # Output gate pre-activation: shape (N, H)
  ag = a[:, 3*H:]      # Candidate cell pre-activation: shape (N, H)

  # Non-linear activations: sigmoids produce gates in [0, 1]; tanh produces candidate in [-1, 1]
  i = 1 / (1 + np.exp(-ai)) # Input gate: how much new info to write
  f = 1 / (1 + np.exp(-af)) # Forget gate: how much old cell state to retain
  o = 1 / (1 + np.exp(-ao)) # Output gate: how much cell state to expose to hidden state
  g = np.tanh(ag)           # Candidate values to add to cell state

  # Cell state update: retain fraction of prev_c + add gated candidate values: shape (N, H)
  next_c = f * prev_c + i * g
  # Hidden state: squashed cell state filtered by output gate: shape (N, H)
  next_h = o * np.tanh(next_c)

  # Cache activations and weights for chain rule in backward pass
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
  #############################################################################
  x, prev_h, prev_c, Wx, Wh, b, i, f, o, g, next_c = cache

  tanh_next_c = np.tanh(next_c) # shape (N, H)

  # Gradient through output gate: next_h = o * tanh(next_c) -> do = dnext_h * tanh(next_c): shape (N, H)
  do = dnext_h * tanh_next_c
  # Total cell gradient: upstream dnext_c + path through next_h (dnext_h * o * (1 - tanh^2)): shape (N, H)
  dtotal_c = dnext_c + dnext_h * o * (1 - tanh_next_c**2)

  # Gradients through next_c = f * prev_c + i * g
  df = dtotal_c * prev_c        # Gradient w.r.t forget gate: shape (N, H)
  dprev_c = dtotal_c * f        # Gradient w.r.t previous cell state: shape (N, H)

  di = dtotal_c * g             # Gradient w.r.t input gate: shape (N, H)
  dg = dtotal_c * i             # Gradient w.r.t candidate cell: shape (N, H)

  # Backprop through gate activations: sigmoid derivative is s*(1-s), tanh derivative is (1-g^2)
  dai = di * i * (1 - i)        # Pre-activation input gate gradient: shape (N, H)
  daf = df * f * (1 - f)        # Pre-activation forget gate gradient: shape (N, H)
  dao = do * o * (1 - o)        # Pre-activation output gate gradient: shape (N, H)
  dag = dg * (1 - g**2)         # Pre-activation candidate gradient: shape (N, H)

  # Concatenate gate pre-activation gradients along feature dimension: shape (N, 4H)
  da = np.hstack((dai, daf, dao, dag))

  # Gradients with respect to inputs, states, and shared parameters
  dx = np.dot(da, Wx.T)         # (N, 4H) @ (4H, D) -> shape (N, D)
  dprev_h = np.dot(da, Wh.T)    # (N, 4H) @ (4H, H) -> shape (N, H)
  dWx = np.dot(x.T, da)         # (D, N) @ (N, 4H) -> shape (D, 4H)
  dWh = np.dot(prev_h.T, da)    # (H, N) @ (N, 4H) -> shape (H, 4H)
  db = np.sum(da, axis=0)       # Sum across batch dimension N -> shape (4H,)
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
  #############################################################################
  N, T, D = x.shape
  H = h0.shape[1]

  # Pre-allocate output hidden state buffer for all T timesteps: shape (N, T, H)
  h = np.zeros((N, T, H))

  # Initial states: prev_h set to h0, prev_c initialized to all zeros: shape (N, H)
  prev_h = h0
  prev_c = np.zeros_like(h0)

  step_caches = []

  # Forward pass unrolled across sequence timesteps t = 0 ... T-1
  for t in range(T):
    # Execute single LSTM step forward with input slice x[:, t, :] of shape (N, D)
    next_h, next_c, step_cache = lstm_step_forward(
        x[:, t, :], prev_h, prev_c, Wx, Wh, b
    )

    # Record hidden state at timestep t: shape (N, H)
    h[:, t, :] = next_h

    step_caches.append(step_cache)

    # Advance both hidden and internal cell states to next timestep
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
  #############################################################################
  x, h0, Wx, Wh, b, step_caches = cache

  N, T, D = x.shape
  H = h0.shape[1]

  # Allocate gradient buffers
  dx = np.zeros_like(x)      # (N, T, D)
  dWx = np.zeros_like(Wx)    # (D, 4H)
  dWh = np.zeros_like(Wh)    # (H, 4H)
  db = np.zeros_like(b)      # (4H,)

  # Recurring gradient states from future timestep (t+1); initially zeros: shape (N, H)
  dprev_h = np.zeros_like(h0)
  dprev_c = np.zeros_like(h0)

  # Backpropagate through time (BPTT) backwards from t = T-1 down to 0
  for t in reversed(range(T)):
    # Sum upstream direct gradient dh[:, t, :] with recurrence gradient from t+1
    dcurrent_h = dh[:, t, :] + dprev_h

    # Step backward through single LSTM cell
    dx_t, dprev_h, dprev_c, dWx_t, dWh_t, db_t = lstm_step_backward(
        dcurrent_h, dprev_c, step_caches[t]
    )

    # Assign input gradient for current timestep: shape (N, D)
    dx[:, t, :] = dx_t
    # Accumulate parameter gradients across all timesteps
    dWx += dWx_t
    dWh += dWh_t
    db += db_t

  # Gradient flowing into initial hidden state h0: shape (N, H)
  dh0 = dprev_h
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db
