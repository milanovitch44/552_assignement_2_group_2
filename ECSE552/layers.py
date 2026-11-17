import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    #############################################################################
    #############################################################################
    nelements = w.shape[1]
    out = np.dot(x.reshape(x.shape[0], -1), w) + b
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    dw = np.dot(dout.T, x.reshape(x.shape[0], -1)).T
    db = np.sum(dout, axis=0)
    dx = np.dot(dout, w.T).reshape(x.shape)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    stride = conv_param.get("stride", 1)
    pad = conv_param.get("pad", 0)

    N, C, H, W = x.shape
    F, C_w, HH, WW = w.shape
    assert C == C_w, "Input channels must match filter channels"

    # Compute output dimensions
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    # Zero-pad the input spatial dimensions
    x_pad = np.pad(
        x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant", constant_values=0
    )  # padding should only be applied to the pixel dimensions (size H and W), not on the channels

    # Initialize output array
    out = np.zeros((N, F, H_out, W_out))

    # Naive nested loops implementation
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    # Calculate indices
                    # multiplying times stride causes jumps in the index, which is what stride requires
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW

                    receptive_field = x_pad[
                        n, :, h_start:h_end, w_start:w_end
                    ]  # receptive field shape = (HH, WW)
                    out[n, f, i, j] = np.sum(receptive_field * w[f]) + b[f]

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """

    x, w, b, conv_param = cache
    stride = conv_param.get("stride", 1)
    pad = conv_param.get("pad", 0)

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = dout.shape

    # Zero-pad the input x and initialize gradients for padded x
    x_pad = np.pad(
        x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant", constant_values=0
    )
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # Compute db by summing over batch, height, and width dimensions of dout
    for f in range(F):
        db[f] = np.sum(dout[:, f, :, :])

    # Naive nested loops for computing dx and dw
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW

                    # Extract the local receptive field from the padded input
                    receptive_field = x_pad[n, :, h_start:h_end, w_start:w_end]

                    # Gradient with respect to filter weights (dw)
                    dw[f] += receptive_field * dout[n, f, i, j]

                    # Gradient with respect to padded input (dx_pad)
                    dx_pad[n, :, h_start:h_end, w_start:w_end] += (
                        w[f] * dout[n, f, i, j]
                    )

    # Remove padding from dx_pad to match the original input shape x
    if pad > 0:
        dx = dx_pad[:, :, pad:-pad, pad:-pad]
    else:
        dx = dx_pad

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    pool_height = pool_param.get("pool_height", 2)
    pool_width = pool_param.get("pool_width", 2)
    stride = pool_param.get("stride", 2)

    N, C, H, W = x.shape

    # Compute output dimensions
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    # Initialize output array
    out = np.zeros((N, C, H_out, W_out))

    # Naive nested loops implementation
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width

                    receptive_field = x[n, c, h_start:h_end, w_start:w_end]
                    out[n, c, i, j] = np.max(receptive_field)

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x, pool_param = cache
    pool_height = pool_param.get("pool_height", 2)
    pool_width = pool_param.get("pool_width", 2)
    stride = pool_param.get("stride", 2)

    N, C, H, W = x.shape
    _, _, H_out, W_out = dout.shape

    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    # Calculate index
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width

                    # Extract local receptive field
                    receptive_field = x[n, c, h_start:h_end, w_start:w_end]

                    # Find index of the maximum value
                    max_idx = np.argmax(receptive_field)  # outputs an index (1D)
                    h_max, w_max = np.unravel_index(
                        max_idx, receptive_field.shape
                    )  # converts the 1D index back into the 2D receptive field index

                    # Route upstream gradient to the maximum element's position
                    dx[n, c, h_start + h_max, w_start + w_max] += dout[n, c, i, j]

    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None

    if mode == "train":
        # Compute minibatch mean and uncorrected variance along axis 0 (batch dimension)
        sample_mean = np.mean(x, axis=0)  # shape (D)
        sample_var = np.var(x, axis=0)

        # Normalize the incoming data
        x_hat = (x - sample_mean) / np.sqrt(
            sample_var + eps
        )  # shape still (N,D), eps needed to prevent divide by zero

        # Scale and shift using gamma and beta
        out = gamma * x_hat + beta  # shape still (N,D)

        # Update running averages
        running_mean = momentum * running_mean + (1.0 - momentum) * sample_mean
        running_var = momentum * running_var + (1.0 - momentum) * sample_var

        # Cache intermediates needed for the backward pass
        cache = (x, sample_mean, sample_var, x_hat, gamma, eps)

    elif mode == "test":
        # Normalize using the accumulated running statistics
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)  # shape still (N,D)
        out = gamma * x_hat + beta  # shape still (N,D)
        cache = None
    else:
        raise ValueError(f'Invalid forward batchnorm mode "{mode}"')

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    x, sample_mean, sample_var, x_hat, gamma, eps = cache
    N, D = dout.shape

    # Gradients with respect to scale and shift parameters
    dbeta = np.sum(dout, axis=0)  # shape (D)
    dgamma = np.sum(dout * x_hat, axis=0)  # shape (D)

    # Gradient with respect to normalized input
    dxhat = dout * gamma  # shape (N,D)

    # Gradients with respect to variance and mean
    dvar = np.sum(
        dxhat * (x - sample_mean) * -0.5 * (sample_var + eps) ** (-1.5), axis=0
    )
    dmean = np.sum(dxhat * -1.0 / np.sqrt(sample_var + eps), axis=0) + dvar * np.mean(
        -2.0 * (x - sample_mean), axis=0
    )

    # Gradient with respect to input x
    dx = (
        dxhat / np.sqrt(sample_var + eps)
        + dvar * 2.0 * (x - sample_mean) / N
        + dmean / N
    )  # All three parts summed up

    return dx, dgamma, dbeta
