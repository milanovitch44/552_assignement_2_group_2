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
    # Stride and zero-padding hyperparameters
    stride = conv_param.get("stride", 1)
    pad = conv_param.get("pad", 0)

    # Input dimensions: batch size N, channels C, spatial height H, width W
    N, C, H, W = x.shape
    # Filter dimensions: number of filters F, channels C_w, kernel height HH, width WW
    F, C_w, HH, WW = w.shape
    assert C == C_w, "Input channels must match filter channels"

    # Compute output spatial dimensions H_out and W_out using standard convolution formula
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    # Zero-pad spatial dimensions (H, W) by 'pad'; batch N and channel C are unpadded
    # x_pad shape: (N, C, H + 2*pad, W + 2*pad)
    x_pad = np.pad(
        x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant", constant_values=0
    )  # padding should only be applied to the pixel dimensions (size H and W), not on the channels

    # Pre-allocate output feature maps of shape (N, F, H_out, W_out)
    out = np.zeros((N, F, H_out, W_out))

    # Naive nested loops implementation: iterate over batches, filters, and spatial output locations
    for n in range(N):             # Loop over each image in minibatch
        for f in range(F):         # Loop over each convolution filter
            for i in range(H_out): # Loop over output vertical grid
                for j in range(W_out): # Loop over output horizontal grid
                    # Spatial slice coordinates in x_pad according to stride step
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW

                    # Extract 3D receptive field across all C channels: shape (C, HH, WW)
                    receptive_field = x_pad[
                        n, :, h_start:h_end, w_start:w_end
                    ]  # receptive field shape = (HH, WW)
                    # Element-wise multiply across (C, HH, WW), sum all elements, and add scalar bias b[f]
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

    # Input and filter shapes
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = dout.shape # dout shape: (N, F, H_out, W_out)

    # Recreate padded input matching forward pass: shape (N, C, H + 2*pad, W + 2*pad)
    x_pad = np.pad(
        x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant", constant_values=0
    )
    # Initialize gradient containers matching their parameter shapes
    dx_pad = np.zeros_like(x_pad) # Shape: (N, C, H + 2*pad, W + 2*pad)
    dw = np.zeros_like(w)         # Shape: (F, C, HH, WW)
    db = np.zeros_like(b)         # Shape: (F,)

    # db[f] = sum of dout[:, f, :, :] over N, H_out, and W_out (broadcast reverse)
    for f in range(F):
        db[f] = np.sum(dout[:, f, :, :])

    # Accumulate gradients dw and dx_pad by sliding receptive windows
    for n in range(N):             # Loop over each image
        for f in range(F):         # Loop over each filter
            for i in range(H_out): # Loop over output height
                for j in range(W_out): # Loop over output width
                    # Receptive window bounds in padded input
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW

                    # Extract local receptive field slice: shape (C, HH, WW)
                    receptive_field = x_pad[n, :, h_start:h_end, w_start:w_end]

                    # dw[f] += input_patch * upstream_scalar dout[n, f, i, j] -> shape (C, HH, WW)
                    dw[f] += receptive_field * dout[n, f, i, j]

                    # dx_pad += filter_weights * upstream_scalar dout[n, f, i, j] -> shape (C, HH, WW)
                    dx_pad[n, :, h_start:h_end, w_start:w_end] += (
                        w[f] * dout[n, f, i, j]
                    )

    # Crop off zero-padding to recover gradient dx with original shape (N, C, H, W)
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
    # Pooling kernel size (pool_height, pool_width) and stride
    pool_height = pool_param.get("pool_height", 2)
    pool_width = pool_param.get("pool_width", 2)
    stride = pool_param.get("stride", 2)

    # Input shape: (N, C, H, W)
    N, C, H, W = x.shape

    # Output spatial dimensions after downsampling
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    # Allocate pooled output array: shape (N, C, H_out, W_out)
    out = np.zeros((N, C, H_out, W_out))

    # Iterate over batch N, channel C, and downsampled spatial grid (H_out, W_out)
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    # Bounding box of pooling window
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width

                    # Extract 2D pooling region: shape (pool_height, pool_width)
                    receptive_field = x[n, c, h_start:h_end, w_start:w_end]
                    # Retain only maximum activation value within region
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
    _, _, H_out, W_out = dout.shape # Shape: (N, C, H_out, W_out)

    # Initialize input gradient array with zeros: shape (N, C, H, W)
    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    # Receptive window bounds in input
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width

                    # Extract 2D window corresponding to this max-pool output cell: shape (pool_height, pool_width)
                    receptive_field = x[n, c, h_start:h_end, w_start:w_end]

                    # Identify local 2D coordinate of the element that achieved maximum during forward pass
                    max_idx = np.argmax(receptive_field)  # outputs an index (1D)
                    h_max, w_max = np.unravel_index(
                        max_idx, receptive_field.shape
                    )  # converts the 1D index back into the 2D receptive field index

                    # Max-pooling subgradient: route upstream gradient exclusively to argmax position
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
    # Running statistics tracked per feature dimension: shape (D,)
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None

    if mode == "train":
        # Minibatch mean and uncorrected variance computed per feature across batch dimension 0: shape (D,)
        sample_mean = np.mean(x, axis=0)  # shape (D)
        sample_var = np.var(x, axis=0)

        # Standardize features to zero-mean and unit-variance; eps prevents division by 0: shape (N, D)
        x_hat = (x - sample_mean) / np.sqrt(
            sample_var + eps
        )  # shape still (N,D), eps needed to prevent divide by zero

        # Affine transform: learnable scale gamma (D,) and shift beta (D,) broadcasted across N: shape (N, D)
        out = gamma * x_hat + beta  # shape still (N,D)

        # Update running averages via exponential moving average for inference time: shape (D,)
        running_mean = momentum * running_mean + (1.0 - momentum) * sample_mean
        running_var = momentum * running_var + (1.0 - momentum) * sample_var

        # Cache intermediate tensors required by chain rule during backward pass
        cache = (x, sample_mean, sample_var, x_hat, gamma, eps)

    elif mode == "test":
        # Test time: normalize using fixed running dataset statistics instead of current minibatch: shape (N, D)
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)  # shape still (N,D)
        # Apply trained gamma and beta: shape (N, D)
        out = gamma * x_hat + beta  # shape still (N,D)
        cache = None
    else:
        raise ValueError(f'Invalid forward batchnorm mode "{mode}"')

    # Store updated running statistics in dictionary for subsequent iterations
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

    # Gradients with respect to beta and gamma (summing over batch dimension N): shape (D,)
    dbeta = np.sum(dout, axis=0)  # shape (D)
    dgamma = np.sum(dout * x_hat, axis=0)  # shape (D)

    # Gradient with respect to normalized input x_hat: shape (N, D)
    dxhat = dout * gamma  # shape (N,D)

    # Gradient with respect to sample variance: dL/dvar = sum_i(dL/dxhat_i * (x_i - mean) * -0.5 * (var + eps)^(-1.5)): shape (D,)
    dvar = np.sum(
        dxhat * (x - sample_mean) * -0.5 * (sample_var + eps) ** (-1.5), axis=0
    )
    # Gradient with respect to sample mean combining direct term and path via variance: shape (D,)
    dmean = np.sum(dxhat * -1.0 / np.sqrt(sample_var + eps), axis=0) + dvar * np.mean(
        -2.0 * (x - sample_mean), axis=0
    )

    # Gradient with respect to input x combining direct normalized path, variance path, and mean path: shape (N, D)
    dx = (
        dxhat / np.sqrt(sample_var + eps)
        + dvar * 2.0 * (x - sample_mean) / N
        + dmean / N
    )  # All three parts summed up

    return dx, dgamma, dbeta
