import numpy as np 


def sigmoid(x):
    """ Sigmoid function
    Parameter:
        x: input vector or matrix
    
    Return:
        the probability vector of x 
    """
    return 1. / (1. + np.exp(-x))


def relu(x):
    """ReLu function
    Parameter:
        x: input vector or matrix

    Return:
        a vector given by numpy.maximum(0, x)
    """
    return np.maximum(0, x)


def softmax(x):
    """Softmax function
    Parameter:
        x: input vector or matrix

    Return:
        the probability vector of x
    """
    if np.ndim(x) == 2: # for a batch (2D matrix)
        x = x - np.amax(x, axis=1, keepdims=True)
        x = np.exp(x)
        x /= np.sum(x, axis=1, keepdims=True)
    elif np.ndim(x) == 1:   # for a vector
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x


def cross_entropy_error(y, t):
    """Cross entropy error function
    Parameter:
        y: the predict result (a single vector or a mini-batch (a matrix with N vectors))
        t: the real label
    
    Return:
        a scalar value which gives the error of current model
    """
    if np.ndim(y) == 1: # construct a batch which has only one line
        y = np.reshape(y, newshape=(1, np.size(y)))
        t = np.reshape(t, newshape=(1, np.size(t)))

    if np.size(y) == np.size(t):
        t = np.argmax(t, axis=1)
    batch_size = np.shape(y)[0]

    return (-1 * np.sum(np.log(y[np.arange(batch_size), t] + 1e-7))/batch_size)


def remove_duplicate(params, grads):
    """ Remove duplicated parameters (weights)

    1. This function removes IDENTICAL OBJECTS from params and grads lists.
    Note that IDENTICAL OBJECTS are NOT the objects having the SAME VALUE 
    but the objects at the SAME MEMORY LOCATION.

    2. This function also removes transposed identical params and grads from
    lists. For example, if transpose(a) == b then removes b from the list.
    
    The operator "a == b" checks if a and b have the same value.
    The key word "a is b" checks if a and b are at the same memory location.

    Parameters:
        params: weights matrices that will be trained
        grads: the gradient matrices calculated during back-propagation

    Return:
        params: weights matrices that have been removed duplicated weights
        grads: the gradient matrices that have been removed duplicated gradient
    """
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # if two matrices are at the same memory location
                if params[i] is params[j]:
                    grads[i] += grads[j]  
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # if two matrices are transposed identical
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break # if there are no identical matrices in list, break out from while-statement

    return params, grads


def clip_grads(grads, max_norm):
    """ Gradient clipping 

    This function is used to avoid exploding gradients.

    Parameters:
        grads: gradient matrix list
        max_norm: threshold of maximum gradient. if a gradient value is larger than max_norm, it will be clipped

    Return:
        None

    Clipping method:
        if norm(grad) >= max_norm:
            grad = (max_norm / norm(grad)) * grad
    """
    total_norm = 0
    for grad in grads: 
        total_norm += np.sum(grad**2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1: 
        for grad in grads: 
            grad *= rate 