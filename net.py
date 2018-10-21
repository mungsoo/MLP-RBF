import numpy as np



# Input number of x is N, each x is transform to a D-dimensional vector.
# The dimension M of w means there are M neurons in this layer, or to say, M kernels.
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

    N = x.shape[0]
    out = x.reshape(N, -1).dot(w) + b

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
      - b: Bias, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None

    N = x.shape[0]
    dx = dout.dot(w.T).reshape(x.shape)
    
    
    # Sum all gradient of each input x as the total gradient of W. The same for b
    dw = x.reshape(N, -1).T.dot(dout)           
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache

    dx = dout
    dx[x <= 0] = 0                                                                          

    return dx


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    # The comment below is a better way(more numerical accurate!!!)
    
    
    # shifted_logits = x - np.max(x, axis=1, keepdims=True)                                 
    # Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)                             
   
    # log_probs = shifted_logits - np.log(Z)
    # probs = np.exp(log_probs)
    # N = x.shape[0]
    # loss = -np.sum(log_probs[np.arange(N), y]) / N
    # dx = probs.copy()
    # dx[np.arange(N), y] -= 1
    # dx /= N
    
    # This approach is less numerical accurate
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N                                      
    dx = probs.copy()                                                                       
    dx[np.arange(N), y] -= 1
    dx /= N

    return loss, dx


def mse_loss(x, y):
    '''
    regression loss: mean square error loss
    :param x: Predict value, of shape (N,1) or (N,). N is batch size.
    :param y: Ground truth, of shape (N,) or (N,1). N is batch size.
    :return: mean square error loss and dx. dx must be of shape (N,1)
    '''

    x = np.squeeze(x)
    y = np.squeeze(y)
    N = x.shape[0]
    loss = np.sum(np.square(x - y)) / (2 * N)
    dx = (x - y) / N
    dx = dx.reshape(dx.shape[0], 1)
    return loss, dx


def affine_relu_forward(x, w, b):
    """
    Convenience layer that performs an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db
class LogisticRegression(object):
    
    def __init__(self, input_dim, weight_scale=1e-3):
        self.params = {}
        self.input_dim = input_dim
        self.weight_scale = weight_scale
        self.init_weight()  
    
    def init_weight(self):
        self.params['W'] = np.random.normal(scale=self.weight_scale,
                                            size=(self.input_dim))
        self.params['b'] = 0
    def loss(self, x, y=None):
        '''
        - x: (N, d_1, ..., d_k) 其中 d_1*d_2...*dk=self.input_dim
        - y: (N,)
        Return
        当y=None时
        - scores: 即分两类的概率
        否则
        - loss
        - grads
        '''
        N = x.shape[0]
        wx = x.dot(self.params['W']).T + self.params['b']
        score = np.exp(wx)/(1+np.exp(wx))
        scores = np.array([1-score,score]).T
        if y is None:
            return scores

        loss = -1/N*np.sum(wx*y - np.log(1+np.exp(wx)))
        grads = {}
        grads['W'] = -1/N*np.sum((y*x.T).T-(score*x.T).T, axis=0)
        grads['b'] = -1/N*np.sum(y-score)
        return loss, grads

        
class LinearRegression(object):
    def __init__(self, input_dim, output_dim, loss_name='mse_loss', reg=0.0, weight_scale=1e-3):
        self.params = {}
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loss_name = eval(loss_name)
        self.reg = reg
        self.weight_scale = weight_scale
        self.init_weight()
    def init_weight(self):
        '''
        - W: (D, M)
        - b: (M,)
        '''
        self.params['W'] = np.random.normal(scale=self.weight_scale,
                                            size=(self.input_dim, self.output_dim))
        self.params['b'] = np.zeros(self.output_dim)
    def loss(self, X, y=None):
        # This is an GD approach of solving linear regression problem
        '''
        - X: (N, d_1, ..., d_k) 其中 d_1*d_2...*dk=self.input_dim
        - y: (N,M)
        Return
        当y=None时
        - scores: 即回归输出
        否则
        - loss
        - grads
        '''
        N = X.shape[0]
        X = X.reshape(N, -1)
        scores = X.dot(self.params['W']) + self.params['b']
        if y is None:
            return scores

        loss, dout = self.loss_name(scores, y)

        grads = {}
        _, grads['W'], grads['b'] = affine_backward(dout, (X, self.params['W'].reshape(-1,1), self.params['b']))
        # L2 regularization
        loss += self.reg /2 /N * np.sum(self.params['W']**2)
        grads['W'] += self.reg*self.params['W'] / N

        return loss, grads
    
    def AnalyticalApproch(self, X, y):
        
        # This is an analytical approach to solve linear regression problem
        N = X.shape[0]
        X = X.reshape(N, -1)
        b = np.ones([N, 1])
        Lambda = 0.1
        X = np.concatenate((X,b), axis=1)
       
        
        # I add a random value to X.T*X so that we can compute its invert matrix when X.T*X is a singular matrix
        # Add such a 'noise' diagonal matrix can guruntee the result maxtrix is invertible matrix! 
        Z = np.array(np.mat(np.dot(X.T,X)+0.01*np.eye(X.shape[1])).I)
        self.params['W'] = Z.dot(X.T).dot(y).reshape(X.shape[1], self.output_dim)#这里做个reshape是把所有一维向量都表示成矩阵形式，更统一，防止乱     
        self.params['b'] = self.params['W'][-1]
        self.params['W'] = np.delete(self.params['W'], self.input_dim-1, 0)
        return self.params.copy()
        
        
        
    
    
class TwoLayerNet(object):

    # This class contain all its parameters(self.param)
    # loss() is used to compute the loss of a iteration(return value loss)
    # and the gradient of W and b(return value grad)
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim, hidden_dim, num_classes,
                 weight_scale=1e-3, reg=0.0, loss_name='softmax'):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        self.input_dim = input_dim
        self.weight_scale = weight_scale
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.loss_name = eval(loss_name)
        self.init_weight()

    def init_weight(self):

        self.params['W1'] = np.random.normal(scale=self.weight_scale,
                                            size=(self.input_dim, self.hidden_dim))
        self.params['W2'] = np.random.normal(scale=self.weight_scale,
                                            size=(self.hidden_dim, self.num_classes))
        self.params['b1'], self.params['b2'] = np.zeros(self.hidden_dim), np.zeros(self.num_classes)
        

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None


        a1, (fc_cache1, relu_cache)= affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores, fc_cache2 = affine_forward(a1, self.params['W2'], self.params['b2'])
        

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}

        N = X.shape[0]
        loss, dscores = self.loss_name(scores, y)

        # Add L2 regularization
        loss += self.reg /2 /N * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))
        da1, grads['W2'], grads['b2'] = affine_backward(dscores, fc_cache2)
        
        grads['W2'] += self.reg*self.params['W2'] / N
        _, grads['W1'], grads['b1'] = affine_relu_backward(da1, (fc_cache1, relu_cache))
        grads['W1'] += self.reg*self.params['W1'] / N
        #print(grads['W2'].shape, self.params['W2'].shape)


        return loss, grads
