# !/bin/env python3

# from lz import *
import numpy as np
import matplotlib.pyplot as plt
from net import *
from opt import *
np.random.seed(16)
# generate double moon data

distance, radius, width = -5, 10, 6
sample = 500

minx, maxx = -(radius + width / 2), (2 * radius + width / 2)
miny, maxy = -(radius + width / 2 + distance), (radius + width / 2)
x = np.random.uniform(minx, maxx, size=sample)
y = np.random.uniform(miny, maxy, size=sample)


def cal_distance(x, y, x0, y0):
    dist = (x - x0) ** 2 + (y - y0) ** 2
    dist = np.sqrt(dist)
    return dist


dist_a = cal_distance(x, y, 0, 0)
dist_b = cal_distance(x, y, radius, -distance)

is_a = (dist_a <= radius + width / 2) * \
       (dist_a >= radius - width / 2) * (y >= 0)
is_b = (dist_b <= radius + width / 2) * \
       (dist_b >= radius - width / 2) * (y <= -distance)

train_pnts = np.vstack((x, y)).transpose()
train_a = train_pnts[is_a, :]
train_b = train_pnts[is_b, :]

train_pnts = np.vstack((train_a, train_b))
train_targets = np.hstack(
    (np.ones(train_a.shape[0]), np.zeros(train_b.shape[0]))
)
npnts = train_pnts.shape[0]
shuffle_ind = np.random.permutation(npnts)

train_pnts = train_pnts[shuffle_ind]
train_targets = train_targets[shuffle_ind]



# Plot the data
def moon_plot(X, y):
    y = np.asarray(y, dtype=float).reshape(-1)
    
    plt.figure()
    for i in range(len(set(y))):
        P  = X[y == i]
        plt.plot(P[:, 0], P[:, 1], '.')
        
# Plot the decision boundary
def plot_decision_boundary(X, y, pred_func):
    plt.figure()

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    
    # Sampling the whole space
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # prediction of all sampling points
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot prediction result and decision boundary
    plt.contourf(xx, yy, Z, alpha = 0.7, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    
moon_plot(train_pnts, train_targets)

# k-means

from sklearn.cluster import KMeans

n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters).fit(train_pnts)
moon_plot(train_pnts, kmeans.labels_)

centers = kmeans.cluster_centers_


def cal_distmat(X, Y=None):
    if Y is None:
        Y = X
    distmat = (X ** 2).sum(axis=1).reshape(-1, 1) + \
              (Y ** 2).sum(axis=1).reshape(1, -1) - 2 * np.dot(X, Y.T)
    distmat = np.sqrt(distmat)
    return distmat


distmat = cal_distmat(centers)
d_max = np.max(distmat)

sigma = d_max / np.sqrt(2 * n_clusters)


def acc(y_scores, y_true):
    y_scores = np.squeeze(y_scores)
    y_true = np.squeeze(y_true)
    return np.mean(y_scores == y_true)


class RBF(object):
    def __init__(self, centers, sigma, std=1e-3, loss_name='mse_loss', reg=0.0001, with_logistic=True):
        self.params = {}
        self.n_clusters = centers.shape[0]
        self.input_dim = centers.shape[1]
        self.out_dim = 1
        self.sigma = sigma
        self.centers = centers
        self.params['W'] = np.random.normal(loc=0.0, scale=std,
                                       size=(self.n_clusters, self.out_dim))
        self.params['b'] = np.zeros((1, self.out_dim))
        self.loss_name = eval(loss_name)
        self.reg = reg
        
        #This parameter is used to decide to use logistic or not
        self.with_logistic = with_logistic
    def gaussian(self, dist):
        return np.exp(- np.square(dist) / (2. * self.sigma ** 2))

    def fit(self, X, y):
        batch_size = X.shape[0]
        hidden = self.cal_hidden(X)
        datamat = np.concatenate((hidden, np.ones((batch_size, 1))), axis=1)
        param = np.dot(np.dot(np.linalg.inv(np.dot(datamat.T, datamat)), datamat.T), y)
        param = param.reshape(self.n_clusters + 1, self.out_dim)
        self.params['W'] = param[:self.n_clusters, :self.out_dim]
        self.params['b'] = param[self.n_clusters:, :self.out_dim]

    def cal_hidden(self, X):
        distmat = cal_distmat(X, self.centers)
        return self.gaussian(distmat)

    @staticmethod
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def pred(self, X):
        hidden = self.cal_hidden(X)
        y = np.dot(hidden, self.params['W']) + self.params['b']
        if self.with_logistic:
            y = self.sigmoid(y)
        return y>0.5
    
    # Used for sgd
    def loss(self, X, y=None):
        hidden = self.cal_hidden(X)
        scores, fc_cache = affine_forward(hidden, self.params['W'], self.params['b'])        
        if self.with_logistic:
            wx = scores.T
            scores = np.exp(scores)/(1+np.exp(scores))
        if y is None:
            return scores>0.5

        loss, grads = 0, {}
        N = X.shape[0]
        # Logistic regression
        if self.with_logistic:
            scores = scores.T
            hidden = hidden.reshape(N, -1)
            loss = -1/N*np.sum(wx*y - np.log(1+np.exp(wx)))
            #y = y.reshape(1,N)
            grads['W'] = -1/N*np.sum((y*hidden.T).T-(scores*hidden.T).T, axis=0).reshape(-1,self.out_dim)
            grads['b'] = -1/N*np.sum(y-scores)
   
        else:
            loss, dscores = self.loss_name(scores, y)
            _, grads['W'], grads['b'] = affine_backward(dscores, fc_cache)

            # L2 regularization
        
            loss += self.reg/2/N*np.sum(self.params['W']**2)
            _, grads['W'], grads['b'] = affine_backward(dscores, fc_cache)
            grads['W'] += self.reg*self.params['W'] / N
        return loss, grads
        
        
        

#reg = 0.0001 and do not use logistic as default
rbf = RBF(centers, sigma, with_logistic=False)
#rbf.fit(train_pnts, train_targets)
learning_rate = 0.5
small_data = {'X_train': train_pnts, 'y_train':train_targets, 'X_val':train_pnts, 'y_val':train_targets, 'X_test':train_pnts, 'y_test':train_targets}

solver = Solver(rbf, small_data,
                print_every=100, num_epochs=200, batch_size=512,
                update_rule='sgd_momentum',
                optim_config={
                    'learning_rate': learning_rate,
                    'momentum': 0.9
                },
                metric=acc
                )
                
solver.train()
plt.figure()
plt.subplot(2, 1, 1)
plt.title('Training loss')
plt.plot(solver.loss_history, 'o')
plt.xlabel('Iteration')

plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(solver.train_acc_history, '-o', label='train')
plt.plot(solver.val_acc_history, '-o', label='val')
plt.plot([0.5] * len(solver.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)


# sample = 999
# x = np.linspace(minx, maxx, num=sample)
# y = np.linspace(miny, maxy, num=sample)
# x, y = np.meshgrid(x, y)
# x, y = x.ravel(), y.ravel()

# sample = 9999
# x = np.random.uniform(minx, maxx, size=sample)
# y = np.random.uniform(miny, maxy, size=sample)

# test_pnts = np.vstack((x, y)).transpose()

# pred = rbf.pred(test_pnts)
# pred = pred > 0.5
# moon_plot(test_pnts, pred)

plot_decision_boundary(train_pnts, train_targets, lambda x: rbf.pred(x))
print('Sigma: ',rbf.sigma)
print('Center: \n' ,rbf.centers)
print( 'Weight: \n', rbf.params['W'])
print('Bias: \n', rbf.params['b'])
plt.show()
