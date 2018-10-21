import numpy as np
import matplotlib.pyplot as plt

train_pnts = np.array([[0,0],[1,1],[-1,1],[0,2],[2,0],[-2,0]])
train_targets = np.array([0,0,0,1,1,1])
# k-means

from sklearn.cluster import KMeans

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters).fit(train_pnts)

centers = kmeans.cluster_centers_

def plot_decision_boundary(X, y, pred_func):
    plt.figure()

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z > 0.5
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha = 0.7, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
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


class RBF(object):
    def __init__(self, centers, sigma, std=1e-3):
        self.n_clusters = centers.shape[0]
        self.input_dim = centers.shape[1]
        self.out_dim = 1
        self.sigma = sigma
        self.centers = centers
        self.weight = np.random.normal(loc=0.0, scale=std,
                                       size=(self.n_clusters, self.out_dim))
        self.bias = np.zeros((1, self.out_dim))

    def gaussian(self, dist):
        return np.exp(- np.square(dist) / (2. * self.sigma ** 2))

    def fit(self, X, y):
        batch_size = X.shape[0]
        hidden = self.cal_hidden(X)
        datamat = np.concatenate((hidden, np.ones((batch_size, 1))), axis=1)
        param = np.dot(np.dot(np.linalg.inv(np.dot(datamat.T, datamat)), datamat.T), y)
        param = param.reshape(self.n_clusters + 1, self.out_dim)
        self.weight = param[:self.n_clusters, :self.out_dim]
        self.bias = param[self.n_clusters:, :self.out_dim]

    def cal_hidden(self, X):
        distmat = cal_distmat(X, self.centers)
        return self.gaussian(distmat)

    @staticmethod
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def pred(self, X):
        hidden = self.cal_hidden(X)
        y = np.dot(hidden, self.weight) + self.bias
        return y

rbf = RBF(centers, sigma)
rbf.fit(train_pnts, train_targets)
# sample = 999
# x = np.linspace(-2.5, 2.5, num=sample)
# y = np.linspace(-0.5, 1.5, num=sample)
# x, y = np.meshgrid(x, y)
# x, y = x.ravel(), y.ravel()

# test_pnts = np.vstack((x, y)).transpose()
# def moon_plot(X, y):
    # y = np.asarray(y, dtype=float).reshape(-1)
    # X_a = X[y == 1]
    # X_b = X[y == 0]
    # plt.figure()
    # plt.plot(X_a[:, 0], X_a[:, 1], '.')
    # plt.plot(X_b[:, 0], X_b[:, 1], '.')

# pred = rbf.pred(test_pnts)
# pred = pred > 0.5
# print(pred)
# moon_plot(test_pnts, pred)
plot_decision_boundary(train_pnts, train_targets, lambda x: rbf.pred(x))
print('Sigma: ',rbf.sigma)
print('Center: \n' ,rbf.centers)
print( 'Weight: \n', rbf.weight)
print('Bias: \n', rbf.bias)
plt.scatter(train_pnts[:, 0], train_pnts[:, 1], c=train_targets, cmap=plt.cm.Spectral)
plt.show()