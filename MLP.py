import numpy as np
import matplotlib.pyplot as plt

from net import TwoLayerNet
from opt import Solver

print('==================')
print('Training ')
x_t = np.array([[0,0],[1,1],[-1,1],[0,2],[2,0],[-2,0]])
y_t = np.array([0,0,0,1,1,1])
small_data = {'X_train': x_t.copy(), 'y_train':y_t.copy(), 'X_val':x_t.copy(), 'y_val':y_t.copy(), 'X_test':x_t.copy(), 'y_test':y_t.copy()}
for k, v in list(small_data.items()):
    print(('%s: ' % k, v.shape))

N, D, H, C = small_data['X_train'].shape[0], 2, 4, 2
loss = 'softmax_loss'
reg = 0.0001
std = 1e-3


model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, 
                    reg=0, weight_scale=std, loss_name=loss)
                    
def acc(y_scores, y_true):
    y_pred = np.argmax(y_scores, axis=1)
    return np.mean(y_pred == y_true)

def plot_decision_boundary(X, y, pred_func):
    plt.figure()
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
   
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = np.argmax(pred_func(np.c_[xx.ravel(), yy.ravel()]), axis=1)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha = 0.7, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

    
learning_rate = 0.05
solver = Solver(model, small_data,
                print_every=100, num_epochs=300, batch_size=6,
                update_rule='sgd_momentum',
                optim_config={
                    'learning_rate': learning_rate,
                    'momentum': 0.9
                },
                metric=acc,
                #checkpoint_name='ZHY'
                )
solver.train()
test_acc = solver.check_accuracy(small_data['X_test'], small_data['y_test'])
print('Test Accuracy : ', test_acc)

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

plot_decision_boundary(small_data['X_train'], small_data['y_train'], lambda x: model.loss(x))
plt.show()
print(model.params)
