import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

# nn with forward and backward pass
# 2 hidden layers

# plot
    # predictions and ground truth
    # learning curves
    # accuracy of my prediction

# each layer contains at least one Transformaton & Activation fcn


# Generate the input data
# Function usage 
    # x, y = generate_linear(n=100)
    # x, y = generate_XOR_easy()
def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) /1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []
    
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        
        if 0.1*i == 0.5:
            continue
            
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
        
    return np.array(inputs), np.array(labels).reshape(21, 1)


# Activation function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

def tanh(x):
    return (1.0 - np.exp(-2 * x)) / (1.0 + np.exp(-2 * x))

def derivative_tanh(x):
    return 1 - tanh(x) ** 2


# MSE
def mse_loss(y_train, y_pred):
    return np.mean((y_pred - y_train) ** 2)

def derivative_mse_loss(y_train, y_pred):
    return 2*(y_pred - y_train) / y_train.shape[0]


# Result
def show_result(x, y, pred_y, title=None):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
            
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    if title is not None:
        plt.suptitle(title)
    plt.show()


# Learning curve(loss, epoch)
def learning_curve(train_loss, title=None):
    x = len(train_loss)
    if title is not None:
        plt.title(title + ' Learning Curve')
    else:
        plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(train_loss)
    plt.show()

# def learning_curve(losses, title=None):
#     plt.figure(figsize=(6, 5))
#     for lr, train_loss in losses.items():
#         plt.plot(train_loss, label=f'lr={lr:.2f}')

#     plt.title('Learning Curve' if title is None else title + ' Learning Curve')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()


class Layer():
    def __init__(self, in_dim, out_dim, activate='sigmoid', bias=False):
        self.bias = bias
        self.w = np.random.normal(0, 1, size=(in_dim, out_dim))
        if self.bias:
            self.b = np.zeros(out_dim)
        self.activate = activate
        
        # Adam
        # m_t = beta1 m_t-1 + (1 - beta1)
        self.m = 0
        self.v = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        # print(self.w)
        self.local_grad = x
        self.z = x @ self.w
        if self.bias:
            self.local_grad_b = np.ones((x.shape[0], 1))
            self.z += self.b
        out = self.z
        
        # Activate function
        if self.activate is not None:
            if self.activate == 'sigmoid':
                out = sigmoid(out)
            if self.activate == 'tanh':
                out = tanh(out)
        
        return out
    
    def backward(self, x):
        self.up_grad = x
        if self.activate is not None:
            if self.activate == 'sigmoid':
                self.up_grad *= derivative_sigmoid(self.z)
            if self.activate == 'tanh':
                self.up_grad *= derivative_tanh(self.z)
        
        return self.up_grad @ self.w.T
    
    def step(self, lr, optim):
        # update the weight
        if optim == 'Adam':
            # Adam, contrains: weight_decay=0, minimize, betas=[0.9, 0.999], amsgrad=False, eps = 1e-08
            grad_w = self.local_grad.T @ self.up_grad
            self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad_w
            self.v = self.beta2 * self.v + (1.0 - self.beta2) * np.square(grad_w)

            m_hat = self.m / (1.0 - self.beta1)
            v_hat = self.v / (1.0 - self.beta2)

            self.w -= lr * m_hat / np.sqrt(v_hat + 1e-08)
        else:
        # gradient decent
            grad_w = self.local_grad.T @ self.up_grad
            self.w -= lr * grad_w
            if self.bias:
                grad_b = self.local_grad_b.T @ self.up_grad
                self.b -= lr * grad_b.squeeze()


class Network():
    def __init__(self, x_train, y_train, lr=1e-3, bias=False, hidden_unit=8, activate='sigmoid', optim='Adam'):
        self.lr = lr
        self.bias = bias
        self.optim = optim
        self.layers = []
        
        self.layers.append(Layer(len(x_train[0]), hidden_unit, bias=self.bias, activate=activate))
        self.layers.append(Layer(hidden_unit, hidden_unit, bias=self.bias, activate=activate))
        self.layers.append(Layer(hidden_unit, 1, bias=self.bias, activate=activate))
            
        self.num_layer = len(self.layers)
        
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        for i in range(self.num_layer):
            x = self.layers[i].forward(x)
        return x
    
    def backward(self, loss):
        x = loss
        for i in reversed(range(self.num_layer)):
            x = self.layers[i].backward(x)
            
    def step(self,):
        for i in range(self.num_layer):
            self.layers[i].step(self.lr, self.optim)


def train(model, x_train, y_train, batch_size=-1):
    
    max_epochs = 100000
    epsilon = 0.005
    target = 0
    epoch = 0
    batch = batch_size if batch_size != -1 else x_train.shape[0]
    train_loss = []
    
    while True:
        start_idx = 0
        end_idx = min(start_idx + batch, x_train.shape[0])
        loss = 0
        batch_num = 0
        while True:
            x_batch = x_train[start_idx : end_idx]
            y_batch = y_train[start_idx : end_idx]
            y_pred = model.forward(x_batch)
            loss += mse_loss(y_batch, y_pred)
            model.backward(derivative_mse_loss(y_batch, y_pred))
            model.step()
            
            batch_num += 1
            
            start_idx = end_idx
            end_idx = min(start_idx + batch, x_train.shape[0])
            if start_idx >= x_train.shape[0]:
                break

        
        loss /= batch_num
        train_loss.append(loss)
        if epoch % 500 == 0:
            print(f"epoch {epoch:6} loss : {loss:.6f}")

        epoch += 1

        if loss <= epsilon or epoch >= max_epochs:
            print(f"epoch {epoch:6} loss : {loss:.6f}")
            break
            
    return model, train_loss
        
def test(model, x_test, y_test, dataset=None):
    y_pred = model(x_test)
    
    loss = mse_loss(y_pred, y_test)
    
    y_result = np.round(y_pred)
    y_result[y_result < 0] = 0
    
    correct = np.sum(y_result == y_test)
    acc = correct / len(y_test)
    show_result(x_test, y_test, y_result, title=dataset)

    for i in range(len(y_test)):
        print(f"Iter{i:2} |\tGround truth: {y_test[i][0]:.1f} |\t prediction: {y_pred[i][0]:.5f} |")
        
    print(f"loss={loss:.5f} accuracy={acc*100:.2f}%")
    


# Parse arguements
def parse_arguments():
    parser = argparse.ArgumentParser(description='Lab1')
    parser.add_argument('--lr', default=1.0, type=float)
    parser.add_argument('--batch_size', default=-1, type=int)
    parser.add_argument('--activate', default='sigmoid', type=str)
    parser.add_argument('--hidden_unit', default=8, type=int)
    parser.add_argument('--optim', default='GD', type=str)
    
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_arguments()
    lr = args.lr
    batch_size = args.batch_size
    activate = args.activate
    hidden_unit = args.hidden_unit
    optim = args.optim
    
    x_train, y_train = generate_linear()
    
    # # For different learning rate
    # learning_rates = np.arange(0.1, 1.1, 0.1)
    # losses = {}
    # for lr in learning_rates:
    #     model = Network(x_train, y_train, lr=lr, bias=False, hidden_unit=hidden_unit, activate=activate, optim=optim)
    #     _, train_loss = train(model, x_train, y_train, batch_size=batch_size)
    #     losses[lr] = train_loss
    
    model = Network(x_train, y_train, lr=lr, optim=optim, activate=activate, hidden_unit=hidden_unit)
    model, train_loss = train(model, x_train, y_train, batch_size=batch_size)
    
    # generate test data
    x_test, y_test = generate_linear(n=100)
    test(model, x_test, y_test, dataset='Linear')
    learning_curve(train_loss, title='Linear')
    # learning_curve(losses, title='Linear')
    print()
    # generate the XOR data
    x_train, y_train = generate_XOR_easy()
    # build the model
    model1 = Network(x_train, y_train, lr=lr, optim=optim, activate=activate, hidden_unit=hidden_unit)
    model1, train_loss = train(model1, x_train, y_train, batch_size=batch_size)
    # generate test data
    x_test, y_test = generate_XOR_easy()
    test(model1, x_test, y_test, dataset='XOR')
    learning_curve(train_loss, title='XOR')
    # learning_curve(losses, title='XOR')