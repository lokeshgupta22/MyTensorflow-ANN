import numpy as np
import matplotlib.pyplot as plt
np.random.seed(22)


class MyTensorFlow:
    def __init__(self, X, y, layers, epoch, lr=0.01, print_cost=True, plot_graph=True):
        self.features = X
        self.labels = y
        self.lr = lr
        self.layers = layers
        self.print_cost = print_cost
        self.plot_graph = plot_graph
        self.epoch = epoch
        self.nodes = list(self.layers.keys())
        self.active = list(self.layers.values())
        self.cache = {}
        self.params = {}
        self.gradients = {}

    def activation(self, symbol, z):
        if symbol == "sigmoid":
            return (1/(1+np.exp(-z)))
        if symbol == "tanh":
            return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
        if symbol == "relu":
            return (np.maximum(0, z))
        if symbol == "lrelu":
            return (np.maximum((0.01*z), z))

    def deactivation(self, symbol, s, alpha=0.01):
        if symbol == "dsigmoid":
            return (s*(1-s))
        if symbol == "dtanh":
            return (1-s**2)
        if symbol == "drelu":
            return (np.int64(s > 0))
        if symbol == "dlrelu":
            return (np.where(s > 0, 1, alpha))

    def print_dict(self, my_dict):
        for key, value in my_dict.items():
            print(f"  '{key}': array({value.tolist()}),")

    def initialize_wnb(self):
        for i in range(len(self.layers)):
            if i == 0:
                self.params[f"w{i+1}"] = np.random.randn(self.nodes[i], self.features.shape[0])*0.01
                self.params[f"b{i+1}"] = np.zeros((self.nodes[i], 1))
                continue
            elif i == (len(self.layers)-1):
                self.params[f"w{i+1}"] = np.random.randn(self.labels.shape[0], self.nodes[i-1])*0.01
                self.params[f"b{i+1}"] = np.zeros((self.labels.shape[0], 1))
                continue
            else:
                self.params[f"w{i+1}"] = np.random.randn(self.nodes[i], self.nodes[i-1])*0.01
                self.params[f"b{i+1}"] = np.zeros((self.nodes[i], 1))
        return self.params

    def fwdprop(self):
        for i in range(len(self.layers)):
            if i == 0:
                self.cache[f"z{i+1}"] = np.dot(self.params[f"w{i+1}"],self.features)+self.params[f"b{i+1}"]
            else:
                self.cache[f"z{i+1}"] = np.dot(self.params[f"w{i+1}"],self.cache[f"a{i}"])+self.params[f"b{i+1}"]
            self.cache[f"a{i+1}"] = self.activation(self.active[i],self.cache[f"z{i+1}"])
        return self.cache

    def calc_cost(self):
        logp = np.multiply(np.log(self.cache[f"a{len(self.layers)}"]), y)+np.multiply(np.log(1-(self.cache[f"a{len(self.layers)}"])), (1-y))
        cost = -np.sum(logp)/(self.labels.shape[1])
        return cost

    def bwdprop(self):
        m = (self.labels.shape[1])
        for i in range(len(self.layers), 0, -1):
            if i == (len(self.layers)):
                self.gradients[f"dz{i}"] = (self.cache[f"a{len(self.layers)}"])-self.labels
                self.gradients[f"dw{i}"] = np.dot(self.gradients[f"dz{len(self.layers)}"], (self.cache[f"a{len(self.layers)-1}"]).T)/m
            elif i == 1:
                self.gradients[f"da{i}"] = np.dot((self.params[f"w{i+1}"]).T, self.gradients[f"dz{i+1}"])
                self.gradients[f"dz{i}"] = (self.gradients[f"da{i}"]) * (self.deactivation(symbol=f"d{self.active[len(self.layers)-1-i]}", s=self.cache[f"a{i}"]))
                self.gradients[f"dw{i}"] = np.dot(self.gradients[f"dz{i}"], self.features.T)/m
            else:
                self.gradients[f"da{i}"] = np.dot((self.params[f"w{i+1}"]).T, self.gradients[f"dz{i+1}"])
                self.gradients[f"dz{i}"] = (self.gradients[f"da{i}"]) * (self.deactivation(symbol=f"d{self.active[len(self.layers)-i-1]}", s=self.cache[f"a{i}"]))
                self.gradients[f"dw{i}"] = np.dot(self.gradients[f"dz{i}"], self.cache[f"a{i-1}"].T)/m
            self.gradients[f"db{i}"] = np.sum(self.gradients[f"dz{i}"], axis=1, keepdims=True)/m
        return self.gradients

    def update_wnb(self):
        for i in range(len(self.layers)):
            self.params[f"w{i+1}"] -= self.lr * self.gradients[f"dw{i+1}"]
            self.params[f"b{i+1}"] -= self.lr * self.gradients[f"db{i+1}"]
        return self.params

    def fit(self):
        self.initialize_wnb()
        costall = []
        for i in range(self.epoch):
            self.fwdprop()
            c = self.calc_cost()
            costall.append(c)
            self.bwdprop()
            self.update_wnb()
            if self.print_cost and i % (self.epoch/10) == 0:
                print("Cost at %ith iteration: %f" % (i, c))
        print("Cost at %ith iteration: %f" % (i+1, c))
        if self.plot_graph:
            plt.plot(costall)
            plt.show()
        return self.params

    def predict(self):
        if self.active[len(self.layers)-1] == "sigmoid":
            pred = (self.cache[f"a{len(self.layers)}"] > 0.5)
            return pred
        else:
            print("Still in progress. Please use sigmoid in the final layer to predict.")

    def accuracy(self, print_pred=True):
        count = 0
        pred = self.predict()
        if print_pred:
            print(pred)
        for i in range(self.labels.shape[1]):
            if pred[0][i] == self.labels[0][i]:
                count += 1
        return count/(self.labels.shape[1])


X = np.random.randn(4, 10)
y = (np.random.randn(1, 10) > 0)

tf = MyTensorFlow(
    X=X,
    y=y,
    layers={4: "relu",
            3: "relu",
            2: "lrelu",
            1: "sigmoid"},
    epoch=2500,
    plot_graph=False
)
print(tf.fit())
print(f"Accuracy: {tf.accuracy(print_pred=False)}")
