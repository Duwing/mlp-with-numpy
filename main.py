from sklearn import datasets
import numpy as np

class Linear():
    def __init__(self, input: int, output: int, batch: int):
        self.input = input
        self.output = output
        self.weight = np.random.normal(0, 3, (self.input, self.output))
        self.batch = batch

    def forward(self, dataset, y):
        total_samples = len(dataset)
        y_preds = np.zeros((total_samples, self.output,))

        for i in range(0, total_samples, self.batch):
            end_idx = min(i + self.batch, total_samples)
            batch_data = dataset[i:end_idx]
            z_l = np.matmul(batch_data, self.weight)
            y_preds[i:end_idx] = z_l
            return y_preds
            
    
    def back_propagation(self, lr, error_grad, a_prev, z_current):
        self.weight = self.weight - lr * np.matmul(a_prev.T, error_grad * sigmoid_grad(z_current))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mse(y_pred: np.array, y: np.array) -> np.array:
    return (y_pred - y)**2

def mse_grad(y_pred: np.array, y: np.array) -> np.array:
    return 2 * (y_pred - y)

def main():
    iris = datasets.load_iris()
    
    iris_data = np.array(iris.data)
    y = np.array(iris.target)
    

    input_dim = iris_data.shape[1]
    hidden_dim = 10
    output_dim = 1
    batch_size = 64
    learning_rate = 0.01
    epochs = 1000

    # weight
    fc = Linear(input_dim, hidden_dim, batch_size)

    for i in range(epochs):

        # feed forward
        # z = x * w
        z_l = fc.forward(iris_data, y)
        a_l = sigmoid(z_l)

        error = mse(a_l, y)

        # back propagation
        # grad(Cost)/grad(w_L) = grad(z_L) / grad(w_L) * grad(a_L) / grad(z_L) * grad(Cost) / grad(a_L)
        # 
        # Cost = (a_L - y) ** 2
        # z_L = w_L * a_L-1 + b_L
        # a_L = sigma(z_L)
        # 
        # a_L-1 ->
        # w_L   ->
        # b_L   ->  z_L -> a_L ->  
        #                    y -> Cost
        #
        # w_L = a_L-1 @ (sigma_grad(z_L) * cost_grad)

        fc.back_propagation(learning_rate, np.sum(mse_grad(a_l, y), axis=1, keepdims=True), iris_data, z_l)

        if i % 100 == 0:
            print(f"Epoch: {i}, Loss: {np.mean(error)}")




if __name__ == '__main__':
    main()