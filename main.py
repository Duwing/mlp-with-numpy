from sklearn import datasets
import numpy as np

class Linear():
    def __init__(self, input: int, output: int, batch: int):
        self.input = input
        self.output = output
        self.weight = np.random.normal(0, 3, (self.input, self.output))
        self.batch = batch

    def forward(self, dataset):
        total_samples = len(dataset)
        y_preds = np.zeros((total_samples, self.output,))

        for i in range(0, total_samples, self.batch):
            end_idx = min(i + self.batch, total_samples)
            batch_data = dataset[i:end_idx]
            batch_preds = np.matmul(batch_data, self.weight)
            y_preds[i:end_idx] = batch_preds
        return y_preds
    
    def back_propagation(self, error):
        return error


def main():
    iris = datasets.load_iris()
    
    iris_data = np.array(iris.data)
    y = np.array(iris.target)
    

    input_dim = iris_data.shape[1]
    output_dim = 1
    batch_size = 64

    # weight
    fc = Linear(input_dim, output_dim, batch_size)

    # feed forward
    y_preds = fc.forward(iris_data)
    

    # back propagation




if __name__ == '__main__':
    main()