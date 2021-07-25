import numpy as np
import matplotlib.pyplot as plt

def train(x, y):
    m = np.size(x)

    mx, my = np.mean(x), np.mean(y)

    a = np.sum((x - mx) * (y - my))/np.sum((x - mx)*x)

    b = my - a * mx
 
    def model(z):
        return z*a + b
    
    return model

def plot_regression(x, y, pred_y):
    plt.scatter(x, y, color="b", marker="o", s=30)

    plt.plot(x, pred_y, color="k")

    plt.show()

x = np.array([1, 2,3, 4, 5, 6, 7, 8])

y = np.array([1, 5,7, 9, 10, 15, 16, 20])

model = train(x, y)

preds = model(x)

plot_regression(x, y, preds)