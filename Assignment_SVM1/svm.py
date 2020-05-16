import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.svm import SVC

seed=10
X, y = make_blobs(n_samples=1000, centers=2, center_box=[-6.5, 6.5], random_state=seed)

def plot_data(X,y):
    c0 = np.where(y == 0)[0]
    plt.scatter(X[c0, 0], X[c0, 1], c="purple", marker='s')
    
    c1 = np.where(y == 1)[0]
    plt.scatter(X[c1, 0], X[c1, 1], c="black", marker='o')
    
plot_data(X, y)
plt.title("Separable dataset of size 1000 with 2 features")
plt.show()

svm = SVC(C=100, kernel='linear', random_state=seed)
svm.fit(X,y)

plot_data(X,y)

def plot_margins(svm, X, y):
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    xx = np.linspace(xmin, xmax, 20)
    yy = np.linspace(ymin, ymax, 20)
    XX, YY = np.meshgrid(xx, yy)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = svm.decision_function(xy).reshape(XX.shape)
    
    plt.contour(XX,YY,Z,colors='green',levels=[-1, 0, 1],alpha=1.0,linestyles=['--','-','--'])
    
plot_margins(svm, X, y)
plt.title("Separable Vectors for the for the above dataset")
plt.show()

print("On changing the size of the dataset thereby changing the vectors other than support vectors")
def plotting_svm(var):
    seed=10
    X, y = make_blobs(n_samples=var, centers=2, center_box=[-6.5, 6.5], random_state=seed)
    svm = SVC(C=100, kernel='linear', random_state=seed)
    svm.fit(X,y)

    plot_data(X,y)
    plot_margins(svm, X, y)
    plt.title("Number of data points = {0}".format(var))
    plt.show()

plotting_svm(400)
plotting_svm(600)
print("Thus by looking into graph we can conclude that changing the vectors other than the support vectors will not affect the decision boundary")