import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib
kwargs = {'linewidth' : 3.5}
font = {'weight' : 'normal', 'size'   : 24}
matplotlib.rc('font', **font)

def gradient_descent(X, y, alpha, learning_rate, iterations):
    m, n = X.shape
    theta = np.zeros(n)
    losses = []
    for _ in range(iterations):
        gradient = (X.T @ (X @ theta - y) + alpha * theta) / m # Calcul du gradient formule théorique du calcul du gradient
        theta -= learning_rate * gradient # Mise à jour des paramètres
        # theta = theta - learning_rate * gradient
        loss = mean_squared_error(y, X @ theta) # Calcul de la loss
        losses.append(loss)
    return theta, losses


def projected_gradient_descent(X, y, alpha, learning_rate, iterations):
    m, n = X.shape
    theta = np.zeros(n)
    losses = []
    for _ in range(iterations):
        gradient = (X.T @ (X @ theta - y) + alpha * theta) / m
        theta -= learning_rate * gradient
        theta = np.maximum(theta, 0) # Projection sur l'ensemble des solutions admissibles
        loss = mean_squared_error(y, X @ theta)
        losses.append(loss)
    return theta, losses

def error_plot(ys, yscale='log'):
    plt.figure(figsize=(8, 8))
    plt.xlabel('Step')
    plt.ylabel('Error')
    plt.yscale(yscale)
    plt.plot(range(len(ys)), ys, **kwargs)

