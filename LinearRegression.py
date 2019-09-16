# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 22:39:10 2019

@author: Sean R Grant
"""

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    def fit(self, X, y, eta= 0.001, epochs= 1000):
        self.X = X
        self.Y = y
        self.J = []
        D = X.ndim
        if D == 1:
            denom = self.X.dot(self.X) - self.X.mean() * self.X.sum()
            self.a = (self.X.dot(self.Y) - self.Y.mean()*X.sum()) / denom
            self.b = (self.Y.mean() * self.X.dot(X) - self.X.mean() * self.X.dot(self.Y)) / denom
    
        else:
            self.w = np.random.randn(self.X.shape[1]) / np.sqrt(self.X.shape[1])

            for t in range(epochs):
                y_hat = self.X.dot(self.w)
                delta = y_hat - self.Y
                self.w = self.w - eta*self.X.T.dot(delta)
                mse = delta.dot(delta) / self.X.shape[0]
                self.J.append(mse)
    
    def predict(self, X):
        D = X.ndim
        if D == 1:
            y_hat = self.a*X + self.b
        else:
            y_hat = X.dot(self.w)
        
        return y_hat
