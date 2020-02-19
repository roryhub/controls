import numpy as np
from numpy.linalg import inv


class KalmanFilter:

    def __init__(self):
        self.initialize_matrices()
        

    def initialize_matrices(self):
        '''
            current dimension correspond to filtering a 4 state feedback system
        '''
        self.Xpred = np.zeros((4,1))

        self.P = np.diag([1, 1, 1, 1])

        self.Q = np.diag(np.array([1, 1, 1, 1])) # process noise covariance

        self.R = np.diag(np.array([1, 1, 1, 1])) # measurement noise covariance

    
    def store_model(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C

    
    def predict_states(self, U):
        self.Xpred = self.A @ self.Xpred + self.B @ U

    
    def predict_P(self):
        self.P = self.A @ self.P @ self.A.T + self.Q


    def calculate_gain(self):
        S = self.C @ self.P @ self.C.T + self.R
        self.K = self.P @ self.C.T @ inv(S)

    
    def update_state_estimate(self, X):
        Y = self.C @ X
        X_updated = self.Xpred + self.K @ (Y - self.C @ self.Xpred)
        self.Xpred = X_updated

        return X_updated.squeeze()

    
    def update_P(self):
        self.P = (np.eye(4) - self.K @ self.C) @ self.P
