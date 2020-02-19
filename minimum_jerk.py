import numpy as np 


class MinimumJerk:

    def __init__(self, start_states, final_states, move_time, frequency):
        # start and end state arrays could contain an arbitrary number of dimensions (x,y,z...) stacked row wise
        self.start_states = start_states # initial position, velocity, and acceleration stacked column wise
        self.final_states = final_states # final position, velocity, and acceleration stacked column wise
        self.move_time = move_time # total move time in seconds
        self.frequency = frequency # control loop frequency in Hz
        self.N = int(move_time * frequency) # total number of control signals in trajectory
        self.dimensions = start_states.shape[0] # states of different dimensions are row stacked

        self.trajectory = np.array([])
        self.positions = np.array([])
        self.velocities = np.array([])
        self.accelerations = np.array([])
        self.jerks = np.array([])

        self.generate_trajectory()


    def calculate_coefficients(self, start, end):
        t = self.move_time
        conditions = np.concatenate((start,end))
        solution_array = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, t, t**2, t**3, t**4, t**5],
            [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4],
            [0, 0, 2, 6*t, 12*t**2, 20*t**3]
        ])

        return np.linalg.inv(solution_array) @ conditions


    def calculate_states(self, coefficients, t):
        '''
            generates a single position, velocity, acceleration, and jerk
            for a given time point in the trajectory
        '''
        state_array = np.array([
            [1, t, t**2, t**3, t**4, t**5],
            [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4],
            [0, 0, 2, 6*t, 12*t**2, 20*t**3],
            [0, 0, 0, 6, 24*t, 60*t**2]
        ])

        return state_array @ coefficients


    def generate_trajectory(self):
        coeff_array = np.array([
            self.calculate_coefficients(self.start_states[d], self.final_states[d])
            for d in range(self.dimensions)
        ])

        self.trajectory = np.array([
            [
                self.calculate_states(coeff_array[d], n / self.frequency)
                for n in range(1, self.N+1)
            ]
            for d in range(self.dimensions)
        ])
        
        self.positions = self.trajectory[:, :, 0]
        self.velocities = self.trajectory[:, :, 1]
        self.accelerations = self.trajectory[:, :, 2]
        self.jerks = self.trajectory[:, :, 3]