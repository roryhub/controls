import matplotlib.pyplot as plt
import numpy as np
from minimum_jerk import MinimumJerk


class Plotter:

    def __init__(self, nrows, ncols, figsize, title=None):

        plt.rcParams['savefig.facecolor'] = 'xkcd:black'
        self.nrows = nrows
        self.ncols = ncols
        self.prepare_fig(figsize, title)
    

    def prepare_fig(self, figsize, title):

        self.fig, self.ax = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=figsize)
        self.fig.patch.set_facecolor('xkcd:black')

        for i in range(self.nrows):
            for j in range(self.ncols):
                self.ax[i, j].set_facecolor('xkcd:black')
                self.ax[i, j].tick_params(color='xkcd:white', labelcolor='xkcd:white')
                self.ax[i, j].spines['bottom'].set_color('xkcd:white')
                self.ax[i, j].spines['top'].set_color('xkcd:white')
                self.ax[i, j].spines['right'].set_color('xkcd:white')
                self.ax[i, j].spines['left'].set_color('xkcd:white')

        if title is not None:
            self.fig.suptitle(title, color='xkcd:white', y=1.0)
        
        self.fig.set_tight_layout(True)

    
    def plot(self, x, y, row, col, title, xlabel, ylabel):

        self.ax[row, col].plot(x, y)
        self.ax[row, col].set_title(title, color='xkcd:white')
        self.ax[row, col].set_xlabel(xlabel, color='xkcd:white')
        self.ax[row, col].set_ylabel(ylabel, color='xkcd:white')

    
    def make_legend(self, labels):
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.ax[i, j].legend(labels)


if __name__ == '__main__':

    start_states = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    final_states = np.array([
        [5, 0, 0],
        [10, 0, 0],
        [15, 0, 0]
    ])
    move_time = 10
    frequency = 100

    mj = MinimumJerk(start_states, final_states, move_time, frequency)

    positions = mj.positions
    velocities = mj.velocities
    accelerations = mj.accelerations
    jerks = mj.jerks

    time = np.linspace(0, move_time, mj.N)

    plotter = Plotter(2, 2, (7,7))
    xlabel = 'time'

    plotter.plot(time, positions.T, 0, 0, 'Position', xlabel, 'positions')
    plotter.plot(time, velocities.T, 0, 1, 'Velocity', xlabel, 'velocity')
    plotter.plot(time, accelerations.T, 1, 0, 'Acceleration', xlabel, 'acceleration')
    plotter.plot(time, jerks.T, 1, 1, 'Jerk', xlabel, 'jerk')

    dimension_labels = ['x', 'y', 'z']
    plotter.make_legend(dimension_labels)

    plt.tight_layout()
    plt.show()