'''
An optimizer for running brute force optimizations.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from dynamic_fea.optimizers.optimizer import Optimizer

class GridSearchOptimizer(Optimizer):

    def __init__(self) -> None:
        super().__init__()
        self.bounds = None


    '''
    Sets the range on the brute force optimization.
    - format: [[lower_1, upper_1, np1], [lower_2, upper_2, np2], ...]

    @param bounds the bounds of the range of design variable values.
    '''
    def set_bounds(self, bounds) -> None:
        self.bounds = bounds
        self.num_dv = len(bounds)
        self.axes = []
        for i in range(self.num_dv):
            lower = bounds[i][0]
            upper = bounds[i][1]
            num_div = bounds[i][2]
            self.axes.append(np.linspace(lower, upper, num_div))

        self.indexing_vars = np.zeros(self.num_dv).astype(int)

        self.x = np.zeros(self.num_dv)


    '''
    Updates the mesh indexing variables.
    '''
    def update_indexing_vars(self): 
        for i in range(self.num_dv):
            self.indexing_vars[-(i+1)] += 1
            if self.indexing_vars[-(i+1)] == len(self.axes[-(i+1)]):
                self.indexing_vars[-(i+1)] = 0
            else:
                return
        
        # If not returned yet, then all cycle has been completed.
        self.is_converged = True

    
    '''
    Sets up the optimizer.
    '''
    def setup(self):
        super().setup()
        self.x0 = self.x
        for i in range(len(self.axes)):            
            self.x0[i] = self.axes[i][0]

        self.x = self.x0

        # self.update_indexing_vars()


    '''
    Runs the brute force optimization update.

    @param model_outputs the model outputs in form [f, c]
    '''
    def evaluate(self, model_outputs):
        if self.bounds is None:
            Exception('Plase call the set_bounds method before evaluating.')

        if model_outputs[0] < self.solution_f and model_outputs[1] <= 0:
            self.solution_f = model_outputs[0].copy()
            self.solution_c = model_outputs[1].copy()
            self.solution_x = self.x.copy()

        self.update_indexing_vars()

        for i in range(len(self.axes)):
            self.x[i] = self.axes[i][self.indexing_vars[i]]
        
        
        return self.x


    def check_convergence(self):
        return self.is_converged



    '''
    If num_dv == 1 or 2, creates a plot of the objective vs. dv value(s)
    '''
    def plot(self, x_history, f_history, c_history):
        if self.num_dv == 1:
            pass
        elif self.num_dv == 2:
            plot_x1 = x_history[:,0].reshape((self.bounds[0][2], self.bounds[1][2]))
            plot_x2 = x_history[:,1].reshape((self.bounds[0][2], self.bounds[1][2]))
            plot_f = f_history.reshape((self.bounds[0][2], self.bounds[1][2]))
            plot_c = c_history.reshape((self.bounds[0][2], self.bounds[1][2]))

            plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot_surface(plot_x1, plot_x2, plot_f, cmap=cm.coolwarm)# , label='Objective Surface')
            ax.plot_surface(plot_x1, plot_x2, plot_c, cmap='Greens_r')# , label='Constraint Surface')
            ax.scatter(self.solution_x[0], self.solution_x[1], self.solution_f, s=100, c='k')# , label='Solution Found')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('objective')
            ax.set_title('Optimization Results')
            plt.show()
        else:
            print("Brute force optimizer will only plot for 1 or 2 dv.")

