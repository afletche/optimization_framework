'''
A class for attaching a Newton Optimizer object.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from dynamic_fea.optimizers.optimizer import Optimizer

class NewtonOptimizer(Optimizer):

    def __init__(self) -> None:
        super().__init__()

    
    '''
    Sets initial guess.
    '''
    def set_initial_guess(self, x0):
        self.x0 = x0
        self.x = x0.copy()


    '''
    Sets up the optimizer.
    '''
    def setup(self):
        super().setup()

    '''
    Runs the Newton update.

    @param model_outputs the model outputs in form [f, c, df_dx, dc_dx, d2f_dx2, dl_dx, kkt]
    '''
    def evaluate(self, model_outputs):
        super().evaluate(model_outputs)

        self.delta_x = np.linalg.solve(self.d2f_dx2, -self.df_dx)
        self.x += self.delta_x
        
        return self.x


    def check_convergence(self, grad_norm_abs_tol):
        if np.linalg.norm(self.df_dx) < grad_norm_abs_tol:
            return True
        else:
            return False



    '''
    If num_dv == 1 or 2, creates a plot of the objective vs. dv value(s)
    '''
    def plot(self, x_history, f_history, c_history, model):
        self.num_dv = len(self.x)
        if self.num_dv == 1:
            pass
        elif self.num_dv == 2:
            plot_x1 = x_history[:,0]
            plot_x2 = x_history[:,1]
            plot_f = f_history
            plot_c = c_history

            plt.figure()
            plt.plot(plot_x1, plot_x2, '-bo')
            plt.scatter(self.solution_x[0], self.solution_x[1], s=30, c='r')# , label='Solution Found')
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.title(f'Newton Optimization path with x0={self.x0}')

            # adding contours
            # x1_sampling = np.linspace(np.min(x_history[:,0]), np.max(x_history[:,0]), 100)
            x1_sampling = np.linspace(2.-6., 2.+6, 500)
            # x2_sampling = np.linspace(np.min(x_history[:,1]), np.max(x_history[:,1]), 100)
            x2_sampling = np.linspace(4.+6, 4-6., 500)
            x_mesh_grid, y_mesh_grid = np.meshgrid(x1_sampling, x2_sampling)
            meshgrid_objectives = np.zeros_like(x_mesh_grid)
            for i, x1 in enumerate(x1_sampling):
                for j, x2 in enumerate(x2_sampling):
                    x = np.array([x1,x2])
                    meshgrid_objectives[i,j] = model(x)[0]

            plt.contour(x1_sampling, x2_sampling, meshgrid_objectives, levels=[0., 0.001, 0.05, 50 , 150, 300, 1000, 5000, 10000, 30000])#, 50000, 100000])
           
            plt.show()
        else:
            pass

