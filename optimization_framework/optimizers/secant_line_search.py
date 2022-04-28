'''
A class for attaching a Backtracking Line Search Optimizer object.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from optimization_framework.optimizers.optimizer import Optimizer

class SecantLineSearch(Optimizer):

    def __init__(self, x0, f0, df_dx0) -> None:
        super().__init__()
        self.x0 = x0
        self.x = x0.copy()
        self.f0 = f0
        self.df_dx0 = df_dx0
        self.search_direction = -df_dx0/np.linalg.norm(df_dx0)
        self.initial_step = 1e-3
        self.eval_num = 0
        self.is_converged = False

        self.f_history = np.zeros((2,))
        self.x_history = np.zeros((2,) + x0.shape)
        self.x_history[0,:] = x0.copy()
        self.dfdx_search_direction_history = np.ones(2,)


    '''
    Sets up the optimizer.
    '''
    def setup(self):
        super().setup()

    '''
    Runs the "Newton" (using secants) update.

    @param model_outputs the model outputs in form [f, c, df_dx, dc_dx, d2f_dx2, dl_dx, kkt]
    '''
    def evaluate(self, model_outputs):
        self.eval_num += 1
        super().evaluate(model_outputs)

        self.f_history[1] = self.f_history[0].copy()
        self.f_history[0] = self.f.copy()

        if self.eval_num == 1:
            self.delta_x_dist = self.initial_step     # take initial stte for fd
        elif self.eval_num == 2:
            # fd 1st derivative
            self.dfdx_search_direction_history[0] = (self.f_history[0] - self.f_history[1])/self.delta_x_dist

            self.delta_x_dist = self.initial_step     # take another initial step for fd (to get 2nd order fd)
        else:
            # fd 1st derivative (x)
            self.dfdx_search_direction_history[1] = self.dfdx_search_direction_history[0].copy()
            self.dfdx_search_direction_history[0] = (self.f_history[0] - self.f_history[1])/self.delta_x_dist
            print(self.eval_num, self.dfdx_search_direction_history[0])

            # fd 2nd derivative (not performed because it's built into the "Newton" update.)
            # self.d2fdx2_search_direction = (self.dfdx_search_direction_history[1] - self.dfdx_search_direction_history[0])/(self.x_history[1]-self.x_history[0])

            self.delta_x_dist = -(self.dfdx_search_direction_history[0] * np.abs(self.delta_x_dist)) / np.abs((self.dfdx_search_direction_history[0] - self.dfdx_search_direction_history[1]))

        self.x = self.x_history[0] + self.delta_x_dist*self.search_direction
        
        self.x_history[1,:] = self.x_history[0].copy()
        self.x_history[0,:] = self.x.copy()

        return self.x


    def check_convergence(self):
        if np.linalg.norm(self.dfdx_search_direction_history[0]) < 1e-5:
            return True
        else:
            return False



    '''
    If num_dv == 1 or 2, creates a plot of the objective vs. dv value(s)
    '''
    def plot(self, x_history, f_history, c_history, model):
        print("Sorry, plotting for line searches has not been set up yet.")

