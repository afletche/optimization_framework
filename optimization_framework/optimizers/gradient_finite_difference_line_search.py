'''
A class for attaching a Backtracking Line Search Optimizer object.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from optimization_framework.optimizers.optimizer import Optimizer

class GradientFiniteDifferenceLineSearch(Optimizer):

    def __init__(self, x0, f0, df_dx0) -> None:
        super().__init__()
        self.x0 = x0
        self.x = x0.copy()
        self.f0 = f0
        self.df_dx0 = df_dx0
        self.search_direction = -df_dx0/np.linalg.norm(df_dx0)
        self.initial_step = 1e-7
        self.eval_num = 0
        self.is_converged = False

        self.df_dx_history = np.zeros((3,))
        self.x_history = np.zeros((3,) + x0.shape)
        self.x_history[0,:] = x0.copy()
        self.dfdx_search_direction = 1

        self.num_evaluations = 0


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


        self.df_dx_history[1] = self.df_dx_history[0].copy()
        self.df_dx_history[0] = self.df_dx.copy().dot(self.search_direction)

        if self.eval_num == 1:
            self.delta_x_dist = self.initial_step     # take initial step for central fd (-h)
        else:
            self.eval_num = 0   # resetting cycle so next 3 evalualtions will be made to calc the fd
            self.dfdx_search_direction = self.df_dx_history[0]
            self.d2fdx2_search_direction = (self.df_dx_history[0]-self.df_dx_history[1])/self.initial_step
            # print('f1', self.dfdx_search_direction)
            # print('f2', self.d2fdx2_search_direction)

            # self.delta_x_dist = -self.dfdx_search_direction/np.abs(self.d2fdx2_search_direction)
            self.delta_x_dist = -self.dfdx_search_direction/np.abs((self.d2fdx2_search_direction))
            # print('delta_x', self.delta_x_dist)

        print(self.num_evaluations)
        print('eval_num', self.eval_num)
        if self.eval_num == 0:
            if np.abs(self.delta_x_dist) < 1e-8:
                self.x = self.x + 1e-8*-self.df_dx
            else:
                self.x = self.x + self.delta_x_dist*self.search_direction
        else:
            self.x = self.x_history[0] + self.delta_x_dist*self.search_direction
        
        self.x_history[1,:] = self.x_history[0].copy()
        self.x_history[0,:] = self.x.copy()

        self.num_evaluations += 1

        return self.x


    def check_convergence(self):
        if np.abs(self.dfdx_search_direction) < 1e-1:
            return True
        else:
            return False



    '''
    If num_dv == 1 or 2, creates a plot of the objective vs. dv value(s)
    '''
    def plot(self, x_history, f_history, c_history, model):
        print("Sorry, plotting for line searches has not been set up yet.")

