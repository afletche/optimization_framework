'''
A class for attaching a Backtracking Line Search Optimizer object.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from dynamic_fea.optimizers.optimizer import Optimizer

class FiniteDifferenceLineSearch(Optimizer):

    def __init__(self, x0, f0, df_dx0) -> None:
        super().__init__()
        self.x0 = x0
        self.x = x0.copy()
        self.f0 = f0
        self.df_dx0 = df_dx0
        self.search_direction = -df_dx0/np.linalg.norm(df_dx0)
        self.initial_step = 1e-6
        self.eval_num = 0
        self.is_converged = False

        self.f_history = np.zeros((3,))
        self.x_history = np.zeros((3,) + x0.shape)
        self.x_history[0,:] = x0.copy()
        self.dfdx_search_direction = 1


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

        self.f_history[2] = self.f_history[1].copy()
        self.f_history[1] = self.f_history[0].copy()
        self.f_history[0] = self.f.copy()

        if self.eval_num == 1:
            self.delta_x_dist = -self.initial_step     # take initial step for central fd (-h)
            self.x_mid = self.x.copy()  # this step is the x (as opposed to x+h or x-h), so using this makes it central difference
        elif self.eval_num == 2:
            self.delta_x_dist = 2*self.initial_step     # take another initial step for fd (+h)
        else:
            self.eval_num = 0   # resetting cycle so next 3 evalualtions will be made to calc the fd
            self.dfdx_search_direction = (self.f_history[0]-self.f_history[1])/(2*self.initial_step)
            self.d2fdx2_search_direction = (self.f_history[0] - 2*self.f_history[2] + self.f_history[1])/(self.initial_step**2)
            # print('f1', self.dfdx_search_direction)
            # print('f2', self.d2fdx2_search_direction)

            # self.delta_x_dist = -self.dfdx_search_direction/np.abs(self.d2fdx2_search_direction)
            self.delta_x_dist = -self.dfdx_search_direction/np.abs((self.d2fdx2_search_direction))
            # print('delta_x', self.delta_x_dist)

        if self.eval_num == 0:
            if np.abs(self.delta_x_dist) < 1e-5:
                # print('clipping...')
                self.x = self.x_mid + 1e-5*-self.df_dx
            else:
                self.x = self.x_mid + self.delta_x_dist*self.search_direction
        else:
            self.x = self.x_history[0] + self.delta_x_dist*self.search_direction
        
        self.x_history[1,:] = self.x_history[0].copy()
        self.x_history[0,:] = self.x.copy()

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

