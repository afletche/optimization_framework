'''
TODO
A base class for optimizers.
'''

import numpy as np

class Optimizer:
    def __init__(self) -> None:
        self.iteration_number = 0
        self.is_converged = False

        self.solution_f = np.Inf
        self.solution_x = None
        self.solution_c = None


    '''
    Sets up the optimizer.
    '''
    def setup(self):
        pass


    '''
    Evaluates the optimizer (calculates the new set of design variables.)

    @param model_outputs the outputs of the model.
    @return x the design_variables (which are the model inputs)
    '''
    def evaluate(self, model_outputs):
        self.f = model_outputs[0]
        self.c = model_outputs[1]
        self.df_dx = model_outputs[2]
        self.dc_dx = model_outputs[3]
        self.d2f_dx2 = model_outputs[4]
        self.dl_dx = model_outputs[5]
        self.kkt = model_outputs[6]

        self.solution_f = model_outputs[0].copy()
        self.solution_c = model_outputs[1].copy()
        self.solution_x = self.x.copy()


    '''
    Checks to see if the optimizer has converged.
    '''
    def check_convergence(self):
        return False


    '''
    Creates plots relevent to the optimizer.
    '''
    def plot(self):
        pass

