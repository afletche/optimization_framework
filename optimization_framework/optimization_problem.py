'''
A framework for setting up optimization problems.
'''

import time
import numpy as np
from optimization_framework.optimizers.backtracking_line_search import BacktrackingLineSearch
from optimization_framework.optimizers.finite_difference_line_search import FiniteDifferenceLineSearch
from optimization_framework.optimizers.secant_line_search import SecantLineSearch
from optimization_framework.optimizers.gradient_finite_difference_line_search import GradientFiniteDifferenceLineSearch
from optimization_framework.optimizers.finite_difference import finite_difference

class OptimizationProblem:
    def __init__(self) -> None:
        self.iteration_number = 0
        self.model_evaluations = 0
        self.x_history = None
        self.f_history = None
        self.c_history = None

        self.solution_x = None
        self.solution_f = np.Inf
        self.solution_c = None

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_model(self, model):
        self.model = model
        if callable(model):
            self.model_type = 'function'
        else:
            self.model_type = 'object'

    def setup(self):
        if self.model_type == 'object':
            self.model.setup()
        self.optimizer.setup()

    '''
    Runs the optimization problem.

    @param max_iter the maximum number of optimization iterations.
    '''
    def run(self, line_search=None, objective_penalty=0., updating_penalty=False, max_iter=100000, grad_norm_abs_tol=1e-5, delta_x_abs_tol=1e-5):
        x = self.optimizer.x0
        rho = objective_penalty
        for iteration_number in range(max_iter):
            if updating_penalty:
                if np.mod(iteration_number, 10) == 0:
                    rho *= 10
            self.iteration_number += 1
            # print(self.iteration_number, x)
            

            model_outputs = self.evaluate_model(x, rho)
            self.model_evaluations += 1
            self.add_to_history(x, model_outputs)

            if line_search is not None:
                self.model_evaluations += 1
                if line_search == 'FD':
                    line_searcher = FiniteDifferenceLineSearch(x, model_outputs[0], model_outputs[2])
                elif line_search == 'GFD':
                    line_searcher = GradientFiniteDifferenceLineSearch(x, model_outputs[0], model_outputs[2])
                elif line_search == "Secant":
                    line_searcher = SecantLineSearch(x, model_outputs[0], model_outputs[2])
                elif line_search == "Backtracking":
                    line_searcher = BacktrackingLineSearch(x, model_outputs[0], model_outputs[2])
                else:
                    print("Line search is defaulting to FD.")
                    line_searcher = FiniteDifferenceLineSearch(x, model_outputs[0], model_outputs[2])
                line_search_is_converged = False
                # line_search_iterations = 0
                max_line_search_iterations = 100
                for line_search_iteration in range(max_line_search_iterations):
                    model_outputs = self.evaluate_model(x, rho)
                    x = line_searcher.evaluate(model_outputs)
                    # print(f'Line Search for Iteration {self.iteration_number}: ', x)
                    self.model_evaluations += 1
                    # line_search_iterations += 1
                    line_search_is_converged = line_searcher.check_convergence()
                    if line_search_is_converged:
                        break
                self.optimizer.x = x.copy()
                self.optimizer.evaluate(model_outputs)  # this is just for the sake of having a gradient stored in the optimizer
            else:
                x = self.optimizer.evaluate(model_outputs)

            delta_x_norm = np.Inf
            if iteration_number > 1:
                delta_x_norm = np.linalg.norm(self.x_history[-1,:] - self.x_history[-2,:])
            is_converged = self.optimizer.check_convergence(grad_norm_abs_tol, delta_x_abs_tol)
            if is_converged or delta_x_norm < delta_x_abs_tol:
                print("Optimizer has converged.")
                self.solution_x = self.optimizer.solution_x
                self.solution_f = self.optimizer.solution_f
                self.solution_c = self.optimizer.solution_c

                return

        print('Optimizer reach max number of iterations')
        self.solution_x = self.optimizer.solution_x
        self.solution_f = self.optimizer.solution_f
        self.solution_c = self.optimizer.solution_c
        self.report()
        # return self.report()  Not working for some reason TODO
        return


    def evaluate_model(self, x, rho=0.):
        # t0 = time.time()
        if self.model_type == 'function':
            model_outputs = self.model(x, rho)
        elif self.model_type == 'object':
            model_outputs = self.model.evaluate(x, rho)

        t1 = time.time()
        # print('model evaluation time: ', t1-t0)

        if model_outputs[2] is None:    # If gradient is not calculated
            if self.model.h is None:
                self.model.h = 1e-8
            df_dx = finite_difference(self.model, x, rho, h=self.model.h, model_type=self.model_type)
            model_outputs[2] = df_dx

        # t2 = time.time()
        # print('FD time: ', t2-t1)

        return model_outputs



    '''
    Saves the information of the current iteration.
    '''
    def add_to_history(self, x, model_outputs):
        if self.x_history is None:
            self.x_history = x.copy()
            self.f_history = model_outputs[0]
            self.c_history = model_outputs[1]
        else:
            self.x_history = np.vstack((self.x_history, x))
            self.f_history = np.append(self.f_history, model_outputs[0])
            self.c_history = np.vstack((self.c_history, model_outputs[1]))

    
    '''
    Clears the optimization history.
    '''
    def clear_history(self):
        self.iteration_number = 0
        self.model_evaluations = 0
        self.x_history = None
        self.f_history = None
        self.c_history = None


    '''
    Prints and returns the solution.
    '''
    def report(self, solution=True, history=False):
        if history == True:
            print('design variables history: ', self.x_history)
            print('objective value history: ', self.f_history)
            print('constraint values history:', self.c_history)
            print("Number of iterations: ", self.iteration_number)
        if solution == True:
            print("Number of iterations: ", self.iteration_number)
            print("Number of function evaluations: ", self.model_evaluations)
            print('Solution design variables: ', self.solution_x)
            print('Solution objective value: ', self.solution_f)
            print('Solution constraint values:', self.solution_c)
            return [self.solution_x, self.solution_f, self.solution_c]

    
    '''
    Calls the optimizers plot function.
    '''
    def plot(self):
        self.optimizer.plot(self.x_history, self.f_history, self.c_history, self.model)

