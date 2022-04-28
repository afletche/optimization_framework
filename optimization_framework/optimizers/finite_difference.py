import numpy as np


'''
Calculates the gradient using FD.

Inputs:
- model : function or Object : the model function or object
- x : np.ndarray : The set of design variables/inputs
- h : np.ndarray : The step size to take in each direction
'''
def finite_difference(model, x, rho=0., h=1e-8, model_type='function'):
    gradient = np.zeros_like(x)
    if np.isscalar(h):
        h = np.ones_like(x)*h
    
    model_outputs_at_x = evaluate_model(model, x, model_type=model_type)
    f_x = model_outputs_at_x[0]

    for i, x_i in enumerate(x):   # for each design variable, take a step
        x_plus_h = x.copy()
        x_plus_h[i] = x_i + h[i]

        model_outputs_at_x_plus_h = evaluate_model(model, x_plus_h, model_type=model_type)
        f_x_plus_h = model_outputs_at_x_plus_h[0]
        gradient[i] = (f_x_plus_h - f_x)/h[i]

    return gradient


def evaluate_model(model, x, model_type='function'):
    if model_type == 'function':
        model_outputs = model(x)
    elif model_type == 'object':
        model_outputs = model.evaluate(x)
    else:
        Exception('WARNING: neither function nor model fed into FD')

    return model_outputs