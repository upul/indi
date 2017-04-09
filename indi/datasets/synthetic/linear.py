import numpy as np


def generate_dataset(equation,
                       min_limit=0,
                       max_limit=10,
                       noise_mean=0,
                       noise_var=1,
                       number_of_items=100):
    x = np.linspace(min_limit, max_limit, number_of_items)
    y = np.zeros(number_of_items)
    for i in range(number_of_items):
        y[i] = equation(x[i]) + np.random.normal(noise_mean, noise_var)
    return x, y