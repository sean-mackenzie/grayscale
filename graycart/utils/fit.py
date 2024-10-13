import numpy as np
import pandas as pd

from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def line(x, a, b):
    return a * x + b


def parabola(x, a, b, c):
    return a * x ** 2 + b * x + c


def parabola_origin(x, a, b):
    return a * x ** 2 + b * x


def exp_four(x, a, b, c, d, e):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e


def power_law(x, a, b):
    return a * x ** b


# ----------------------------------------------------------------------------------------------------------------------

def fit_curve(fit_func, x, y):
    """
    popt, fit_func = fit.fit_curve('line', x, y)
    """
    # instantiate fit function
    if fit_func == 'line':
        fit_func = line
    elif fit_func == 'parabola':
        fit_func = parabola
    elif fit_func == 'parabola_origin':
        fit_func = parabola_origin
    elif fit_func == 'exp_four':
        fit_func = exp_four
    else:
        raise ValueError("Fit function can be: {}".format(['line', 'parabola', 'parabola_origin', 'exp_four']))

    popt, pcov = curve_fit(fit_func, x, y)

    return popt, fit_func

def fit_single_peak(df, peak_properties, peak_properties_idx, width_space, fit_func):
    """

    :param df:
    :param peak_properties:
    :param width_space: [50, 1250]
    :param fit_func:
    :return:
    """

    # parse peak properties
    peak_properties = {'peak_heights': peak_properties['peak_heights'][peak_properties_idx],
                       'prominences': peak_properties['prominences'][peak_properties_idx],
                       'left_bases': peak_properties['left_bases'][peak_properties_idx],
                       'right_bases': peak_properties['right_bases'][peak_properties_idx],
                       'widths': peak_properties['widths'][peak_properties_idx],
                       'width_heights': peak_properties['width_heights'][peak_properties_idx],
                       'left_ips': peak_properties['left_ips'][peak_properties_idx],
                       'right_ips': peak_properties['right_ips'][peak_properties_idx],
                       }

    # helper rename
    lips, rips = peak_properties["left_ips"], peak_properties["right_ips"]

    # instantiate fit function
    if fit_func == 'parabola':
        fit_func = parabola
    elif fit_func == 'parabola_origin':
        fit_func = parabola_origin
    elif fit_func == 'exp_four':
        fit_func = exp_four
    else:
        raise ValueError("Fit function can be: {}".format(['parabola', 'parabola_origin', 'exp_four']))

    # fit function to "true-est" peak
    il, ir = int(lips - width_space), int(rips + width_space)
    xf = np.arange(il, ir)
    zf = df.z.iloc[il:ir].to_numpy() * -1
    popt, pcov = curve_fit(fit_func, xf, zf)
    xf_pk_idx = np.argmax(fit_func(xf, *popt))

    # compute fit function
    res_x = np.linspace(xf.min(), xf.max(), 50)
    res_z = fit_func(res_x, *popt)

    # find center of fitted parabola
    pk_idx = il + xf_pk_idx
    pk_xc = df.x.iloc[pk_idx]
    pk_height = df.z.iloc[pk_idx]
    pk_radius = (df.x.iloc[ir] - df.x.iloc[il]) / 2

    fit_properties = {'width_space': width_space,  # arbitrary padding outside of 'left_ips' and 'right_ips'
                      'il': il,  # index value: left_ips - width_space
                      'ir': ir,  # index value: right_ips + width_space
                      'fit_x': xf,  # index array (0, 1, 2, 3,...) from 'il' to 'ir'
                      'fit_z': zf,  # z-scan data over index array, 'il' to 'ir'
                      'fit_func': fit_func,
                      'popt': popt,  # requires x-array = index array
                      'res_x': res_x,
                      'res_z': res_z,
                      'fit_pk_idx': xf_pk_idx,  # index of peak over index array 'fit_x', from 'il' to 'ir' in org. df.
                      'pk_idx': pk_idx,  # index of peak over original dataframe
                      'pk_xc': np.round(pk_xc, 1),  # x-coordinate @ peak in original dataframe
                      'pk_h': np.round(pk_height, 2),  # z-coordinate @ peak in original dataframe
                      'pk_r': int(np.round(pk_radius, -1)),  # x-span in original dataframe, from 'il' to 'ir'
                      }

    peak_properties.update(fit_properties)

    return peak_properties


def calculate_fit_error(fit_results, data_fit_to, fit_func=None, fit_params=None, data_fit_on=None):
    """
    To run:
    rmse, r_squared = fit.calculate_fit_error(fit_results, data_fit_to)

    Two options for calculating fit error:
        1. fit_func + fit_params: the fit results are calculated.
        2. fit_results: the fit results are known for each data point.

    Old way of doing this (updated 6/11/22):
    abs_error = fit_results - data_fit_to
    r_squared = 1.0 - (np.var(abs_error) / np.var(data_fit_to))

    :param fit_func: the function used to calculate the fit.
    :param fit_params: generally, popt.
    :param fit_results: the outputs at each input data point ('data_fit_on')
    :param data_fit_on: the input data that was inputted to fit_func to generate the fit.
    :param data_fit_to: the output data that fit_func was fit to.
    :return:
    """

    # --- calculate prediction errors
    if fit_results is None:
        fit_results = fit_func(data_fit_on, *fit_params)

    residuals = calculate_residuals(fit_results, data_fit_to)
    r_squared_me = 1 - (np.sum(np.square(residuals))) / (np.sum(np.square(fit_results - np.mean(fit_results))))

    se = np.square(residuals)  # squared errors
    mse = np.mean(se)  # mean squared errors
    rmse = np.sqrt(mse)  # Root Mean Squared Error, RMSE
    r_squared = 1.0 - (np.var(np.abs(residuals)) / np.var(data_fit_to))

    # print("wiki r-squared: {}; old r-squared: {}".format(np.round(r_squared_me, 4), np.round(r_squared, 4)))
    # I think the "wiki r-squared" is probably the correct one...
    # 8/23/22 - the wiki is definitely wrong because values range from +1 to -20...

    return rmse, r_squared


def calculate_residuals(fit_results, data_fit_to):
    residuals = fit_results - data_fit_to
    return residuals