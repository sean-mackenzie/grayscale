import numpy as np
import pandas as pd
from scipy.signal import find_peaks, resample, correlate
from scipy.interpolate import splev, splrep, interp1d

from graycart.utils import fit

import matplotlib.pyplot as plt

# ---


def universal_dose_to_depth(exposure_intensity, a=0.0, b=0.0, c=0.0003, d=-0.0728, e=3.15):
    return fit.exp_four(exposure_intensity, a, b, c, d, e)


def backout_process_from_target(df, px, py, pys, thickness_PR, thickness_PR_budget, r_target=5, did=None):
    """

    est_process_flow = process.estimate_process_to_achieve_target(
    df, px, py, pys='z_surf', thickness_PR=6.5, thickness_PR_budget=0.5, r_target=10,
    )

    """

    # TARGET profile
    df_pred = df[[px, py]]

    # create sections, where photoresist would be, as extensions of target profile
    extend_target_radius = 1000
    sampling_rate = (df_pred[px].max() - df_pred[px].min()) / len(df_pred)
    px_ext = np.arange(extend_target_radius // sampling_rate) + df_pred[px].max() + sampling_rate
    py_ext = np.zeros_like(px_ext)
    df_ext = pd.DataFrame(np.vstack([np.hstack([-px_ext, px_ext]), np.hstack([py_ext, py_ext])]).T, columns=[px, py])

    # combine target profile and extensions; NOTE: at this point, df_pred is the silicon profile after stripping
    df_pred = pd.concat([df_pred, df_ext]).sort_values(px).reset_index()

    # create 'z_surface' column by replacing silicon surface (y = 0) with photoresist budget (y = thickness_PR_budget)
    df_pred[pys] = df_pred[py].where(df_pred[py] < 0, thickness_PR_budget)

    # ---

    # etch rates
    etch_rate_SF6_O2_Si = -4 / 60  # um/min --> negative etch rate b/c running a 'reverse' etch.
    etch_rate_SF6_O2_PR = -0.2 / 60  # um/min

    # ---

    # initialize
    etch_recipe = 'SF6+O2.S20'
    etch_rate_Si = etch_rate_SF6_O2_Si
    etch_rate_PR = etch_rate_SF6_O2_PR

    est_process_flow = {}
    for i in np.arange(2):

        # OF TARGET: calculate y-min
        y_min = df_pred[df_pred[px].abs() < r_target][pys].mean()  # a negative number (e.g., -50 um)
        y_max = df_pred[pys].max()  # a positive number (e.g., 0.5 um)

        if y_min >= 0 or y_max > thickness_PR:
            continue

        # etch time
        etch_time = y_min / etch_rate_Si  # etch_time = positive number

        # re-calculate etch time if PR thickness < PR thickness budget
        if np.abs(etch_rate_PR * etch_time) > thickness_PR - thickness_PR_budget:
            etch_time = (thickness_PR_budget - thickness_PR) / etch_rate_PR  # = positive number

        # calculate etch depths
        etch_depth_Si = etch_rate_Si * etch_time  # etch_depth_Si = negative number
        # etch_depth_PR = etch_rate_PR * etch_time  # etch_depth_PR = negative number

        # apply the etch

        # store a copy of the pre-reverse etch profile, where:
        #   1. z_surf > 0: remains the same (i.e., z_surf of photoresist)
        #   2. z_surf < 0: becomes = 0.
        df_pre_process = df_pred.copy()
        df_pre_process[pys] = df_pre_process[pys].where(df_pre_process[pys] > 0, 0)

        # store a copy of the post-reverse etch profile, where:
        #   * the etch rate of all materials is defined by 'etch_rate_..._PR'
        df_process = df_pred.copy()
        df_process[pys] = df_process[pys] - etch_depth_Si

        # calculate the z-distance ('dz_pos') that was etched into the photoresist
        #   * z-distance = difference between post-process and pre-process profiles.
        # 'dz_neg' = (df_pred[py] - etch_depth_PR) - df_pred[py].where(df_pred[py] < 0, 0)
        """
        Example 1: 
            * df_pred[py] = 3, etch time = 5, etch depth = 2, df_pred[py].where() = 0
            * dz_neg = (3 - 2) - 0 = 1
            * dz_neg = 0
            * etch_time_Si = 0
            * etch_time_PR = 5
            * df_pred[py] = 3 - etch_rate_PR * 5 --> CORRECT!

        Example 2:
            * df_pred[py] = 1, etch time = 5, etch depth = 2, df_pred[py].where() = 0
            * dz_neg = (1 - 2) - 0 = -1  
            * dz_neg = -1 
            * etch_time_Si = 1 / etch_rate_PR
            * etch_time_PR = 5 - etch_time_Si
            * df_pred[py] = df_pred[py] - time_PR * rate_PR - time_Si * rate_Si --> CORRECT!

        Example 3:
            * df_pred[py] = -1, etch time = 5, etch depth = 2, df_pred[py].where() = -1
            * dz_neg = (-1 - 2) - -1 = -2  
            * dz_neg = -2 
            * etch_time_Si = 2 / etch_rate_PR = 5
            * etch_time_PR = 5 - etch_time_Si = 0
            * df_pred[py] = df_pred[py] - time_PR * rate_PR - time_Si * rate_Si --> CORRECT!
        """
        df_process['dz_pos'] = df_process[pys] - df_pre_process[pys]
        df_process['dz_pos'] = df_process['dz_pos'].where(df_process['dz_pos'] > 0, 0)

        # NOTE: 'etch_rate_SF6_O2_PR' could be replaced with 'etch_rate_PR' because they are identical
        df_process['etch_time_PR'] = np.abs(df_process['dz_pos'] / etch_rate_Si)
        df_process['etch_time_Si'] = etch_time - df_process['etch_time_PR']

        df_pred[pys] = df_pred[pys] - df_process['etch_time_Si'] * etch_rate_Si - df_process['etch_time_PR'] * etch_rate_PR

        # multiply 'dz_neg' by Si:PR etch rate selectivity to determine the correct etch depth into silicon.
        #   * this essentially solves for the amount of time that 'etch_rate_..._Si' was applied to Si.
        # df_pred[py] = df_pred[py] - etch_depth_PR + df_process['dz_neg'] * selectivity

        # re-calculate y-min and y-max
        y_min = df_pred[df_pred[px].abs() < r_target][pys].mean()
        y_max = df_pred[pys].max()  # a positive number (e.g., 0.5 um)

        # add process
        prcss = {'process_type': 'Etch',
                 'recipe': etch_recipe,
                 'time': etch_time,
                 'details': 'etch_rate_PR={}, etch_rate_Si={}'.format(
                     etch_rate_PR,
                     etch_rate_Si,
                     ),
                 'path': None,
                 'df': df_pred.copy(),
                 'thickness_PR': y_max,
                 'did': did,
                 }
        est_process_flow.update({i: prcss})

    return est_process_flow


def estimate_process_to_achieve_target(df, px, py, p_materials, recipe_smOOth, recipe_SF6_O2,
                                       target_depth, thickness_PR_budget=0.5, r_target=25):
    """
    est_process_flow = process.estimate_process_to_achieve_target(df, px, py, target_depth=-50, r_target=25)

    :param p_materials:
    :param recipe_smOOth:
    :param recipe_SF6_O2:
    :param df:
    :param px:
    :param py:
    :param target_depth:
    :param thickness_PR_budget: thickness of PR that triggers the etch to stop
    :param r_target: center radial distance to average over to estimate the max depth of profile
    :return:
    """

    def calc_y_min(df_pred, px, py, r_target):
        ymin = df_pred[py].min()  # df_pred[df_pred[px].abs() < r_target][py].mean()
        return ymin

    # setup PR etch
    etch_rate_smOOth = p_materials['Photoresist'].etch_rates[recipe_smOOth] / 60  # (um/s)
    etch_rate_SF6_O2_PR = p_materials['Photoresist'].etch_rates[recipe_SF6_O2] / 60  # (um/s)
    etch_rate_SF6_O2_Si = p_materials['Silicon'].etch_rates[recipe_SF6_O2] / 60  # (um/s)

    # ---

    # current profile
    df_pred = df[[px, py]]

    est_process_flow = {}
    for i in np.arange(2):

        # calculate y-min and PR thickness
        # y_min = df_pred[py].min()  # df_pred[df_pred[px].abs() < r_target][py].mean()
        y_min = calc_y_min(df_pred, px, py, r_target)
        est_thicknesss_PR = df_pred[df_pred[py] > df_pred[py].max() - thickness_PR_budget / 2][py].mean()

        if y_min <= -target_depth + 1.0 or est_thicknesss_PR < thickness_PR_budget + 0.01:
            continue

        # decide what etch to run
        if y_min > 0.05:
            etch_recipe = recipe_smOOth
            etch_rate_PR = etch_rate_smOOth

            etch_time = y_min / etch_rate_PR * 1.025
            etch_depth_PR = etch_rate_PR * etch_time

        else:
            etch_recipe = recipe_SF6_O2
            etch_rate_PR = etch_rate_SF6_O2_PR

            etch_time = (target_depth + y_min) / etch_rate_SF6_O2_Si
            etch_depth_PR = etch_rate_PR * etch_time

        # re-calculate etch time if PR thickness < PR thickness budget
        if etch_depth_PR > est_thicknesss_PR - thickness_PR_budget:
            etch_time = (est_thicknesss_PR - thickness_PR_budget) / etch_rate_PR
            etch_depth_PR = etch_rate_PR * etch_time

        # apply the etch
        if etch_recipe == recipe_smOOth:
            df_pred[py] = df_pred[py] - etch_depth_PR
            df_pred.loc[df_pred[py] < 0, py] = 0

        elif etch_recipe == recipe_SF6_O2:

            # no point in running Si etch for short time (etch depth < 5 um)
            if etch_time < 180:
                continue

            # store a copy of the pre-etch profile, where:
            #   1. z_surf < 0: remains the same (i.e., z_surf)
            #   2. z_surf > 0: becomes = 0.
            df_pre_process = df_pred.copy()
            df_pre_process[py] = df_pre_process[py].where(df_pre_process[py] < 0, 0)

            # store a copy of the post-etch profile, where:
            #   * the etch rate of all materials is defined by 'etch_rate_..._PR'
            df_process = df_pred.copy()
            df_process[py] = df_process[py] - etch_depth_PR

            # calculate the z-distance ('dz_neg') that was etched into the silicon
            #   * z-distance = difference between post-process and pre-process profiles.
            # 'dz_neg' = (df_pred[py] - etch_depth_PR) - df_pred[py].where(df_pred[py] < 0, 0)
            """
            Example 1: 
                * df_pred[py] = 3, etch time = 5, etch depth = 2, df_pred[py].where() = 0
                * dz_neg = (3 - 2) - 0 = 1
                * dz_neg = 0
                * etch_time_Si = 0
                * etch_time_PR = 5
                * df_pred[py] = 3 - etch_rate_PR * 5 --> CORRECT!
            
            Example 2:
                * df_pred[py] = 1, etch time = 5, etch depth = 2, df_pred[py].where() = 0
                * dz_neg = (1 - 2) - 0 = -1  
                * dz_neg = -1 
                * etch_time_Si = 1 / etch_rate_PR
                * etch_time_PR = 5 - etch_time_Si
                * df_pred[py] = df_pred[py] - time_PR * rate_PR - time_Si * rate_Si --> CORRECT!
                
            Example 3:
                * df_pred[py] = -1, etch time = 5, etch depth = 2, df_pred[py].where() = -1
                * dz_neg = (-1 - 2) - -1 = -2  
                * dz_neg = -2 
                * etch_time_Si = 2 / etch_rate_PR = 5
                * etch_time_PR = 5 - etch_time_Si = 0
                * df_pred[py] = df_pred[py] - time_PR * rate_PR - time_Si * rate_Si --> CORRECT!
            """
            df_process['dz_neg'] = df_process[py] - df_pre_process[py]
            df_process['dz_neg'] = df_process['dz_neg'].where(df_process['dz_neg'] < 0, 0)

            # NOTE: 'etch_rate_SF6_O2_PR' could be replaced with 'etch_rate_PR' because they are identical
            df_process['etch_time_Si'] = df_process['dz_neg'].abs() / etch_rate_SF6_O2_PR
            df_process['etch_time_PR'] = etch_time - df_process['etch_time_Si']

            df_pred[py] = df_pred[py] - df_process['etch_time_PR'] * etch_rate_SF6_O2_PR - df_process['etch_time_Si'] * etch_rate_SF6_O2_Si

            # multiply 'dz_neg' by Si:PR etch rate selectivity to determine the correct etch depth into silicon.
            #   * this essentially solves for the amount of time that 'etch_rate_..._Si' was applied to Si.
            # df_pred[py] = df_pred[py] - etch_depth_PR + df_process['dz_neg'] * selectivity

        # re-calculate y-min
        # y_min = df_pred[df_pred[px].abs() < r_target][py].mean()
        y_min = calc_y_min(df_pred, px, py, r_target)

        # add process
        prcss = {'process_type': 'Etch',
                 'recipe': etch_recipe,
                 'time': etch_time,
                 'details': 'etch_rate_smOOth={}, etch_rate_SF6_O2_PR={}, etch_rate_SF6_O2_Si={}'.format(etch_rate_smOOth,
                                                                                                         etch_rate_SF6_O2_PR,
                                                                                                         etch_rate_SF6_O2_Si,
                                                                                                         ),
                 'path': None,
                 'df': df_pred.copy(),
                 'z_min': y_min,
                 }
        est_process_flow.update({i: prcss})

    return est_process_flow


# ------

def find_single_peak(df, target_width, min_width, width_space, fit_func='parabola',
                     rel_height=0.95, prominence=1, plot_width_rel_target=1.05):
    """
    dfpk, peak_properties = process.find_single_peak(df, target_width, min_width, width_space=50, fit_func='parabola', rel_height=0.9375, prominence=1)

    :param df:
    :param width_space: 500
    :param rel_height:
    :param prominence:
    :return:
    """

    # 1. pre-process data
    x = df.x.to_numpy()
    z = df.z.to_numpy() * -1

    # 2. find peaks
    height = z.max() / 2  # minimum peak height
    distance = target_width
    peaks, peak_properties = find_peaks(z, height=height, width=min_width, distance=distance,
                                        prominence=prominence, rel_height=rel_height)

    # 3. fit function to peak
    peak_properties = fit.fit_single_peak(df, peak_properties, width_space, fit_func)

    # 4. get data
    dfpk = df[np.abs(df['x'] - peak_properties['pk_xc']) < target_width * plot_width_rel_target]

    return dfpk, peak_properties


def find_multi_peak(df, peak_labels, peak_ids, target_width, min_width, width_space, fit_func='parabola',
                    rel_height=0.95, prominence=1, plot_width_rel_target=1.1):
    """
    dfpk, peak_details = process.find_multi_peak(df, peak_labels, peak_ids, target_width, min_width, width_space=50, fit_func='parabola', rel_height=0.9375, prominence=1)

    :param df:
    :param width_space: 500
    :param rel_height:
    :param prominence:
    :return:
    """
    if not isinstance(peak_labels, (list, np.ndarray)):
        peak_labels = [peak_labels]

    if not isinstance(peak_ids, (list, np.ndarray)):
        peak_ids = [peak_ids]

    # 1. pre-process data
    x = df.x.to_numpy()
    z = df.z.to_numpy() * -1

    # 2. find peaks
    height = z.max() / 1.25  # 2  # minimum peak height
    distance = min_width * 0.5
    # min_width = min_width * 0.5
    # rel_height = 0.9
    # prominence = 0.9

    peaks, peak_properties = find_peaks(z, height=height, width=min_width, distance=distance, prominence=prominence,
                                        rel_height=rel_height)

    # 3. iterate through each peak
    peak_details = {}
    dfpks = []
    peak_idx = np.arange(len(peaks))
    for pidx, plbl, pid, pk in zip(peak_idx, peak_labels, peak_ids, peaks):
        # 4. fit function to peak
        pk_properties = fit.fit_single_peak(df, peak_properties, pidx, width_space, fit_func)

        # 5. get data
        dfpk = df[np.abs(df['x'] - pk_properties['pk_xc']) < target_width * plot_width_rel_target]
        dfpk['r'] = dfpk['x'] - pk_properties['pk_xc']
        dfpk['fid'] = pid

        # 6. store data
        dfpks.append(dfpk)
        peak_details.update({plbl: {'df': dfpk, 'peak_properties': pk_properties}})

    if len(dfpks) == 0:
        plt.plot(df.x, df.z)
        plt.show()

    dfpk = pd.concat(dfpks)

    return dfpk, peak_details


def resample_dataframe(df, xcol, ycol, num_points, sampling_rate=None):
    """
    df = process.resample_dataframe(df, xcol, ycol, num_points, sampling_rate)
    """

    # down-sample to reduce computation and file size
    x, y = resample_array(x=df[xcol].to_numpy(),
                          y=df[ycol].to_numpy(),
                          num_points=num_points,
                          sampling_rate=sampling_rate,
                          )

    data_ds = np.vstack([x, y]).T
    df = pd.DataFrame(data_ds, columns=[xcol, ycol])

    return df


def downsample_dataframe(df, xcol, ycol, num_points, sampling_rate):
    """
    df = process.downsample_dataframe(df, xcol, ycol, num_points, sampling_rate)
    """

    # down-sample to reduce computation and file size
    x, y = downsample_array(x=df[xcol].to_numpy(),
                            y=df[ycol].to_numpy(),
                            num_points=num_points,
                            sampling_rate=sampling_rate,
                            )

    data_ds = np.vstack([x, y]).T
    df = pd.DataFrame(data_ds, columns=[xcol, ycol])

    return df


def interpolate_dataframe(df, xcol, ycol, num_points, sampling_rate=None):
    """
    df = process.interpolate_dataframe(df, xcol, ycol, num_points, sampling_rate=None)
    """

    # down-sample to reduce computation and file size
    x, y = interpolate_array(x=df[xcol].to_numpy(),
                             y=df[ycol].to_numpy(),
                             num_points=num_points,
                             sampling_rate=sampling_rate,
                             )

    data_ds = np.vstack([x, y]).T
    df = pd.DataFrame(data_ds, columns=[xcol, ycol])

    return df


def integrate_dataframe_radial(df, ycol, xcol, num_slices):
    """
    V = process.integrate_dataframe_radial(df, ycol, xcol, num_slices)
    """
    h = np.linspace(0, df[ycol].min(), num_slices)
    dh = np.abs(h[1])
    V = 0
    for i in range(len(h) - 1):
        dV = np.pi * df[(df[ycol] <= h[i]) & (df[ycol] > h[i + 1])][xcol].abs().mean() ** 2 * dh
        if np.isnan(dV):
            dV = 0
        V += dV
    return V * 1e-6


def fit_func_dataframe(df, xcol, ycol, fit_func, num_points):
    """
    df, fit_func, popt = process.fit_func_dataframe(df, xcol, ycol, fit_func, num_points)
    """
    x = df[xcol].to_numpy()
    y = df[ycol].to_numpy()

    popt, fit_func = fit.fit_curve(fit_func, x, y)

    x2 = np.linspace(x.min(), x.max(), num_points)
    y2 = fit_func(x2, *popt)

    data_ds = np.vstack([x2, y2]).T
    df = pd.DataFrame(data_ds, columns=[xcol, ycol])

    return df, fit_func, popt


def resample_array(x, y, num_points, sampling_rate=None):
    """
    x, y = process.resample_array(x, y, num_points, sampling_rate)
    """

    if sampling_rate is not None:
        x_span = x.max() - x.min()
        num_points = x_span / sampling_rate
    elif num_points is None:
        raise ValueError("Must provide either 'num_points' or 'sampling_rate'.")

    num_points = int(num_points)

    y2 = resample(y, num_points)
    x2 = np.linspace(x.min(), x.max(), num_points)

    return x2, y2


def downsample_array(x, y, num_points, sampling_rate):
    """
    x, y = process.downsample_array(x, y, num_points, sampling_rate)
    """

    if sampling_rate is not None:
        x_span = x.max() - x.min()
        num_points = x_span / sampling_rate
    elif num_points is None:
        raise ValueError("Must provide either 'num_points' or 'sampling_rate'.")

    num_points = int(num_points)

    sp1 = splrep(x, y)
    x2 = np.linspace(x.min(), x.max(), num_points)
    y2 = splev(x2, sp1)

    return x2, y2


def interpolate_array(x, y, num_points, sampling_rate=None):
    """
    x, y = process.interpolate_array(x, y, num_points, sampling_rate)
    """

    if sampling_rate is not None:
        x_span = x.max() - x.min()
        num_points = x_span / sampling_rate
    elif num_points is None:
        raise ValueError("Must provide either 'num_points' or 'sampling_rate'.")

    num_points = int(num_points)

    f = interp1d(x, y)
    x2 = np.linspace(x.min(), x.max(), num_points)
    y2 = f(x2)

    return x2, y2


def tilt_correct_array(x, y, num_points):
    """
    y_corr, fit_func, popt, rmse, r_squared = process.tilt_correct_array(x, y, num_points)
    """
    # tilt_y = np.mean(y[-num_points:]) - np.mean(y[:num_points])
    # tilt_x = np.mean(x[-num_points:]) - np.mean(x[:num_points])
    # tilt_deg = np.rad2deg(np.arcsin(tilt_y / tilt_x))

    x_edges = np.hstack([x[:num_points], x[-num_points:]])
    y_edges = np.hstack([y[:num_points], y[-num_points:]])

    popt, fit_func = fit.fit_curve('line', x_edges, y_edges)
    y_corr = y - fit_func(x, *popt)

    rmse, r_squared = fit.calculate_fit_error(fit_results=fit_func(x_edges, *popt), data_fit_to=y_edges)

    return y_corr, fit_func, popt, rmse, r_squared


def rotate_cylindrical_coords(df, input_col, output_cols, rotation_col):
    """
    df = process.rotate_cylindrical_coords(df=df, input_col='r', output_cols=('rx', 'ry'), rotation_col='rot')

    :param df:
    :param input_col: 'r'
    :param output_cols: ('rx', 'ry')
    :param rotation_col: 'rot'
    :return:
    """
    x, y = output_cols

    if isinstance(rotation_col, str):
        deg = df[rotation_col].astype(float)
    else:
        df['rot'] = rotation_col
        deg = df['rot']

    df[x] = (df[input_col] - df[input_col].mean()) * np.cos(np.deg2rad(deg))
    df[y] = (df[input_col] - df[input_col].mean()) * np.sin(np.deg2rad(deg))

    return df


def map_lists_a_to_b(a, b):
    """
    returns a new list which is a mapping of a onto b.
    """
    mapped_vals = []
    for val in a:
        # get the distance of val from every element in our list to map to
        dist = np.abs(np.ones_like(b) * val - b)

        # append the value of minimum distance to our mapping list
        mapped_vals.append(b[np.argmin(dist)])

    return mapped_vals


def bin_by_column(df, column_to_bin, number_of_bins, round_to_decimal=2):
    """
    Creates a new column "bin" of which maps column_to_bin to equi-spaced bins. Note, that this does not change the
    original dataframe in any way. It only adds a new column to enable grouping.

    # rename column if mis-named
    if 'true_z' in df.columns:
        df = df.rename(columns={"true_z": "z_true"})

    """

    # turn off pandas warning to try using .loc[row_indexer, col_indexer] = value
    pd.options.mode.chained_assignment = None  # default='warn'

    # round the column_to_bin to integer for easier mapping
    temp_column = 'temp_' + column_to_bin
    df[temp_column] = np.round(df[column_to_bin].values, round_to_decimal)

    # copy the column_to_bin to 'mapped' for mapping
    df.loc[:, 'bin'] = df.loc[:, temp_column]

    # get unique values
    unique_vals = df[temp_column].astype(float).unique()

    # drop temp column
    df = df.drop(columns=[temp_column])

    # calculate the equi-width stepsize
    stepsize = (np.max(unique_vals) - np.min(unique_vals)) / number_of_bins

    # re-interpolate the space
    new_vals = np.linspace(np.min(unique_vals) + stepsize / 2, np.max(unique_vals) - stepsize / 2, number_of_bins)

    # round to reasonable decimal place
    new_vals = np.around(new_vals, decimals=round_to_decimal)

    # create the mapping list
    mappping = map_lists_a_to_b(unique_vals, new_vals)

    # create the mapping dictionary
    mapping_dict = {unique_vals[i]: mappping[i] for i in range(len(unique_vals))}

    # insert the mapped values
    df.loc[:, 'bin'] = df.loc[:, 'bin'].map(mapping_dict)

    return df


# ----------------------------------------------------------------------------------------------------------------------
# TWO-ARRAY FUNCTIONS
"""
Functions that compare, contrast, or synthesize two different plots.
"""


def get_max_sampling_rate(df1, df2, x):
    return np.max([(df1[x].max() - df1[x].min()) / len(df1), (df2[x].max() - df2[x].min()) / len(df2)])


def uniformize_x_wrapper(dfs, xy_cols, num_points, sampling_rate=None, split_sampling_rate=500):
    """
    x_new, y1, y2 = process.uniformize_x_wrapper(dfs=[], xy_cols=[[], []], num_points, sampling_rate, split_sampling_rate=100)
    """
    eval_split = False
    df1 = dfs[0]
    df2 = dfs[1]

    if not isinstance(xy_cols[0], (list, np.ndarray)):
        xy_cols = [xy_cols, xy_cols]

    # get minimum x-array
    x1 = df1[xy_cols[0][0]].to_numpy()
    x2 = df2[xy_cols[1][0]].to_numpy()
    x_min, x_max = np.max([x1[0], x2[0]]), np.min([x1[-1], x2[-1]])

    if np.max(np.diff(x1)) > split_sampling_rate:
        eval_split = True
        idx_split_1 = np.argmax(np.diff(x1))
        # df11 = df1.iloc[:idx_split_1]
        # df12 = df1.iloc[idx_split_1 + 1:]
        inner_x_min1, inner_x_max1 = df1.iloc[idx_split_1 - 1][xy_cols[0][0]], df1.iloc[idx_split_1 + 1][xy_cols[0][0]]
    else:
        inner_x_min1, inner_x_max1 = x1.max(), x1.min()

    if np.max(np.diff(x2)) > split_sampling_rate:
        eval_split = True
        idx_split_2 = np.argmax(np.diff(x2))
        # df21 = df2.iloc[:idx_split_2]
        # df22 = df2.iloc[idx_split_2 + 1:]
        inner_x_min2, inner_x_max2 = df2.iloc[idx_split_2 - 1][xy_cols[1][0]], df2.iloc[idx_split_2 + 1][xy_cols[1][0]]
    else:
        inner_x_min2, inner_x_max2 = x2.max(), x2.min()

    if eval_split is True:
        # determine inner coordinate range
        x_min_inner, x_max_inner = np.min([inner_x_min1, inner_x_min2]), np.max([inner_x_max1, inner_x_max2])

        # uniformize lower range
        df11 = df1[(df1[xy_cols[0][0]] > x_min) & (df1[xy_cols[0][0]] < x_min_inner)]
        df21 = df2[(df2[xy_cols[1][0]] > x_min) & (df2[xy_cols[1][0]] < x_min_inner)]
        x1_new, y11, y21 = uniformize_x_dataframes([df11, df21], xy_cols, num_points, sampling_rate)

        # uniformize upper range
        df12 = df1[(df1[xy_cols[0][0]] > x_max_inner) & (df1[xy_cols[0][0]] < x_max)]
        df22 = df2[(df2[xy_cols[1][0]] > x_max_inner) & (df2[xy_cols[1][0]] < x_max)]

        if np.min([len(df12), len(df22)]) < 5:
            figg, axx = plt.subplots()
            axx.plot(df1[xy_cols[0][0]], df1[xy_cols[0][1]])
            axx.plot(df2[xy_cols[1][0]], df2[xy_cols[1][1]])
            figg.show()

        x2_new, y12, y22 = uniformize_x_dataframes([df12, df22], xy_cols, num_points, sampling_rate)

        x_new, y1, y2 = np.hstack([x1_new, x2_new]), np.hstack([y11, y12]), np.hstack([y21, y22])

    else:
        x_new, y1, y2 = uniformize_x_dataframes(dfs, xy_cols, num_points, sampling_rate)

    return x_new, y1, y2


def uniformize_x_dataframes(dfs, xy_cols, num_points, sampling_rate=None):
    """
    x_new, y1, y2 = process.uniformize_x_dataframes(dfs=[], xy_cols=[[], []], num_points, sampling_rate)
    """
    df1 = dfs[0]
    df2 = dfs[1]

    if not isinstance(xy_cols[0], (list, np.ndarray)):
        xy_cols = [xy_cols, xy_cols]

    # get minimum x-array
    x1 = df1[xy_cols[0][0]].to_numpy()
    x2 = df2[xy_cols[1][0]].to_numpy()

    if np.min([len(x1), len(x2)]) < 5:
        fig, ax = plt.subplots()
        ax.plot(df1.r, df1.z_surf, label='1')
        ax.plot(df2.r, df2.z_surf, label='2')
        ax.legend()
        plt.show()
        j = 1

    x_min, x_max = np.max([x1[0], x2[0]]), np.min([x1[-1], x2[-1]])

    if sampling_rate is not None:
        x_span = x_max - x_min
        num_points = x_span / sampling_rate
    elif num_points is None:
        raise ValueError("Must provide either 'num_points' or 'sampling_rate'.")
    num_points = int(num_points)

    # calculate interpolation for both arrays
    f1 = interp1d(x1, df1[xy_cols[0][1]].to_numpy())
    f2 = interp1d(x2, df2[xy_cols[1][1]].to_numpy())

    # interpolate over the same x-array
    x_new = np.linspace(x_min, x_max, num_points)
    y1 = f1(x_new)
    y2 = f2(x_new)

    return x_new, y1, y2


def correlate_signals(y1, y2):
    corr = correlate(y1, y2, mode='valid', method='direct')
    return corr