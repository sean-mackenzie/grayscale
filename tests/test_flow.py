import numpy as np
from graycart.GraycartWafer import *

# ----------------------------------------------------------------------------------------------------------------------
# INPUTS

# target feature
target_radius = 1920  # microns
plot_width_rel_target_radius = 1.2  # plot radius = target_radius * plot_width_rel_target_radius
target_depth_profile = 50

# data processing
evaluate_signal_processing = False  # True
lambda_peak_rel_height = lambda x: min([0.94 + x / 100, 0.9875])
z_standoff_measure = -0.125
z_standoff_design = 0.75
save_all_results = False

# WAFER
wid = 13

# DESIGN
design_lbls = ['erf5_LrO']
target_lbls = ['erf5_LrO']
design_ids = [0]
design_spacing = 5e3
design_locs = [[0, n * design_spacing] for n in np.arange(-2, 3)]

# field exposure matrix
dose_lbls = ['a', 'b', 'c']  # , 'c', 'd', 'e'
focus_lbls = [1]
fem_dxdy = [0e3, 0e3]

# data processing
perform_rolling_on = False  # [[3, 'b1', 25]]  # False
features_of_interest = ['a1', 'b1', 'c1']  # ['b1', 'b2', 'c1', 'c2', 'd1', 'd2']


# SHARED

base_path = '/Users/mackenzie/Desktop/Zipper/Fabrication/Wafer{}'.format(wid)
fn_pflow = 'process-flow_w{}.xlsx'.format(wid)
path_results = 'results'
profilometry_tool = 'KLATencor-P7'

# results
save_type = '.png'
step_develop = 3

# ----------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------
# SET UP THE DATA HIERARCHY

# 3. 'features' undergo 'processes'
process_flow = io.read_process_flow(fp=join(base_path, fn_pflow))

# 4. 'measurements' record the effect of 'processes' on 'features'
measurement_methods = io.read_measurement_methods(profilometry=profilometry_tool)

# 1. initialize 'designs'
designs = initialize_designs(base_path, design_lbls, target_lbls, design_locs, design_ids)

# 2. 'designs' on a wafer form 'features'
features = initialize_design_features(designs,
                                      design_spacing,
                                      dose_lbls,
                                      focus_lbls,
                                      process_flow,
                                      fem_dxdy,
                                      target_radius=target_radius,
                                      )

# 5. the 'wafer' structures all of this data as a historical record of 'cause' and 'effect'
wfr = GraycartWafer(wid=wid,
                    path=base_path,
                    path_results=path_results,
                    designs=designs,
                    features=features,
                    process_flow=process_flow,
                    processes=None,
                    measurement_methods=measurement_methods,
                    )

wfr.evaluate_process_profilometry(plot_fits=save_all_results,
                                  perform_rolling_on=perform_rolling_on,
                                  evaluate_signal_processing=evaluate_signal_processing,
                                  plot_width_rel_target=plot_width_rel_target_radius,
                                  peak_rel_height=lambda_peak_rel_height,
                                  downsample=5,
                                  width_rel_radius=0.01,
                                  fit_func='parabola',
                                  prominence=1,
                                  )