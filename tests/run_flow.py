from os.path import join
import numpy as np

from graycart.GraycartWafer import evaluate_wafer_flow
from graycart.utils import plotting


# ----------------------------------------------------------------------------------------------------------------------
# INPUTS

"""
Some important notes:
    
    1. On "Design Labels" or 'design_lbls':
        * if a wafer only has a single design, then features will be identified by 'a1', 'c2', etc...
        * if a wafer has multiple designs, the 'design_lbls' get attached to the feature labels, 'a1_LrO' or 'a1_erf5'
    2. On "Target Labels" or 'target_lbls':
        * the string 'target-profile_' is inserted before the design label, or 'design_lbl'. The target label doesn't
        really serve a purpose at this point. The 'design file' (x, y, r, l) and 'target profile' (r, z) should be
        combined. 
    3. On design spacing and design locations:
        * 'design_spacing' is primarily used to filter multiple peaks in a single scan. 
        * 'design_locs' isn't really used at this point. 
        * 'design_spacing' should be removed as an input and calculated using 'design_locs'. 

"""

"""
Etch Rates:

smOOth.V2: 
    SPR220-7:   (no hard bake)      vertical etch rate = 600 nm/min; lateral etch rate = 0 nm/s (calculated using w17)
                (Laser Monitor)     3.5 wavelengths --> 1 wavelength ~= 190 nm 
    Si:                             vertical etch rate = 0 nm/min; 

SF6+O2.V6:
    SPR220-7:   (no hard bake)      vertical etch rate = 260 nm/min                             (calculated using w16)
    Si:         (no hard bake)      vertical etch rate = 1950 nm/min 
    
    wafer 16:
        1.  post-dev PR thickness (b1, silicon-to-PR-surface)   =   6.5 microns
        2.  post-etch PR thickness (b1, silicon-to-PR-surface)  =   2.25 microns    -->     260 nm/min
            post-etch Si depth (b1, trench-to-PR-interface)     =   32 microns      -->     1.95 um/min

SF6:
    SPR220-7:   (no hard bake)      vertical etch rate = 10 nm/s; lateral etch rate = 0 nm/s (calculated using w17)
    Si:                             vertical etch rate = 0 nm/s; 
"""

# SHARED

# target feature
plot_width_rel_target_radius = 1.2  # plot radius = target_radius * plot_width_rel_target_radius

# data processing
evaluate_signal_processing = False  # True
lambda_peak_rel_height = lambda x: min([0.95 + x / 100, 0.9875])
z_standoff_measure = -0.125
z_standoff_design = -1  # if "-1", calculate the z_standoff from the measured exposure profile

thickness_PR = 8.25
thickness_PR_budget = 1.5
thickness_PR_budget_below = 1  # 0.5  # NOTE: this value affects many of the plots and is highly subjective.

"""
NOTE: Critical Update Necessary

The function 'gcf.calculate_correct_exposure_profile()' calculates the correct photoresist exposure profile
    
    * given, a target depth and the etch rate selectivity (Si:PR).
    
Therefore, I need to:
    (1) add an 'etch_recipe' input variable which pulls etch rates from GraycartMaterial, or
    (2) make it more clear how the user can input what the target profile should be, (which is then solved for), and then
    (3) this same function, or a different function, will:
        i. calculate the linear transformation (gain) between the current PR profile and target Si profile, and
        ii. tell you what etch selectivity (Si:PR) you need to acheive the target Si profile from your PR profile, and
        iii. indicate which of the characterized etch recipes has the closest etch selectivity (Si:PR) to this requirement. 
"""

# WAFER
wids = [14]

for wid in wids:
    if wid in [2]:
        # DESIGN
        design_lbls = ['linear_ramp_dR3um_dI4_Tol1e1__graycart']
        target_lbls = ['linear_ramp_dR3um_dI4_Tol1e1__graycart']  # NOTE: must be preceded with "target-profile_"

        # field exposure matrix
        dose_lbls = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
        focus_lbls = [1]
        fem_dxdy = [0e3, 0e3]

        design_ids = [0]
        design_spacing = 2.5e3
        design_locs = [[0, n * design_spacing] for n in np.arange(-1, len(dose_lbls))]

        # data processing
        # perform_rolling_on = [[4, 'a1', 25], [4, 'c1', 25]]
        perform_rolling_on = False
        features_of_interest = ['a1', 'c1', 'e1', 'h1', 'i1', 'k1']
        target_radius = 1560 / 2  # microns
        target_depth_profile = 150

        # ---

        # target feature
        plot_width_rel_target_radius = 1.075  # plot radius = target_radius * plot_width_rel_target_radius

        # data processing
        evaluate_signal_processing = False  # True
        lambda_peak_rel_height = lambda x: min([0.95 + x / 100, 0.9875])
        r_standoff_measure = -25  # units: microns
        z_standoff_measure = -0.125  # units: microns
        z_standoff_design = 0  # if "-1", calculate the z_standoff from the measured exposure profile

        thickness_PR = 8.125
        thickness_PR_budget = 0.5
        thickness_PR_budget_below = 0.75  # 0.5  # NOTE: this value affects many of the plots and is highly subjective.
        amplitude = 7  # amplitude of profile in photoresist (not including thickness below and above)

    elif wid in [3, 5]:
        # DESIGN
        design_lbls = ['erf_dia2mm_x01.25_32lyrs_7.5umDepth_Tol1e1_dose800mJ']
        target_lbls = ['erf_dia2mm_x01.25_32lyrs_7.5umDepth_Tol1e1_dose800mJ']

        if wid in [3]:
            # field exposure matrix
            dose_lbls = ['a', 'b', 'c', 'd']
            focus_lbls = [1, 2, 3]
            fem_dxdy = [3.5e3, 3.5e3]

            design_ids = [0]
            design_spacing = 3.5e3
            design_locs = [[0, n * design_spacing] for n in np.arange(-1, len(dose_lbls))]

            # data processing
            # perform_rolling_on = [[4, 'a1', 25], [4, 'c1', 25]]
            perform_rolling_on = False
            features_of_interest = ['a1', 'c1', 'd1', 'd2', 'd3']
            target_radius = 1000  # microns
            target_depth_profile = 150

            # target feature
            plot_width_rel_target_radius = 1.075  # plot radius = target_radius * plot_width_rel_target_radius

            # data processing
            evaluate_signal_processing = False  # True
            lambda_peak_rel_height = lambda x: min([0.95 + x / 100, 0.9875])
            r_standoff_measure = -25  # units: microns
            z_standoff_measure = -0.125  # units: microns
            z_standoff_design = 0  # if "-1", calculate the z_standoff from the measured exposure profile

            thickness_PR = 8.125
            thickness_PR_budget = 0.5
            thickness_PR_budget_below = 0.75  # 0.5  # NOTE: this value affects many of the plots and is highly subjective.
            amplitude = 7  # amplitude of profile in photoresist (not including thickness below and above)

        elif wid in [5]:
            # field exposure matrix
            dose_lbls = ['a', 'b', 'c', 'd', 'e', 'f']
            focus_lbls = [1, 2, 3]
            fem_dxdy = [16e3, 9e3]

            design_ids = [0]
            design_spacing = fem_dxdy[1]
            design_locs = [[0, n * design_spacing] for n in np.arange(-1, len(dose_lbls))]

            # data processing
            # perform_rolling_on = [[4, 'a1', 25], [4, 'c1', 25]]
            perform_rolling_on = False
            features_of_interest = ['d1', 'e1', 'd2', 'e2']
            target_radius = 1000  # microns
            target_depth_profile = 150

            # target feature
            plot_width_rel_target_radius = 1.075  # plot radius = target_radius * plot_width_rel_target_radius

            # data processing
            evaluate_signal_processing = False  # True
            lambda_peak_rel_height = lambda x: min([0.95 + x / 100, 0.9875])
            r_standoff_measure = -25  # units: microns
            z_standoff_measure = -0.125  # units: microns
            z_standoff_design = 0  # if "-1", calculate the z_standoff from the measured exposure profile

            thickness_PR = 8.125
            thickness_PR_budget = 0.5
            thickness_PR_budget_below = 0.75  # 0.5  # NOTE: this value affects many of the plots and is highly subjective.
            amplitude = 7  # amplitude of profile in photoresist (not including thickness below and above)

    elif wid in [6]:
        # DESIGN
        design_lbls = ['erf_dia2mm_x01.25_40lyrs_7.5umDepth_Tol1e1_dose800mJ']
        target_lbls = ['erf_dia2mm_x01.25_40lyrs_7.5umDepth_Tol1e1_dose800mJ']

        # field exposure matrix
        dose_lbls = ['a', 'b', 'c', 'd']
        focus_lbls = [1]
        fem_dxdy = [0, 7e3]

        design_ids = [0]
        design_spacing = fem_dxdy[1]
        design_locs = [[0, n * design_spacing] for n in np.arange(-1, len(dose_lbls))]

        # data processing
        # perform_rolling_on = [[4, 'a1', 25], [4, 'c1', 25]]
        perform_rolling_on = False
        features_of_interest = ['a1', 'b1', 'c1', 'd1']
        target_radius = 1000  # microns
        target_depth_profile = 150

        # target feature
        plot_width_rel_target_radius = 1.075  # plot radius = target_radius * plot_width_rel_target_radius

        # data processing
        evaluate_signal_processing = False  # True
        lambda_peak_rel_height = lambda x: min([0.95 + x / 100, 0.9875])
        r_standoff_measure = -25  # units: microns
        z_standoff_measure = -0.125  # units: microns
        z_standoff_design = 0  # if "-1", calculate the z_standoff from the measured exposure profile

        thickness_PR = 8.125
        thickness_PR_budget = 0.5
        thickness_PR_budget_below = 0.75  # 0.5  # NOTE: this value affects many of the plots and is highly subjective.
        amplitude = 7  # amplitude of profile in photoresist (not including thickness below and above)

    elif wid in [7]:
        # DESIGN
        design_lbls = ['erf_dia2.5mm_x01.25_40lyrs_7.5umDepth_Tol1e1_dose800mJ']
        target_lbls = ['erf_dia2.5mm_x01.25_40lyrs_7.5umDepth_Tol1e1_dose800mJ']

        # field exposure matrix
        dose_lbls = ['a', 'b', 'c', 'd', 'e', 'f']
        focus_lbls = [1, 2]
        fem_dxdy = [22e3, 10e3]

        design_ids = [0]
        design_spacing = fem_dxdy[1]
        design_locs = [[0, n * design_spacing] for n in np.arange(-1, len(dose_lbls))]

        # data processing
        # perform_rolling_on = [[4, 'a1', 25], [4, 'c1', 25]]
        perform_rolling_on = False
        features_of_interest = ['d1', 'e1', 'f1']
        target_radius = 1250  # microns
        target_depth_profile = 150

        # target feature
        plot_width_rel_target_radius = 1.075  # plot radius = target_radius * plot_width_rel_target_radius

        # data processing
        evaluate_signal_processing = False  # True
        lambda_peak_rel_height = lambda x: min([0.95 + x / 100, 0.9875])
        r_standoff_measure = -25  # units: microns
        z_standoff_measure = -0.125  # units: microns
        z_standoff_design = 0  # if "-1", calculate the z_standoff from the measured exposure profile

        thickness_PR = 8.125
        thickness_PR_budget = 0.5
        thickness_PR_budget_below = 0.75  # 0.5  # NOTE: this value affects many of the plots and is highly subjective.
        amplitude = 7  # amplitude of profile in photoresist (not including thickness below and above)

    elif wid in [8, 9]:
        # DESIGN
        design_lbls = ['erf_dia1.75mm_x01p25_45lyrs_7.25umDepth_Tol1e-1_dose1000mJ']
        target_lbls = ['erf_dia1.75mm_x01p25_45lyrs_7.25umDepth_Tol1e-1_dose1000mJ']

        if wid in [8]:
            # field exposure matrix
            dose_lbls = ['a', 'b', 'c', 'd', 'e']
            focus_lbls = [1, 2]
            fem_dxdy = [16e3, 8e3]

            design_ids = [0]
            design_spacing = fem_dxdy[1]
            design_locs = [[0, n * design_spacing] for n in np.arange(-1, len(dose_lbls))]

            # data processing
            perform_rolling_on = False
            features_of_interest = ['c1', 'd1', 'e1']
            target_radius = 875  # microns
            target_depth_profile = 150

            # target feature
            plot_width_rel_target_radius = 1.075  # plot radius = target_radius * plot_width_rel_target_radius

            # data processing
            evaluate_signal_processing = False  # True
            lambda_peak_rel_height = lambda x: min([0.95 + x / 100, 0.9875])
            r_standoff_measure = -25  # units: microns
            z_standoff_measure = -0.125  # units: microns
            z_standoff_design = 0  # if "-1", calculate the z_standoff from the measured exposure profile

            thickness_PR = 8.125
            thickness_PR_budget = 0.5
            thickness_PR_budget_below = 0.375  # 0.5  # NOTE: this value affects many of the plots and is highly subjective.
            amplitude = 7.25  # amplitude of profile in photoresist (not including thickness below and above)

        elif wid in [9]:
            # field exposure matrix
            dose_lbls = ['a', 'b', 'c']
            focus_lbls = [1, 2]
            fem_dxdy = [16e3, 10e3]

            design_ids = [0]
            design_spacing = fem_dxdy[1]
            design_locs = [[0, n * design_spacing] for n in np.arange(-1, len(dose_lbls))]

            # data processing
            perform_rolling_on = False
            features_of_interest = ['a1', 'b1', 'c1']
            target_radius = 875  # microns
            target_depth_profile = 150

            # target feature
            plot_width_rel_target_radius = 1.075  # plot radius = target_radius * plot_width_rel_target_radius

            # data processing
            evaluate_signal_processing = False  # True
            lambda_peak_rel_height = lambda x: min([0.95 + x / 100, 0.9875])
            r_standoff_measure = -25  # units: microns
            z_standoff_measure = -0.125  # units: microns
            z_standoff_design = 0  # if "-1", calculate the z_standoff from the measured exposure profile

            thickness_PR = 8.125
            thickness_PR_budget = 0.5
            thickness_PR_budget_below = 0.375  # 0.5  # NOTE: this value affects many of the plots and is highly subjective.
            amplitude = 7.25  # amplitude of profile in photoresist (not including thickness below and above)

    elif wid in [10]:
        # DESIGN
        design_lbls = ['erfLR_dia1.75mm_x00_45lyrs_7.25umDepth_Tol1e-1_dose1000mJ']
        target_lbls = ['erfLR_dia1.75mm_x00_45lyrs_7.25umDepth_Tol1e-1_dose1000mJ']

        # field exposure matrix
        dose_lbls = ['a', 'b', 'c', 'd', 'e', 'f']
        focus_lbls = [1, 2, 3]
        fem_dxdy = [16e3, 9e3]

        design_ids = [0]
        design_spacing = fem_dxdy[1]
        design_locs = [[0, n * design_spacing] for n in np.arange(-1, len(dose_lbls))]

        # data processing
        # perform_rolling_on = [[4, 'a1', 25], [4, 'c1', 25]]
        perform_rolling_on = False
        features_of_interest = ['a1', 'b1', 'c1', 'd1', 'e1', 'f1']
        target_radius = 875  # microns
        target_depth_profile = 150

        # target feature
        plot_width_rel_target_radius = 1.075  # plot radius = target_radius * plot_width_rel_target_radius

        # data processing
        evaluate_signal_processing = False  # True
        lambda_peak_rel_height = lambda x: min([0.95 + x / 100, 0.9875])
        r_standoff_measure = -25  # units: microns
        z_standoff_measure = -0.125  # units: microns
        z_standoff_design = 0  # if "-1", calculate the z_standoff from the measured exposure profile

        thickness_PR = 8.125
        thickness_PR_budget = 0.375
        thickness_PR_budget_below = 0.375  # 0.5  # NOTE: this value affects many of the plots and is highly subjective.
        amplitude = 7.25  # amplitude of profile in photoresist (not including thickness below and above)

        save_wafer_flow = True
        eval_lithography = False
        predict_etch_rates = False
        grade_profile = False

    elif wid in [11]:
        # DESIGN
        design_lbls = ['erf_stacked_LR_dia1.5mm_x00_40lyrs_7.75umDepth_Tol1e-1_dose1050mJ']
        target_lbls = ['erf_stacked_LR_dia1.5mm_x00_40lyrs_7.75umDepth_Tol1e-1_dose1050mJ']

        # field exposure matrix
        dose_lbls = ['a', 'b', 'c', 'd', 'e', 'f']
        focus_lbls = [1, 2, 3]
        fem_dxdy = [16e3, 9e3]

        design_ids = [0]
        design_spacing = fem_dxdy[1]
        design_locs = [[0, n * design_spacing] for n in np.arange(-1, len(dose_lbls))]

        # data processing
        # perform_rolling_on = [[4, 'a1', 25], [4, 'c1', 25]]
        perform_rolling_on = False
        features_of_interest = ['a1', 'b1', 'c1', 'd1', 'e1', 'f1']
        target_radius = 750  # microns
        target_depth_profile = 150

        # target feature
        plot_width_rel_target_radius = 1.075  # plot radius = target_radius * plot_width_rel_target_radius

        # data processing
        evaluate_signal_processing = False  # True
        lambda_peak_rel_height = lambda x: min([0.95 + x / 100, 0.9875])
        r_standoff_measure = -25  # units: microns
        z_standoff_measure = -0.125  # units: microns
        z_standoff_design = 0  # if "-1", calculate the z_standoff from the measured exposure profile

        thickness_PR = 7.5
        thickness_PR_budget = 0.25
        thickness_PR_budget_below = 0.25  # 0.5  # NOTE: this value affects many of the plots and is highly subjective.
        amplitude = 7.25  # amplitude of profile in photoresist (not including thickness below and above)

        save_wafer_flow = True
        eval_lithography = False
        predict_etch_rates = False
        grade_profile = False

    elif wid in [12]:
        # DESIGN
        design_lbls = ['erf_stacked_LR_dia2.49mm_x00_50lyrs_6.58umDepth_Tol1e-1_dose1050mJ']
        target_lbls = ['erf_stacked_LR_dia2.49mm_x00_50lyrs_6.58umDepth_Tol1e-1_dose1050mJ']

        # field exposure matrix
        dose_lbls = ['a', 'b', 'c', 'd', 'e', 'f']
        focus_lbls = [1, 2, 3]
        fem_dxdy = [16e3, 9e3]

        design_ids = [0]
        design_spacing = fem_dxdy[1]
        design_locs = [[0, n * design_spacing] for n in np.arange(-1, len(dose_lbls))]

        # data processing
        # perform_rolling_on = [[4, 'a1', 25], [4, 'c1', 25]]
        perform_rolling_on = False
        features_of_interest = ['a1', 'b1', 'c1', 'd1', 'e1', 'f1']  # foi
        target_radius = 1250  # 1500  # microns
        target_depth_profile = 80

        # target feature
        plot_width_rel_target_radius = 1.075  # plot radius = target_radius * plot_width_rel_target_radius

        # data processing
        evaluate_signal_processing = False  # True
        lambda_peak_rel_height = lambda x: min([0.95 + x / 100, 0.9875])
        r_standoff_measure = -25  # units: microns
        z_standoff_measure = -0.125  # units: microns
        z_standoff_design = 0  # if "-1", calculate the z_standoff from the measured exposure profile

        thickness_PR = 7.5
        thickness_PR_budget = 0.25
        thickness_PR_budget_below = 0.25  # 0.5  # NOTE: this value affects many of the plots and is highly subjective.
        amplitude = 6.575  # amplitude of profile in photoresist (not including thickness below and above)

        save_wafer_flow = True
        eval_lithography = False
        predict_etch_rates = False
        grade_profile = False

    elif wid in [13, 14]:
        # DESIGN
        design_lbls = ['erf_LR_dia2.75mm_x01.25_32lyrs_6.0umDepth_Tol1e-1_dose1040mJ']
        target_lbls = ['erf_LR_dia2.75mm_x01.25_32lyrs_6.0umDepth_Tol1e-1_dose1040mJ']

        # field exposure matrix
        dose_lbls = ['a', 'b', 'c', 'd', 'e', 'f']
        focus_lbls = [1, 2, 3]
        fem_dxdy = [16e3, 9e3]

        design_ids = [0]
        design_spacing = fem_dxdy[1]
        design_locs = [[0, n * design_spacing] for n in np.arange(-1, len(dose_lbls))]

        # data processing
        # perform_rolling_on = [[4, 'a1', 25], [4, 'c1', 25]]
        perform_rolling_on = False
        features_of_interest = ['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'd2', 'c2']  # foi
        target_radius = 1650  # microns (NOTE: this is actually a very important parameter, because data gets clipped)
        target_depth_profile = 250

        # target feature
        plot_width_rel_target_radius = 1.075  # plot radius = target_radius * plot_width_rel_target_radius

        # data processing
        evaluate_signal_processing = False  # True
        lambda_peak_rel_height = lambda x: min([0.95 + x / 100, 0.9875])
        r_standoff_measure = -25  # units: microns
        z_standoff_measure = -0.125  # units: microns
        z_standoff_design = 0  # if "-1", calculate the z_standoff from the measured exposure profile

        thickness_PR = 7.2
        thickness_PR_budget = 0.25  # minimum allowable photoresist thickness before stopping etch
        thickness_PR_budget_below = 0.25  # 0.5  # NOTE: this value affects many of the plots and is highly subjective.
        amplitude = 6.0  # amplitude of profile in photoresist (not including thickness below and above)

        save_wafer_flow = True
        eval_lithography = False
        predict_etch_rates = False
        grade_profile = False

    else:
        raise ValueError()

    # SHARED

    base_path = '/Users/mackenzie/Desktop/zipper_paper/Fabrication/grayscale/w{}'.format(wid)
    fn_pflow = 'process-flow_w{}.xlsx'.format(wid)
    path_results = 'results'
    profilometry_tool = 'KLATencor-P7'

    # results
    save_type = '.png'
    step_develop = 3
    save_wafer_flow = True

    plot_feature_evolution = False
    eval_lithography = False
    predict_etch_rates = False

    save_deformation_characteristics = True
    membrane_thickness = 20

    # ------------------------------------------------------------------------------------------------------------------

    # 1. instantiate grayscale class objects and read profilometry data
    wfr = evaluate_wafer_flow(wid, base_path, fn_pflow, path_results, profilometry_tool,
                              design_lbls, target_lbls, design_locs, design_ids,
                              design_spacing, dose_lbls, focus_lbls, fem_dxdy,
                              target_radius=target_radius,
                              plot_width_rel_target_radius=plot_width_rel_target_radius,
                              peak_rel_height=lambda_peak_rel_height,
                              save_all_results=save_wafer_flow,
                              perform_rolling_on=perform_rolling_on,
                              evaluate_signal_processing=evaluate_signal_processing,
                              zr_standoff=[z_standoff_measure, r_standoff_measure],
                              )

    # plot feature evolution
    if plot_feature_evolution:
        wfr.plot_feature_evolution(px='r', py='z', save_fig=plot_feature_evolution)

    # NOTE: THIS IS WRONG! The mapping function incorrectly maps depth onto dose.
    if eval_lithography:
        # 2. relate mask design (intensity profile) to actual outcome (surface profile)
        wfr.characterize_exposure_dose_depth_relationship(plot_figs=eval_lithography,
                                                          save_type=save_type,
                                                          )

        # 3. merge features
        wfr.merge_exposure_doses_to_process_depths(export=eval_lithography)
        wfr.plot_all_exposure_dose_to_depth(step=step_develop)
        wfr.compare_exposure_functions()

        # 4. use intensity+surface profiles to generate LUT, redraw intensity profile using new LUT
        wfr.correct_grayscale_design_profile(amplitude=amplitude,  # None --> auto-calculate based on etch selectivity
                                             target_depth=None,  # None --> uses 'amplitude': depth of PR feature
                                             thickness_PR=thickness_PR,
                                             thickness_PR_budget_below=thickness_PR_budget_below,
                                             plot_figs=eval_lithography,
                                             save_type=save_type,
                                             )

        # plot exposure profile
        for foi in features_of_interest:
            if foi in wfr.features.keys():
                gpf = wfr.features[foi]
                plotting.plot_exposure_profile(gcf=gpf, path_save=join(wfr.path_results, 'figs'), save_type=save_type)

        # plot feature profile overlaid with exposure profile
        step = max(wfr.list_steps)
        for did in wfr.dids:
            plotting.plot_overlay_feature_and_exposure_profiles(gcw=wfr,
                                                                step=step,
                                                                did=did,
                                                                path_save=join(wfr.path_results, 'figs'),
                                                                save_type=save_type,
                                                                )

    # predict etch time to achieve specified profile depth
    if predict_etch_rates:
        wfr.backout_process_to_achieve_target(target_radius=target_radius,
                                              target_depth=target_depth_profile,
                                              thickness_PR=thickness_PR,
                                              thickness_PR_budget=thickness_PR_budget,
                                              save_fig=predict_etch_rates)

        # compare target to features
        """ below function generates: est-process-and-profile...png """
        etch_recipe_PR = 'smOOth.V1'
        etch_recipe_Si = 'sweep'  # 'sweep', None, or a specific recipe ['SF6+O2.V1', 'SF6+O2.V2', 'SF6+O2.S20', 'SF6+O2.S25']#, 'SF6+O2.S30', 'SF6+O2.S40', 'SF6+O2.S50']
        wfr.compare_target_to_feature_evolution(px='r', py='z',
                                                etch_recipe_PR=etch_recipe_PR,
                                                etch_recipe_Si=etch_recipe_Si,
                                                target_depth=target_depth_profile,
                                                thickness_PR_budget=thickness_PR_budget,
                                                save_fig=predict_etch_rates,
                                                )

    # ---

    # ---

    # Functions that require 2D surface profiles from orthogonal directions (i.e., along both x and y)

    # grade target accuracy
    grade_profile = False
    if grade_profile:
        # final_profile_depth = wfr.processes[max(wfr.list_steps)].features['a1'].peak_properties['peak_heights']
        # TODO: grade profile accuracy against: (1) target depth, and (ii) actual depth (i.e., depth-normalized profile).
        wfr.grade_profile_accuracy(step=max(wfr.list_steps), target_radius=target_radius, target_depth=target_depth_profile)

    #wfr.plot_profile_3d(step=max(wfr.list_steps))

    # ---

    # ---

    # Specialty functions

    # export deformation characteristics given membrane thickness
    if save_deformation_characteristics:
        wfr.export_membrane_deformation_characteristics(membrane_thickness=membrane_thickness)

    print("test_flow.py completed without errors.")