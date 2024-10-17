from os.path import join, isdir
from os import makedirs
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd

from .GraycartProcess import GraycartProcess
from .GraycartFeature import ProcessFeature, SiliconFeature, initialize_designs, initialize_design_features
from graycart import utils
from .utils import plotting, io


class GraycartWafer(object):

    def __init__(self, wid, path, designs, features, process_flow, processes, measurement_methods,
                 path_results='results'):
        """

        :param folder:
        """

        super(GraycartWafer, self).__init__()

        # mandatory
        self._wid = wid
        self.path = path

        # file paths
        self.path_results = join(self.path, path_results)
        self.make_dirs()

        # add designs
        self.designs = designs

        # add features
        self.features = features

        # add measurement methods
        self.measurement_methods = measurement_methods

        # add processes
        self.process_flow = process_flow
        self.processes = processes
        self.add_process()

        # initialize variables
        self.dfps = None

    # ------------------------------------------------------------------------------------------------------------------
    # DATA INPUT FUNCTIONS

    def add_process(self):

        wfr_materials = {}

        # iterate through process flow
        processes = OrderedDict()
        for step, prcs in self.process_flow.items():

            # add measurement data to each process
            data_paths = {}
            for meas_meth, data in self.measurement_methods.items():

                # only add if data is present
                if isdir(join(self.path, data['header'], str(prcs['path']))):
                    meas_meth_data_path = join(self.path, data['header'], str(prcs['path']))
                else:
                    meas_meth_data_path = None

                if not 'x_units_read' in data.keys():
                    data.update({'x_units_read': 1, 'x_units_write': 1,
                                 'y_units_read': 1, 'y_units_write': 1})

                data_paths.update({meas_meth: {'tool': data['header'],
                                               'path': meas_meth_data_path,
                                               'filetype_read': data['filetype_read'],
                                               'x_units_read': data['x_units_read'],
                                               'y_units_read': data['y_units_read'],
                                               'filetype_write': data['filetype_write'],
                                               'x_units_write': data['x_units_write'],
                                               'y_units_write': data['y_units_write'],
                                               },
                                   })

            # instantiate process
            process = GraycartProcess(step=prcs['step'],
                                      process_type=prcs['process_type'],
                                      recipe=prcs['recipe'],
                                      time=prcs['time'],
                                      details=prcs['details'],
                                      basepath=self.path,
                                      subpath=prcs['path'],
                                      features=self.features.copy(),
                                      data=data_paths,
                                      materials=wfr_materials,
                                      )

            wfr_materials = deepcopy(process.materials)
            processes.update({process.step: process})

        self.processes = processes

    def get_feature(self, fid, step):
        gcp = self.processes[step]
        gcf_ = None
        for flbl, gcf in gcp.features.items():
            if gcf.fid == fid:
                gcf_ = gcf
        return gcf_

    # ------------------------------------------------------------------------------------------------------------------
    # DATA PROCESSING FUNCTIONS

    def backout_process_to_achieve_target(self,
                                          target_radius, target_depth,
                                          thickness_PR, thickness_PR_budget, r_target=20,
                                          save_fig=False, path_save=None, save_type='.png'):

        if path_save is None and save_fig is True:
            path_save = self.path_results

        dids = self.dids

        for did in dids:
            dft = self.designs[did].mdft.copy()

            # resize
            dft['r'] = dft['r'] * target_radius / 2
            dft['z'] = dft['z'] * target_depth

            est_process_flow = utils.process.backout_process_from_target(df=dft,
                                                                         px='r',
                                                                         py='z',
                                                                         pys='z_surf',
                                                                         thickness_PR=thickness_PR,
                                                                         thickness_PR_budget=thickness_PR_budget,
                                                                         r_target=r_target,
                                                                         did=did,
                                                                         )

            plotting.plot_target_profile_and_process_flow_backout(dft,
                                                                  est_process_flow,
                                                                  path_save=path_save,
                                                                  save_type=save_type,
                                                                  )

    def evaluate_process_profilometry(self,
                                      plot_fits=True,
                                      perform_rolling_on=False,
                                      evaluate_signal_processing=False,
                                      downsample=5,
                                      width_rel_radius=0.01,
                                      peak_rel_height=0.975,
                                      fit_func='parabola',
                                      prominence=1,
                                      plot_width_rel_target=1.1,
                                      zr_standoff=None,
                                      ):
        """
        For each process:
            *  add available profilometry data to each feature
                * plots tilt correction
            * plot peak_finding algorithm
            * plot all profiles on the same figure for comparison
        """

        for step, gcprocess in self.processes.items():
            if gcprocess.ppath is not None:
                if isinstance(peak_rel_height, float):
                    pass
                elif callable(peak_rel_height):
                    peak_rel_height = peak_rel_height(step)  # min([0.93 + step / 100, 0.97])
                else:
                    raise ValueError()

                if 'Photoresist' in gcprocess.materials.keys():
                    pr_thickness = gcprocess.materials['Photoresist'].thickness
                else:
                    pr_thickness = 0

                if gcprocess.process_type == 'Strip':
                    gcprocess.add_3d_profilometry_to_features(plot_fits=plot_fits,
                                                              perform_rolling_on=perform_rolling_on,
                                                              evaluate_signal_processing=evaluate_signal_processing,
                                                              downsample=downsample,
                                                              width_rel_radius=width_rel_radius,
                                                              peak_rel_height=peak_rel_height,
                                                              fit_func=fit_func,
                                                              prominence=prominence,
                                                              plot_width_rel_target=plot_width_rel_target,
                                                              thickness_pr=pr_thickness,
                                                              )

                else:
                    gcprocess.add_profilometry_to_features(plot_fits=plot_fits,
                                                           perform_rolling_on=perform_rolling_on,
                                                           evaluate_signal_processing=evaluate_signal_processing,
                                                           downsample=downsample,
                                                           width_rel_radius=width_rel_radius,
                                                           peak_rel_height=peak_rel_height,
                                                           fit_func=fit_func,
                                                           prominence=prominence,
                                                           plot_width_rel_target=plot_width_rel_target,
                                                           thickness_pr=pr_thickness,
                                                           zr_standoff=zr_standoff,
                                                           )

                if plot_fits:
                    gcprocess.plot_profilometry_feature_fits(save_fig=plot_fits)
                    gcprocess.plot_profilometry_features(save_fig=plot_fits)

    def merge_processes_profilometry(self, export=False):

        dfs = []
        for step, gcp in self.processes.items():
            for f_lbl, gcf in gcp.features.items():
                if isinstance(gcf, (ProcessFeature, SiliconFeature)):
                    df = gcf.dfpk
                    df['did'] = gcf.did
                    df['step'] = step
                    df['dose'] = gcf.dose
                    df['focus'] = gcf.focus
                    dfs.append(df)

        dfs = pd.concat(dfs)
        self.dfps = dfs

        if export:
            dfs.to_excel(join(self.path, 'results',
                              'w{}_merged_process_profiles'.format(self._wid) +
                              self.measurement_methods['Profilometry']['filetype_write']),
                         index=False,
                         )

    def merge_exposure_doses_to_process_depths(self, export=False):
        dfs = []
        for step, gcp in self.processes.items():
            for f_lbl, gcf in gcp.features.items():
                if isinstance(gcf, ProcessFeature):
                    if gcf.exposure_func is not None:
                        df = gcf.exposure_func['dfmap']
                        df['did'] = gcf.did
                        df['fid'] = gcf.fid
                        df['step'] = step
                        df['dose'] = gcf.dose
                        df['focus'] = gcf.focus
                        dfs.append(df)

        dfs = pd.concat(dfs)
        self.df_all = dfs

        if export:
            dfs.to_excel(join(self.path, 'results',
                              'w{}_merged_dose-depths'.format(self._wid) +
                              self.measurement_methods['Profilometry']['filetype_write']),
                         index=False,
                         )

    def characterize_exposure_dose_depth_relationship(self, process_type=None, steps=None,
                                                      plot_figs=False, save_type='.png'):

        if process_type is None:
            process_type = ['Develop', 'Thermal Reflow']

        if not isinstance(process_type, list):
            process_type = [process_type]

        if steps is None:
            steps = self.list_steps

        for step, gcp in self.processes.items():
            if gcp.process_type in process_type and step in steps:
                for flbl, gcf in gcp.features.items():
                    if isinstance(gcf, ProcessFeature):

                        gcf.calculate_exposure_dose_depth_relationship()

                        if plot_figs:
                            plotting.plot_exposure_dose_depth_relationship(gcf,
                                                                           path_save=join(self.path_results, 'figs'),
                                                                           save_type=save_type,
                                                                           )

                            plotting.plot_exposure_functions(gcf,
                                                             path_save=join(self.path_results, 'figs'),
                                                             save_type=save_type,
                                                             )

    def correct_grayscale_design_profile(self, amplitude,
                                         target_depth=None, thickness_PR=None, thickness_PR_budget_below=None,
                                         process_type=None, steps=None, plot_figs=False,
                                         save_type='.png'):
        """
        If amplitude (depth of PR feature) is None, then amplitude will be calculated from 'target_depth' and etch
        rate selectivity of Si:PR (right now, hard-coded as SF6+O2.V5)
        :param amplitude: the height of the profile you want to encode into photoresist (not including top/bot standoff). Note, if amplitude > max_exposure_depth, then amplitude is set equal to max_exposure_depth.
        :param target_depth: the depth of the silicon profile, after etching and strip
        :param thickness_PR: the thickness of the photoresist (i.e., pre-exposure)
        :param thickness_PR_budget_below: the thickness of photoresist between substrate and bottom of exposed profile.
        :param process_type:
        :param steps:
        :param plot_figs:
        :param save_type:
        :return:
        """

        if process_type is None:
            process_type = ['Develop', 'Thermal Reflow']

        if not isinstance(process_type, list):
            process_type = [process_type]

        if steps is None:
            steps = self.list_steps

        for step, gcp in self.processes.items():
            if gcp.process_type in process_type and step in steps:
                for flbl, gcf in gcp.features.items():
                    if isinstance(gcf, ProcessFeature):
                        gcf.calculate_correct_exposure_profile(amplitude=amplitude,
                                                               target_depth=target_depth,
                                                               thickness_PR=thickness_PR,
                                                               thickness_PR_budget_below=thickness_PR_budget_below,
                                                               )

                        if plot_figs:
                            plotting.plot_exposure_profile_and_design_layers(gcf,
                                                                             path_save=join(self.path_results, 'figs'),
                                                                             save_type=save_type,
                                                                             thickness_PR=thickness_PR,
                                                                             )

    def grade_profile_accuracy(self, step, target_radius, target_depth):

        res = []
        for gcf in self.processes[step].features.values():
            if isinstance(gcf, SiliconFeature):
                gcf.calculate_profile_to_target_error(target_radius, target_depth)
                gcf.calculate_volume_to_target_error()
                res.append([gcf.fid, gcf.target_rmse, gcf.target_rmse_percent_depth, gcf.target_r_squared,
                            gcf.volume, gcf.target_volume_error])

                plotting.plot_profile_to_target_error(gcf, path_save=join(self.path_results, 'figs'), save_type='.png')

                # gcf.correlate_profile_to_target()

        res = pd.DataFrame(np.array(res), columns=['fid', 'rmse', 'rmse_percent_depth', 'r_sq', 'vol', 'err_vol'])
        res.to_excel(join(self.path_results, 'profile_accuracy_rmse.xlsx'))
        print(res)

    def plot_profile_3d(self, step):

        for gcf in self.processes[step].features.values():
            if isinstance(gcf, SiliconFeature):
                plotting.plot_3d_profile(df=gcf.dfpk3d, px='rx', py='ry', pz='z',
                                         aspect_ratio_z=1, view_height=20, view_rotation=30,
                                         path_save=join(self.path_results, 'figs', '3d-surf_fid{}.png'.format(gcf.fid)))

    # ------------------------------------------------------------------------------------------------------------------
    # PLOTTING FUNCTIONS (HIGH-LEVEL)

    def plot_all_exposure_dose_to_depth(self, step, save_type='.png'):
        plotting.plot_all_exposure_dose_to_depth(df=self.df_all[self.df_all['step'] == step],
                                                 path_save=join(self.path_results, 'figs',
                                                                'merged_dose-depths_step{}'.format(step) + save_type))

    def plot_feature_evolution(self, px='r', py='z', save_fig=True):
        dids = self.dids
        if len(dids) > 1:
            dids.append(None)

        for did in dids:
            self.plot_features_diff_by_process_and_material(px=px, py='z_surf', did=did, normalize=False,
                                                            save_fig=save_fig,
                                                            save_type='.png')

            self.plot_features_diff_by_process(px=px, py=py, did=did, normalize=False, save_fig=save_fig,
                                               save_type='.png')

            for norm in [False, True]:
                self.plot_features_by_process(px=px, py=py, did=did, normalize=norm, save_fig=save_fig,
                                              save_type='.png')

                self.plot_processes_by_feature(px=px, py=py, did=did, normalize=norm, save_fig=save_fig,
                                               save_type='.png')

    def compare_target_to_feature_evolution(self, px, py, etch_recipe_PR, etch_recipe_Si, save_fig=True):
        dids = self.dids
        if len(dids) > 1:
            dids.append(None)

        if etch_recipe_Si == 'sweep':
            etch_recipe_Sis = ['SF6+O2.V5', 'SF6+O2.V6', 'SF6+O2.V10', 'SF6+O2.V11', 'SF6+O2.V12', 'SF6+O2.V13']
            for etch_recipe_Si in etch_recipe_Sis:
                self.estimated_target_profiles(px, py='z_surf',
                                               etch_recipe_PR=etch_recipe_PR, etch_recipe_Si=etch_recipe_Si,
                                               include_target=True, save_fig=save_fig, save_type='.png')
        elif etch_recipe_Si is not None:
            self.estimated_target_profiles(px, py='z_surf',
                                           etch_recipe_PR=etch_recipe_PR, etch_recipe_Si=etch_recipe_Si,
                                           include_target=True, save_fig=save_fig, save_type='.png')
        else:
            pass

        for did in dids:

            self.compare_target_to_features_by_process(px=px, py='z_surf', did=did, normalize=False, save_fig=save_fig,
                                                       save_type='.png')
            for norm in [False, True]:
                self.compare_target_to_features_by_process(px=px, py=py, did=did, normalize=norm, save_fig=save_fig,
                                                           save_type='.png')  # py = 'z'

    def compare_exposure_functions(self, process_types=None):
        if process_types is None:
            process_types = ['Develop', 'Thermal Reflow']

        for step, gcp in self.processes.items():
            if gcp.process_type in process_types:
                plotting.compare_exposure_function_plots(gcp, path_save=self.path_results, save_type='.png')

    # ------------------------------------------------------------------------------------------------------------------
    # PLOTTING FUNCTIONS (LOW-LEVEL)

    def plot_features_by_process(self, px, py, did=None, normalize=False, save_fig=False, save_type='.png'):
        plotting.plot_features_by_process(self, px, py, did=did, normalize=normalize, save_fig=save_fig,
                                          save_type=save_type)

    def plot_features_diff_by_process(self, px, py, did=None, normalize=False, save_fig=False, save_type='.png'):
        plotting.plot_features_diff_by_process(self, px, py, did=did, normalize=normalize, save_fig=save_fig,
                                               save_type=save_type)

    def plot_features_diff_by_process_and_material(self, px, py='z_surf', did=None, normalize=False, save_fig=False,
                                                   save_type='.png'):
        plotting.plot_features_diff_by_process_and_material(self, px, py, did=did, normalize=normalize,
                                                            save_fig=save_fig,
                                                            save_type=save_type)

    def plot_processes_by_feature(self, px, py, did=None, normalize=False, save_fig=False, save_type='.png'):
        plotting.plot_processes_by_feature(self, px, py, did=did, normalize=normalize, save_fig=save_fig,
                                           save_type=save_type)

    def compare_target_to_features_by_process(self, px, py, did=None, normalize=False, save_fig=False,
                                              save_type='.png'):
        plotting.compare_target_to_features_by_process(self, px, py, did=did, normalize=normalize, save_fig=save_fig,
                                                       save_type=save_type)

    def estimated_target_profiles(self, px, py, etch_recipe_PR, etch_recipe_Si, include_target=True, save_fig=False,
                                  save_type='.png'):
        plotting.estimated_target_profiles(self, px, py, etch_recipe_PR, etch_recipe_Si, include_target, save_fig,
                                           save_type)

    # ------------------------------------------------------------------------------------------------------------------
    # UTILITY FUNCTIONS

    def make_dirs(self):

        if not isdir(self.path_results):
            makedirs(self.path_results)

        if not isdir(join(self.path_results, 'figs')):
            makedirs(join(self.path_results, 'figs'))

    # ------------------------------------------------------------------------------------------------------------------
    # PROPERTIES

    @property
    def dids(self):
        dids = [gcf.did for gcf in self.designs.values()]
        dids = list(set(dids))
        return dids

    @property
    def fids(self):
        fids = [gcf.fid for gcf in self.designs.values()]
        fids = list(set(fids))
        return fids

    @property
    def list_steps(self):
        if self.processes is not None:
            list_steps = [stp for stp in self.processes.keys()]
            list_steps = list(set(list_steps))
        else:
            list_steps = None

        return list_steps

    @property
    def list_processes(self):
        if self.processes is not None:
            list_processes = []
            for gcp in self.processes.items():
                list_processes.append(gcp['process_type'])
            list_processes = list(set(list_processes))
        else:
            list_processes = None

        return list_processes


# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# WRAPPER FUNCTION


def evaluate_wafer_flow(wid, base_path, fn_pflow, path_results, profilometry_tool,
                        design_lbls, target_lbls, design_locs, design_ids,
                        design_spacing, dose_lbls, focus_lbls, fem_dxdy,
                        target_radius=None,
                        plot_width_rel_target_radius=1.15,
                        peak_rel_height=0.975,
                        save_all_results=False,
                        perform_rolling_on=False,
                        evaluate_signal_processing=False,
                        zr_standoff=None,
                        ):
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

    # ------------------------------------------------------------------------------------------------------------------
    # ANALYZE THE PROCESS DATA

    wfr.evaluate_process_profilometry(plot_fits=save_all_results,
                                      perform_rolling_on=perform_rolling_on,
                                      evaluate_signal_processing=evaluate_signal_processing,
                                      plot_width_rel_target=plot_width_rel_target_radius,
                                      peak_rel_height=peak_rel_height,
                                      downsample=5,
                                      width_rel_radius=0.01,
                                      fit_func='parabola',
                                      prominence=1,
                                      zr_standoff=zr_standoff,
                                      )
    wfr.merge_processes_profilometry(export=save_all_results)

    return wfr