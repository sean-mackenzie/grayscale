from os.path import join, isdir
from os import listdir

import numpy as np
import pandas as pd

from .GraycartFeature import GraycartFeature, ProcessFeature, SiliconFeature
from .GraycartMaterial import SiliconMaterial, PhotoresistMaterial
from .utils import io, process, plotting

"""
self note:

* Does GraycartWafer require every GraycartProcess to provide some interface? I think yes...


"""


class GraycartProcess(object):

    def __init__(self, step, process_type, recipe, time, details, basepath, subpath, features, data, materials={}):
        """
        Base class for all process types.

        Example:
            step:           1
            process_type:   'Develop'
            recipe:         'CD-26A_SNGL_60s'
            time:           60
            details:        {'Developer': 'CD-26A', 'Method': 'Single Puddle'}
            path:           'step1_DEV'

        :param folder:
        """

        super(GraycartProcess, self).__init__()

        self.step = step
        self.process_type = process_type
        self.recipe = recipe
        self.time = time
        self.details = details
        self.basepath = basepath
        self.subpath = subpath

        self.features = features

        self._ptool = data['Profilometry']['tool']
        self._ppath = data['Profilometry']['path']
        self._pread = data['Profilometry']['filetype_read']
        self._pread_x_units = data['Profilometry']['x_units_read']
        self._pread_y_units = data['Profilometry']['y_units_read']
        self._pwrite = data['Profilometry']['filetype_write']
        self._pwrite_x_units = data['Profilometry']['x_units_write']
        self._pwrite_y_units = data['Profilometry']['y_units_write']

        self._emtool = data['Etch Monitor']['tool']
        self._empath = data['Etch Monitor']['path']
        self._emread = data['Etch Monitor']['filetype_read']
        self._emwrite = data['Etch Monitor']['filetype_write']

        self._otool = data['Optical']['tool']
        self._opath = data['Optical']['path']
        self._oread = data['Optical']['filetype_read']
        self._owrite = data['Optical']['filetype_write']

        self._mtool = data['Misc']['tool']
        self._mpath = data['Misc']['path']
        self._mread = data['Misc']['filetype_read']
        self._mwrite = data['Misc']['filetype_write']

        self.materials = materials
        self.initialize_materials()
        self.process_materials()

    def __repr__(self):
        class_ = 'GraycartProcess'
        repr_dict = {'Step': self.step,
                     'Process Type': self.process_type,
                     'Recipe': self.recipe,
                     'Time': self.time,
                     'Sub-Path': self.subpath,
                     }
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    # ------------------------------------------------------------------------------------------------------------------
    # DATA INPUT FUNCTIONS

    def initialize_materials(self):
        if len(self.materials) < 1:
            substrate = 'Silicon'
            self.materials = {substrate: SiliconMaterial(name=substrate, properties=None)}

        if self.process_type == 'Coat':
            self.materials.update({'Photoresist': PhotoresistMaterial(name='Photoresist', properties=None)})
        elif self.process_type == 'Strip':
            del self.materials['Photoresist']

    def process_materials(self):
        processed_materials = {}
        for mat, gcm in self.materials.items():
            gcm.apply_process(process=self.process_type, recipe=self.recipe, time=self.time)
            processed_materials.update({mat: gcm})
        self.materials = processed_materials

    def add_profilometry_to_features(self,
                                     plot_fits=False,
                                     perform_rolling_on=False,
                                     evaluate_signal_processing=False,
                                     downsample=5,
                                     width_rel_radius=0.01,
                                     peak_rel_height=0.95,
                                     fit_func='parabola',
                                     prominence=1.0,
                                     plot_width_rel_target=1.1,
                                     thickness_pr=0.0,
                                     zr_standoff=None,
                                     ):
        """
        Routine:
            1. Read profilometry data files and determine if: (1) 1 scan = 1 feature, or (2) 1 scan = all features
            2. Create a dictionary of {feature id (fid): dataframe of feature's scan profile}
                a. if 1 scan = 1 profile:
                    * iterate through files (scans), process data, store in dictionary {fid: df}
                b. if 1 scan = all features:
                    * process data and store in dictionary {fid: df}
        :return:
        """

        # get files
        process_files = self.collect_profilometry_files()
        """ drop_len = len(self._pread) """

        # iterate through process file list
        process_features = {}

        # each item contains: f: path to scan data; fids, flbls: the fids and flbls to assign peaks within scan data
        for fids, flbls, f in process_files:

            dfpk, peak_details, input_sampling_rate, output_sampling_rate, raw_profile = self.process_profilometry(
                fids, flbls, f,
                plot_fits=plot_fits,
                perform_rolling_on=perform_rolling_on,
                evaluate_signal_processing=evaluate_signal_processing,
                downsample=downsample,
                width_rel_radius=width_rel_radius,
                peak_rel_height=peak_rel_height,
                fit_func=fit_func,
                prominence=prominence,
                plot_width_rel_target=plot_width_rel_target,
            )

            # instantiate ProcessFeature to inherit GraycartFeature
            for pk_lbl, pk_details in peak_details.items():
                pk_details['peak_properties'].update({'input_sampling_rate': input_sampling_rate,
                                                      'sampling_rate': output_sampling_rate,
                                                      'raw_profile': raw_profile})

                # declare 'df' for easier processing
                df = pk_details['df']

                # add z-offset for photoresist thickness
                df['z_surf'] = df['z'] + np.max([thickness_pr, 0])

                print("Step {}, {} for {} s: PR thickness = {}".format(self.step, self.recipe, self.time,
                                                                       thickness_pr))
                # df['z_surf'] = df['z_surf'].where(df['z_surf'] > 0, 0)

                pf = {pk_lbl: ProcessFeature(graycart_wafer_feature=self.features[pk_lbl],
                                             step=self.step,
                                             process_type=self.process_type,
                                             subpath=self.subpath,
                                             dfpk=df,
                                             peak_properties=pk_details['peak_properties'],
                                             zr_standoff=zr_standoff,
                                             ),

                      }

                process_features.update(pf)

        self.features.update(process_features)

    def add_3d_profilometry_to_features(self,
                                        plot_fits=False,
                                        perform_rolling_on=False,
                                        evaluate_signal_processing=False,
                                        downsample=5,
                                        width_rel_radius=0.01,
                                        peak_rel_height=0.95,
                                        fit_func='parabola',
                                        prominence=1,
                                        plot_width_rel_target=1.1,
                                        thickness_pr=0,
                                        ):

        # get files
        process_files = self.collect_profilometry_files(scan_3d=True)

        # iterate through process file list
        process_features = {}

        # each item contains: f: path to scan data; fids, flbls: the fids and flbls to assign peaks within scan data
        """process_dict_files = {'a1': {'fid': 1,
                                     '0': path_to_rot0,
                                     '45': path_to_rot45,
                                     '90': path_to_rot90,
                                     }"""
        for flbls, rot_dict in process_files.items():
            fids = rot_dict['fid']
            del rot_dict['fid']
            f_rots = rot_dict.keys()

            # get data from associated features
            fn = flbls

            # get feature
            gcff = self.features[fn]

            # evaluate each rotation
            df_rots = []
            input_sampling_rates = []
            for rot, path_to_data in rot_dict.items():

                if rot != 0:
                    path_to_data = '3d/' + path_to_data

                f = path_to_data

                dfpk, peak_details, input_sampling_rate, output_sampling_rate, raw_profile = self.process_profilometry(
                    fids, [flbls], f,
                    plot_fits=plot_fits,
                    perform_rolling_on=perform_rolling_on,
                    evaluate_signal_processing=evaluate_signal_processing,
                    downsample=downsample,
                    width_rel_radius=width_rel_radius,
                    peak_rel_height=peak_rel_height,
                    fit_func=fit_func,
                    prominence=prominence,
                    plot_width_rel_target=plot_width_rel_target,
                )

                dfpk['rot'] = rot
                df_rots.append(dfpk)
                input_sampling_rates.append(input_sampling_rate)

            # average rotations
            df_rots = pd.concat(df_rots)
            input_sampling_rate = np.mean(input_sampling_rates)

            # calculate (x, y, z) coordinates from cylindrical coordinates
            dfxyz = process.rotate_cylindrical_coords(df=df_rots.copy(), input_col='r', output_cols=('rx', 'ry'), rotation_col='rot')
            # plotting.plot_3d_profile(df=dfxyz, px='rx', py='ry', pz='z', scale_z=50, aspect_ratio_z=1.25, view_height=35, view_rotation=120, path_save=None)

            # average rotations for simple 2D plotting and evaluation
            df_rots = process.bin_by_column(df_rots, column_to_bin='x',
                                            number_of_bins=int(len(df_rots) / len(f_rots)),
                                            round_to_decimal=2)
            dfpk = df_rots.groupby('bin').mean().reset_index()

            # determine the x-data sampling rate
            output_sampling_rate = np.round(dfpk['x'].diff().mean(), 2)
            samples_per_radius = gcff.target_radius / output_sampling_rate

            # find peaks
            min_width = samples_per_radius * 1.0
            width_space = samples_per_radius * width_rel_radius
            dfpk, peak_details = process.find_multi_peak(dfpk,
                                                         peak_labels=flbls,
                                                         peak_ids=fids,
                                                         target_width=gcff.target_radius,
                                                         min_width=min_width,
                                                         width_space=width_space,
                                                         fit_func=fit_func,
                                                         rel_height=peak_rel_height,
                                                         prominence=prominence,
                                                         plot_width_rel_target=plot_width_rel_target,
                                                         )

            # instantiate ProcessFeature to inherit GraycartFeature
            for pk_lbl, pk_details in peak_details.items():
                pk_details['peak_properties'].update({'input_sampling_rate': input_sampling_rate,
                                                      'sampling_rate': output_sampling_rate,
                                                      'raw_profile': df_rots})

                # declare 'df' for easier processing
                df = pk_details['df']

                # add z-offset for photoresist thickness
                df['z_surf'] = df['z'] + np.max([thickness_pr, 0])

                pf = {pk_lbl: SiliconFeature(graycart_wafer_feature=self.features[pk_lbl],
                                             step=self.step,
                                             process_type=self.process_type,
                                             subpath=self.subpath,
                                             dfpk=df,
                                             peak_properties=pk_details['peak_properties'],
                                             dfpk3d=dfxyz,
                                             ),

                      }

                process_features.update(pf)

        self.features.update(process_features)

    def process_profilometry(self,
                             fids, flbls, f,
                             plot_fits=False,
                             perform_rolling_on=False,
                             evaluate_signal_processing=False,
                             downsample=5,
                             width_rel_radius=0.01,
                             peak_rel_height=0.95,
                             fit_func='parabola',
                             prominence=1,
                             plot_width_rel_target=1.1,
                             ):
        """
        dfpk, peak_details, input_sampling_rate, output_sampling_rate, raw_profile = self.process_profilometry(
                             fids, flbls, f,
                             plot_fits=False,
                             perform_rolling_on=False,
                             evaluate_signal_processing=False,
                             downsample=5,
                             width_rel_radius=0.01,
                             peak_rel_height=0.95,
                             fit_func='parabola',
                             prominence=1,
                             plot_width_rel_target=1.1,
                             )
        """

        # get values
        drop_len = len(self._pread)

        # get data from associated features
        if isinstance(flbls, (list, np.ndarray)):
            fn = flbls[0]
        else:
            fn = f[:-drop_len]

        # get feature
        gcff = self.features[fn]

        # read scan
        df = io.read_scan(filepath=join(self._ppath, f), tool=self._ptool)
        raw_profile = df.copy()

        if '/' in f:
            f_save_id = f.split('/')[1][:-drop_len]
        else:
            f_save_id = f[:-drop_len]

        # convert 'read_units' to 'write_units'
        df['x'] = df['x'] * self._pread_x_units / self._pwrite_x_units  # x-units: microns
        df['z'] = df['z'] * self._pread_y_units / self._pwrite_y_units  # z-units: microns
        input_sampling_rate = np.round(df['x'].diff().mean(), 2)

        # evaluate effects of signal processing
        if evaluate_signal_processing:
            self.conditional_evaluate_signal_processing(filename=fn, df=df, perform_rolling_on=True)

        # rolling average to reduce random peaks disturbance on peak_finder (conditional on 'step' and 'fid')
        df = self.conditional_rolling_average(filename=fn,
                                              df=df,
                                              perform_rolling_on=perform_rolling_on,
                                              rolling_window=10,
                                              min_periods=1,
                                              rolling_function='median',
                                              )

        # input sampling rate
        rolling_sampling_rate = np.round(df['x'].diff().mean(), 2)
        input_samples_per_radius = gcff.dr / rolling_sampling_rate
        input_samples_outside_radius = 0.5 * len(df) - input_samples_per_radius

        # correct for profile tilt
        tilt_corr_samples = int(input_samples_per_radius * (plot_width_rel_target - 1))
        tilt_z = df.iloc[-tilt_corr_samples:].z.mean() - df.iloc[:tilt_corr_samples].z.mean()
        tilt_x = df.iloc[-tilt_corr_samples:].x.mean() - df.iloc[:tilt_corr_samples].x.mean()
        tilt_deg = np.rad2deg(np.arcsin(tilt_z / tilt_x / 1000))

        # tilt correction
        y_corr, tilt_func, popt, rmse, r_squared = process.tilt_correct_array(x=df.x.to_numpy(),
                                                                              y=df.z.to_numpy(),
                                                                              num_points=tilt_corr_samples,
                                                                              )
        df['z_raw'] = df['z']
        df['z'] = y_corr

        if plot_fits:
            plotting.plot_tilt_correct_array(x=df.x.to_numpy(),
                                             y=df.z_raw.to_numpy(),
                                             num_points=tilt_corr_samples,
                                             fit_func=tilt_func,
                                             popt=popt,
                                             rmse=rmse,
                                             r_squared=r_squared,
                                             save_id='Step{}_{}'.format(self.step, f_save_id),
                                             save_path=self.ppath,
                                             save_type='.png',
                                             )
        elif rmse > 0.25:
            print("Tilt corr fit, rmse: {} microns".format(rmse) + r'$, R^2=$' + ' {}'.format(r_squared))

        # down-sample to reduce computation and file size
        if downsample != 0:
            df = process.downsample_dataframe(df,
                                              xcol='x',
                                              ycol='z',
                                              num_points=None,
                                              sampling_rate=downsample,
                                              )

        # determine the x-data sampling rate
        output_sampling_rate = np.round(df['x'].diff().mean(), 2)
        samples_per_radius = gcff.target_radius / output_sampling_rate

        min_width = samples_per_radius * 1.0
        width_space = samples_per_radius * width_rel_radius

        # find peak
        dfpk, peak_details = process.find_multi_peak(df,
                                                     peak_labels=flbls,
                                                     peak_ids=fids,
                                                     target_width=gcff.target_radius,
                                                     min_width=min_width,
                                                     width_space=width_space,
                                                     fit_func=fit_func,
                                                     rel_height=peak_rel_height,
                                                     prominence=prominence,
                                                     plot_width_rel_target=plot_width_rel_target,
                                                     )

        return dfpk, peak_details, input_sampling_rate, output_sampling_rate, raw_profile

    # ------------------------------------------------------------------------------------------------------------------
    # DATA PROCESSING FUNCTIONS

    def conditional_rolling_average(self, filename, df, perform_rolling_on, rolling_window=10, min_periods=1,
                                    rolling_function='average'):
        """
        Rolling average to reduce random peaks disturbance on peak_finder

        :param filename:
        :param df:
        :param perform_rolling_on:
        :param rolling_window:
        :param min_periods:
        :return:
        """

        if isinstance(perform_rolling_on, list):
            for step_filename in perform_rolling_on:

                if len(step_filename) == 3:
                    rolling_window = step_filename[2]

                if self.step == step_filename[0] and filename == step_filename[1]:

                    if rolling_function == 'median':
                        df = df.rolling(rolling_window, min_periods).median()
                    else:
                        df = df.rolling(rolling_window, min_periods).mean()

                    print("Step {}, file {}: rolling {}(window={}, min_period={})".format(step_filename[0],
                                                                                          step_filename[1],
                                                                                          rolling_function,
                                                                                          rolling_window,
                                                                                          min_periods))

        elif perform_rolling_on is True:
            if rolling_function == 'median':
                df = df.rolling(rolling_window, min_periods).median()
            else:
                df = df.rolling(rolling_window, min_periods).mean()
        else:
            pass

        return df

    def conditional_evaluate_signal_processing(self, filename, df, perform_rolling_on):
        """
        Rolling average to reduce random peaks disturbance on peak_finder

        :param filename:
        :param df:
        :param perform_rolling_on:
        :param rolling_window:
        :param min_periods:
        :return:
        """
        import matplotlib.pyplot as plt
        min_periods = 1
        lw = 0.5

        if isinstance(perform_rolling_on, list):
            for step_filename in perform_rolling_on:
                if self.step == step_filename[0] and filename == step_filename[1]:
                    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(5.5, 4.5))

                    ax0.plot(df.x, df.z, '.', ms=0.5, color='k', alpha=0.5, label='Raw')
                    ax0.legend()

                    for window_size in [5, 10, 25]:
                        dff = df.rolling(window_size, min_periods).mean()
                        ax1.plot(dff.x, dff.z, linewidth=lw, label=window_size)

                        dff = df.rolling(window_size, min_periods).median()
                        ax2.plot(dff.x, dff.z, linewidth=lw, label=window_size)

                        dff = process.interpolate_dataframe(df,
                                                            xcol='x',
                                                            ycol='z',
                                                            num_points=None,
                                                            sampling_rate=window_size)
                        ax3.plot(dff.x, dff.z, linewidth=lw, label=window_size)

                    ax1.legend(title='Mean R.A.')
                    ax2.legend(title='Median R.A.')
                    ax3.legend(title='Interpolate')

                plt.suptitle('Step {}: {}'.format(self.step, filename))
                plt.tight_layout()
                plt.show()

        elif perform_rolling_on is True:
            fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(5.5, 4.5))

            ax0.plot(df.x, df.z, '.', ms=0.5, color='k', alpha=0.5, label='Raw')
            ax0.legend()

            for window_size in [5, 10, 25]:
                dff = df.rolling(window_size, min_periods).mean()
                ax1.plot(dff.x, dff.z, linewidth=lw, label=window_size)

                dff = df.rolling(window_size, min_periods).median()
                ax2.plot(dff.x, dff.z, linewidth=lw, label=window_size)

                dff = process.interpolate_dataframe(df,
                                                    xcol='x',
                                                    ycol='z',
                                                    num_points=None,
                                                    sampling_rate=window_size)
                ax3.plot(dff.x, dff.z, linewidth=lw, label=window_size)

            ax1.legend(title='Mean R.A.')
            ax2.legend(title='Median R.A.')
            ax3.legend(title='Interpolate')

        plt.suptitle('Step {}: {}'.format(self.step, filename))
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # PLOTTING FUNCTIONS

    def plot_profilometry_features(self, save_fig=False):
        plotting.plot_process_feature_profilometry(self, save_fig=save_fig)

    def plot_profilometry_feature_fits(self, save_fig=False):
        plotting.plot_process_feature_fit_profilometry(self, save_fig=save_fig)

    # ------------------------------------------------------------------------------------------------------------------
    # UTILITY FUNCTIONS

    def collect_profilometry_files(self, scan_3d=False):
        """
        process_files = self.collect_profilometry_files()

        :return:
        """

        if self.ptool == 'Dektak':
            process_files = io.collect_Dektak(self)
        elif self.ptool == 'KLATencor-P7':
            if scan_3d:
                process_files = io.collect_KLATencor_3D(self)
            else:
                process_files = io.collect_KLATencor(self)
        else:
            raise ValueError('No available I/O function for {}'.format(self.ptool))

        return process_files

    # ------------------------------------------------------------------------------------------------------------------
    # PROPERTIES

    @property
    def num_features(self):
        return len(self.features)

    @property
    def num_process_features(self):
        num_process_features = len([pf for pf in self.features.values() if isinstance(pf, (ProcessFeature, SiliconFeature))])
        return num_process_features

    @property
    def feature_ids(self):
        f_ids = [f.fid for f in self.features.values()]
        return f_ids

    @property
    def feature_labels(self):
        f_lbls = [f.label for f in self.features.values()]
        return f_lbls

    @property
    def descriptor(self):
        return "Step {}, {}: {} s, {}".format(self.step, self.process_type, self.time, self.recipe)

    @property
    def ptool(self):
        return self._ptool

    @property
    def ppath(self):
        return self._ppath

    @property
    def pread(self):
        return self._pread

    @property
    def pwrite(self):
        return self._pwrite


"""
class CoatProcess(GraycartProcess):
    def __init__(self, graycart_process):

        GraycartProcess.__init__(self,
                                 graycart_process.step,
                                 graycart_process.process_type,
                                 graycart_process.recipe,
                                 graycart_process.time,
                                 graycart_process.details,
                                 graycart_process.basepath,
                                 graycart_process.subpath,
                                 graycart_process.features,
                                 graycart_process.data,
                                 materials=graycart_process.materials,
                                 )

    def process_material(self):
        if 'Photoresist' in self.materials.keys():
            self.materials['Photoresist'].apply_process(process=self.process_type,
                                                        recipe=self.recipe,
                                                        time=self.time,
                                                        )
        else:
            gcm = PhotoresistMaterial(name=self.details['Photoresist'],
                                      properties=None,
                                      spin_rpm=self.details['RPM'],
                                      spin_time=self.time,
                                      )
            self.materials.update({'Photoresist': gcm})


class EtchProcess(GraycartProcess):
    def __init__(self, graycart_process):

        GraycartProcess.__init__(self,
                                 graycart_process.step,
                                 graycart_process.process_type,
                                 graycart_process.recipe,
                                 graycart_process.time,
                                 graycart_process.details,
                                 graycart_process.basepath,
                                 graycart_process.subpath,
                                 graycart_process.features,
                                 graycart_process.data,
                                 materials=graycart_process.materials,
                                 )

    def process_material(self):
        materials = {}
        for mat, gcm in self.materials.items():
            gcm.apply_process(process=self.process_type, recipe=self.recipe, time=self.time)

"""
#