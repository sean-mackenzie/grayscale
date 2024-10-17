from os.path import join, isdir, exists
from os import listdir

from pandas import read_excel

import pandas as pd
import numpy as np
from scipy.special import erf
from scipy import integrate

from graycart.utils import process, fit


class GraycartFeature(object):

    def __init__(self, did, dlbl, path_design, dxc=0, dyc=0, path_target=None, bits=8):
        super(GraycartFeature, self).__init__()

        self.did = did
        self.dlbl = dlbl

        self.path_design = path_design
        self.dxc = dxc
        self.dyc = dyc

        self.bits = bits

        self._dfd = None
        self.dr = None
        self.read_design_file()

        self.path_target = path_target
        self.dft = None
        self._dft = None
        self.read_target_file()

    def __repr__(self):
        class_ = 'GraycartFeature'
        repr_dict = {'Design ID': self.did,
                     'Design Label': self.dlbl,
                     'Design(xc, yc, r)': '({}, {}, {})'.format(self.dxc, self.dyc, self.dr)
                     }
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def read_design_file(self):
        df = read_excel(self.path_design)
        df = df.sort_values('r')
        self._dfd = df

        if 'ro' in df.columns:
            self.dr = df.ro.max()
        else:
            self.dr = df.r.max()

    def read_target_file(self):

        if exists(self.path_target):
            dft = pd.read_excel(self.path_target)

        else:
            print("No target profile found at {}. Using standard erf(r) function instead.".format(self.path_target))

            x = np.linspace(-2, 2, self.bit_resolution)
            px = (x + 2) / 2
            py = erf(x) / 2 - 0.5
            dft = pd.DataFrame(np.vstack([px, py]).T, columns=['r', 'z'])

        self.dft = dft
        self._dft = dft.copy()

    def resize_target_profile(self, radius=1, amplitude=1):
        self.dft['r'] = self._dft['r'] * radius / 2
        self.dft['z'] = self._dft['z'] * amplitude

    @property
    def bit_resolution(self):
        return 2 ** self.bits

    @property
    def dfd(self):
        return self._dfd

    @property
    def mdft(self):
        dft_mirrored = self.dft.copy()
        dft_mirrored['r'] = dft_mirrored['r'] * -1
        mdft = pd.concat([dft_mirrored, self.dft])
        mdft = mdft.sort_values('r')
        return mdft


class WaferFeature(GraycartFeature):

    def __init__(self, graycart_feature, fid, label, fxc, fyc, dose, focus, feature_extents=None, feature_spacing=None,
                 target_radius=None, target_depth=None, target_profile=None):

        super().__init__(graycart_feature.did, graycart_feature.dlbl, graycart_feature.path_design,
                         dxc=graycart_feature.dxc, dyc=graycart_feature.dyc, path_target=graycart_feature.path_target)

        self.fid = fid  # Feature ID: unique identifier across the wafer
        self.label = label  # Feature Label: unique identifying label for each 'feature' across the wafer
        self.fxc = fxc
        self.fyc = fyc
        self.dose = dose
        self.focus = focus

        self.dfe = None
        self.compute_exposure_profile()

        self.xc = self.fxc + self.dxc
        self.yc = self.fyc + self.dyc

        if feature_extents is None:
            feature_extents = self.dr * 1.15
        self.feature_extents = feature_extents
        self.feature_spacing = feature_spacing

        if target_radius is None:
            target_radius = self.dr
        self.target_radius = target_radius
        self.target_depth = target_depth
        self.target_profile = target_profile

    def __repr__(self):
        class_ = 'GraycartFeature'
        repr_dict = {'Label': self.label,
                     'Feature ID': self.fid,
                     'Design ID': self.did,
                     'Feature(xc, yc, r)': '({}, {}, {})'.format(self.xc, self.yc, self.dr),
                     'Dose, Focus': '{}, {}'.format(self.dose, self.focus),

                     }
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def compute_exposure_profile(self):
        dfe = self.dfd.copy()
        dfe['percent_dose'] = dfe['l'] / self.bit_resolution
        dfe['exposure_dose'] = dfe['percent_dose'] * self.dose
        self.dfe = dfe


class ProcessFeature(WaferFeature):

    def __init__(self, graycart_wafer_feature, step, process_type, subpath, dfpk, peak_properties,
                 dfpk3d=None, zr_standoff=None):
        super().__init__(graycart_wafer_feature,
                         fid=graycart_wafer_feature.fid,
                         label=graycart_wafer_feature.label,
                         fxc=graycart_wafer_feature.fxc,
                         fyc=graycart_wafer_feature.fyc,
                         dose=graycart_wafer_feature.dose,
                         focus=graycart_wafer_feature.focus,
                         feature_extents=graycart_wafer_feature.feature_extents,
                         feature_spacing=graycart_wafer_feature.feature_spacing,
                         target_radius=graycart_wafer_feature.target_radius,
                         target_depth=graycart_wafer_feature.target_depth,
                         target_profile=graycart_wafer_feature.target_profile,
                         )

        self.step = step
        self.process_type = process_type
        self.subpath = subpath

        self.dfpk = dfpk
        self.peak_properties = peak_properties

        if zr_standoff is None:
            self.z_standoff = self.calculate_z_standoff()
            self.r_standoff = 0.0
        else:
            self.z_standoff = zr_standoff[0]
            self.r_standoff = zr_standoff[1]

        self.dfpk3d = dfpk3d

        # derived values
        self.exposure_func = None
        self.target_rmse = None
        self.target_rmse_percent_depth = None
        self.target_r_squared = None
        self.target_corr = None
        self.target_volume = None
        self.volume = None
        self.target_volume_error = None

    def __repr__(self):
        class_ = 'ProcessFeature'
        repr_dict = {'Label': self.label,
                     'Feature ID': self.fid,
                     'Design ID': self.did,
                     'Step': self.step,
                     'Process Type': self.process_type,
                     }
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def calculate_profile_to_target_error(self, target_radius, target_depth):
        self.resize_target_profile(radius=target_radius, amplitude=target_depth)
        dft = self.mdft.copy()  # mirrored target profile
        dfpk = self.dfpk.copy()

        target_radius_limits = [dft.r.min(), dft.r.max()]
        target_samples = len(dft)
        target_sampling_rate = target_samples / (target_radius_limits[1] - target_radius_limits[0])

        dfpk = dfpk[(dfpk['r'] > target_radius_limits[0]) & (dfpk['r'] < target_radius_limits[1])]
        profile_radius_limits = [dfpk.r.min(), dfpk.r.max()]
        profile_samples = len(dfpk)
        profile_sampling_rate = profile_samples / (profile_radius_limits[1] - profile_radius_limits[0])

        max_sampling_rate = np.max([target_sampling_rate, profile_sampling_rate])

        tx, ty = process.resample_array(x=dft.r.to_numpy(), y=dft.z.to_numpy(), num_points=profile_samples,
                                        sampling_rate=None)
        px, py = process.resample_array(x=dfpk.r.to_numpy(), y=dfpk.z.to_numpy(), num_points=profile_samples,
                                        sampling_rate=None)

        rmse, r_squared = fit.calculate_fit_error(fit_results=py, data_fit_to=ty)

        self.target_rmse = rmse
        self.target_rmse_percent_depth = rmse / target_depth * 100
        self.target_r_squared = r_squared

    def calculate_volume_to_target_error(self):
        """ units: nanoliters (nL) """
        # area_target = integrate.trapezoid(self.mdft.z, self.mdft.r)
        # area_profile = integrate.trapezoid(self.dfpk.z, self.dfpk.r)
        self.target_volume = process.integrate_dataframe_radial(self.mdft, ycol='z', xcol='r', num_slices=100)
        self.volume = process.integrate_dataframe_radial(self.dfpk, ycol='z', xcol='r', num_slices=100)
        self.target_volume_error = self.volume - self.target_volume

    def correlate_profile_to_target(self, target_radius=None, target_depth=None):
        if target_depth is not None:
            self.resize_target_profile(radius=target_radius, amplitude=target_depth)

        dft = self.mdft.copy()  # mirrored target profile
        dfpk = self.dfpk.copy()

        target_radius_limits = [dft.r.min(), dft.r.max()]
        target_samples = len(dft)
        target_sampling_rate = target_samples / (target_radius_limits[1] - target_radius_limits[0])

        # dfpk = dfpk[(dfpk['r'] > target_radius_limits[0]) & (dfpk['r'] < target_radius_limits[1])]
        profile_radius_limits = [dfpk.r.min(), dfpk.r.max()]
        profile_samples = len(dfpk)
        profile_sampling_rate = profile_samples / (profile_radius_limits[1] - profile_radius_limits[0])

        max_sampling_rate = np.max([target_sampling_rate, profile_sampling_rate])

        tx, ty = process.resample_array(x=dft.r.to_numpy(), y=dft.z.to_numpy(), num_points=None,
                                        sampling_rate=max_sampling_rate)
        px, py = process.resample_array(x=dfpk.r.to_numpy(), y=dfpk.z.to_numpy(), num_points=None,
                                        sampling_rate=max_sampling_rate)

        corr = process.correlate_signals(ty, py)
        corr_idxmax = np.argmax(corr)
        corr = corr / np.max(corr)
        self.target_corr = corr

    def calculate_z_standoff(self):
        rdf = self.fold_dfpk.copy()  # measured 'exposure profile' of feature

        # get data points around r = target_radius
        r_around = 200
        rdf = rdf[np.abs(rdf['r'] - self.target_radius) < r_around]

        # the z(r_around).mean() should split data points into two groups: (1) unexposed, and (2) exposed areas
        z_around_mean = rdf['z'].mean()

        # the difference in the average z-positions of these two groups is the calculated z_standoff
        z_standoff = rdf[rdf['z'] > z_around_mean]['z'].mean() - rdf[rdf['z'] < z_around_mean]['z'].mean()

        return z_standoff

    def calculate_exposure_dose_depth_relationship(self):
        # inputs
        z_standoff = self.z_standoff
        r_standoff = self.r_standoff
        num_points = 2 ** 8  # 8-bit exposure resolution

        # datasets
        rdf = self.fold_dfpk.copy()  # 'exposure profile'
        dfe = self.dfe.copy()  # 'dose profile'

        # 1. filter: get r < R + r_standoff (not always necessary, but sometimes)
        rdf = rdf[rdf['r'].abs() < self.dr + r_standoff]

        # 2. perform rolling average to smooth (before filtering 'r' to reduce edge effects) (1 sample every 25 microns)
        # folded_sampling_rate = len(rdf) / rdf.r.max()
        folded_sampling_resolution = rdf.r.max() / len(rdf)
        rolling_window = int(np.round(20 / folded_sampling_resolution, 0))
        rdf = rdf.rolling(rolling_window, min_periods=1).mean()

        # 3. filter: get z only in "good" range for fitting
        #       filter #1: z < z_standoff       (because many values at zero exposure depth throws off curve_fit)
        #       filter #2: z > z.min() + 0.1    (because a flat bottom will also throw off curve_fit)
        rdf = rdf[(rdf['z'] < z_standoff) & (rdf['z'] > rdf['z'].min() + 0.1)]

        # 4a. maximum radial coordinates we can analyze (may depend on design or on profilometry data)
        radius_extent = np.min([rdf.r.max(), self.dr])

        # 4b. get r < outer radius of data
        rdf = rdf[rdf['r'].abs() < radius_extent]
        dfe = dfe[dfe['r'] < radius_extent]

        # 5. interpolate: (a) exposure profile; (b) dose profile
        ddf = process.downsample_dataframe(df=rdf, xcol='r', ycol='z', num_points=num_points, sampling_rate=None)
        ddfe = process.interpolate_dataframe(df=dfe, xcol='r', ycol='exposure_dose', num_points=num_points)

        # 6. slice and recombine these profiles to create the exposure mapping function
        dfmap_dose_to_depth = pd.DataFrame({'exposure_dose': ddfe.exposure_dose,
                                            'exposure_r': ddfe.r,
                                            'z': ddf.z,
                                            'profile_r': ddf.r,
                                            }
                                           )

        # 7. get functions limits
        depth_limits = [dfmap_dose_to_depth.z.min(), dfmap_dose_to_depth.z.max()]
        dose_limits = [dfmap_dose_to_depth.exposure_dose.min(), dfmap_dose_to_depth.exposure_dose.max()]

        # 8. calculate mapping functions
        dfmap_, dose_func, dose_popt = process.fit_func_dataframe(df=dfmap_dose_to_depth,
                                                                  xcol='z',
                                                                  ycol='exposure_dose',
                                                                  fit_func='exp_four',
                                                                  num_points=num_points,
                                                                  )

        dfmap_, depth_func, depth_popt = process.fit_func_dataframe(df=dfmap_dose_to_depth,
                                                                    xcol='exposure_dose',
                                                                    ycol='z',
                                                                    fit_func='exp_four',
                                                                    num_points=num_points,
                                                                    )

        self.exposure_func = {'depth_limits': depth_limits,
                              'z_standoff': z_standoff,
                              'dose_limits': dose_limits,
                              'depth_func': depth_func,
                              'depth_popt': depth_popt,
                              'dose_func': dose_func,
                              'dose_popt': dose_popt,
                              'dfmap': dfmap_dose_to_depth,
                              }

    def calculate_correct_exposure_profile(self, amplitude=None, target_depth=None,
                                           thickness_PR=None, thickness_PR_budget_below=None,
                                           dose=None, bit_resolution=None):

        if amplitude is None and target_depth is None:
            raise ValueError("Must define either 'amplitude' (depth) of photoresist profile you want, or \n"
                             "'target_depth' of eventual silicon feature, from which 'amplitude' will be calculated \n"
                             "according to the etch selectivity of Si:PR (right now, hard-coded as SF6+O2.V5)")

        if dose is None:
            dose = self.dose

        if bit_resolution is None:
            bit_resolution = self.bit_resolution

        # amplitude is the z-distance spanning the profile
        if amplitude is None:
            # amplitude = depth_range[1] - depth_range[0] - z_standoff
            etch_rate_Si = 2.25  # (um/min) SF6+O2.V6: 1.875, SF6+O2.V5: 2.25
            etch_rate_PR = 0.222  # (um/min) SF6+O2.V6
            etch_selectivity_Si_to_PR = etch_rate_Si / etch_rate_PR

            # the 'amplitude' should be derived from our target feature depth and etch selectivity
            amplitude = target_depth / etch_selectivity_Si_to_PR  # equates to ~6 microns

            # NOTE: this is a placeholder. This fork of data flow needs to be reviewed.
            z_offset = -0.125

        else:
            # Note: the following is used to redefine amplitude to fit
            # within the maximum exposure depth of "the settings used
            # for this particular feature." Although, this makes some
            # practical sense, it is confusing for the user, who expects
            # amplitude=A, but the function returns amplitude=<A.
            # So, I'm going to skip it for now and maybe come back to it.
            # calculate amplitude of target exposure profile
            depth_range = self.exposure_func['depth_limits']
            amplitude_max = np.max(depth_range) - np.min(depth_range)
            if amplitude > amplitude_max:
                print("NOTE: amplitude {} > max exposure depth {}. "
                      "Setting amplitude = max exposure depth.".format(amplitude, amplitude_max))
                amplitude = amplitude_max

                # z_offset is the z-distance we translate the target profile
                # if max exposure depth < amplitude, then we set z_offset to
                # a negligibly small non-zero value
                # so that we pattern the grayscale pattern over largest possible
                # depth of photoresist.
                z_offset = -0.125

            else:
                # z_offset is the z-distance we translate the target profile
                z_offset = amplitude - thickness_PR + thickness_PR_budget_below

        if z_offset > 0:
            raise ValueError("z_offset = {} > 0, but it should be < 0. Options to resolve are: \n"
                             "(1) reduce amplitude, \n"
                             "(2) increase PR thickness, \n"
                             "(3) reduce thickness_PR_budge_below.")

        # resize target profile to fit new target
        self.resize_target_profile(radius=self.target_radius * 2, amplitude=amplitude)
        dft = self.dft

        # the target profile assumes z begins at z = 0, if we want some extra PR protection, we must add z_standoff
        dft['z'] = dft['z'] + z_offset

        # calculate layers according to dose and resolution
        arr_z = dft.z.to_numpy()
        arr_popt = self.exposure_func['dose_popt']
        dft['exposure_intensity'] = self.exposure_func['dose_func'](arr_z, *arr_popt)
        dft['l'] = dft['exposure_intensity'] / dose * bit_resolution

    @property
    def mdfe(self):
        dfe_mirrored = self.dfe.copy()
        dfe_mirrored['r'] = dfe_mirrored['r'] * -1
        mdfe = pd.concat([dfe_mirrored, self.dfe.copy()])
        mdfe = mdfe.sort_values('r')
        return mdfe

    @property
    def fold_dfpk(self):
        df = self.dfpk.copy()
        df_r = df[df['r'] > 0]
        df_l = df[df['r'] < 0]
        df_l['r'] = df_l['r'] * -1
        fold_dfpk = pd.concat([df_l, df_r])
        fold_dfpk = fold_dfpk.sort_values('r')
        return fold_dfpk


class SiliconFeature(WaferFeature):

    def __init__(self, graycart_wafer_feature, step, process_type, subpath, dfpk, peak_properties, dfpk3d=None):
        super().__init__(graycart_wafer_feature,
                         fid=graycart_wafer_feature.fid,
                         label=graycart_wafer_feature.label,
                         fxc=graycart_wafer_feature.fxc,
                         fyc=graycart_wafer_feature.fyc,
                         dose=graycart_wafer_feature.dose,
                         focus=graycart_wafer_feature.focus,
                         feature_extents=graycart_wafer_feature.feature_extents,
                         feature_spacing=graycart_wafer_feature.feature_spacing,
                         target_radius=graycart_wafer_feature.target_radius,
                         target_depth=graycart_wafer_feature.target_depth,
                         target_profile=graycart_wafer_feature.target_profile)

        self.step = step
        self.process_type = process_type
        self.subpath = subpath

        self.dfpk = dfpk
        self.peak_properties = peak_properties

        self.dfpk3d = dfpk3d

        # derived values
        self.exposure_func = None
        self.target_rmse = None
        self.target_rmse_percent_depth = None
        self.target_r_squared = None
        self.target_corr = None
        self.target_volume = None
        self.volume = None
        self.target_volume_error = None

    def __repr__(self):
        class_ = 'ProcessFeature'
        repr_dict = {'Label': self.label,
                     'Feature ID': self.fid,
                     'Design ID': self.did,
                     'Step': self.step,
                     'Process Type': self.process_type,
                     }
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def calculate_profile_to_target_error(self, target_radius, target_depth):
        self.resize_target_profile(radius=target_radius, amplitude=target_depth)
        dft = self.mdft.copy()  # mirrored target profile
        dfpk = self.dfpk.copy()

        target_radius_limits = [dft.r.min(), dft.r.max()]
        target_samples = len(dft)
        target_sampling_rate = target_samples / (target_radius_limits[1] - target_radius_limits[0])

        dfpk = dfpk[(dfpk['r'] > target_radius_limits[0]) & (dfpk['r'] < target_radius_limits[1])]
        profile_radius_limits = [dfpk.r.min(), dfpk.r.max()]
        profile_samples = len(dfpk)
        profile_sampling_rate = profile_samples / (profile_radius_limits[1] - profile_radius_limits[0])

        max_sampling_rate = np.max([target_sampling_rate, profile_sampling_rate])

        tx, ty = process.resample_array(x=dft.r.to_numpy(), y=dft.z.to_numpy(), num_points=profile_samples,
                                        sampling_rate=None)
        px, py = process.resample_array(x=dfpk.r.to_numpy(), y=dfpk.z.to_numpy(), num_points=profile_samples,
                                        sampling_rate=None)

        rmse, r_squared = fit.calculate_fit_error(fit_results=py, data_fit_to=ty)

        self.target_rmse = rmse
        self.target_rmse_percent_depth = rmse / target_depth * 100
        self.target_r_squared = r_squared

    def calculate_volume_to_target_error(self):
        """ units: nanoliters (nL) """
        # area_target = integrate.trapezoid(self.mdft.z, self.mdft.r)
        # area_profile = integrate.trapezoid(self.dfpk.z, self.dfpk.r)
        self.target_volume = process.integrate_dataframe_radial(self.mdft, ycol='z', xcol='r', num_slices=100)
        self.volume = process.integrate_dataframe_radial(self.dfpk, ycol='z', xcol='r', num_slices=100)
        self.target_volume_error = self.volume - self.target_volume

    def correlate_profile_to_target(self, target_radius=None, target_depth=None):
        if target_depth is not None:
            self.resize_target_profile(radius=target_radius, amplitude=target_depth)

        dft = self.mdft.copy()  # mirrored target profile
        dfpk = self.dfpk.copy()

        target_radius_limits = [dft.r.min(), dft.r.max()]
        target_samples = len(dft)
        target_sampling_rate = target_samples / (target_radius_limits[1] - target_radius_limits[0])

        # dfpk = dfpk[(dfpk['r'] > target_radius_limits[0]) & (dfpk['r'] < target_radius_limits[1])]
        profile_radius_limits = [dfpk.r.min(), dfpk.r.max()]
        profile_samples = len(dfpk)
        profile_sampling_rate = profile_samples / (profile_radius_limits[1] - profile_radius_limits[0])

        max_sampling_rate = np.max([target_sampling_rate, profile_sampling_rate])

        tx, ty = process.resample_array(x=dft.r.to_numpy(), y=dft.z.to_numpy(), num_points=None,
                                        sampling_rate=max_sampling_rate)
        px, py = process.resample_array(x=dfpk.r.to_numpy(), y=dfpk.z.to_numpy(), num_points=None,
                                        sampling_rate=max_sampling_rate)

        corr = process.correlate_signals(ty, py)
        corr_idxmax = np.argmax(corr)
        corr = corr / np.max(corr)
        self.target_corr = corr

    @property
    def mdfe(self):
        dfe_mirrored = self.dfe.copy()
        dfe_mirrored['r'] = dfe_mirrored['r'] * -1
        mdfe = pd.concat([dfe_mirrored, self.dfe.copy()])
        mdfe = mdfe.sort_values('r')
        return mdfe

    @property
    def fold_dfpk(self):
        df = self.dfpk.copy()
        df_r = df[df['r'] > 0]
        df_l = df[df['r'] < 0]
        df_l['r'] = df_l['r'] * -1
        fold_dfpk = pd.concat([df_l, df_r])
        fold_dfpk = fold_dfpk.sort_values('r')
        return fold_dfpk


# ------------------------------------------------------------------------------------------------------------------
# UTILITY FUNCTIONS

def initialize_designs(base_path, design_lbls, target_lbls, design_locs=None, design_ids=None):
    """

    :param base_path: top-level directory for wafer
    :param design_lbls: string identifier, must match data files for each feature (coordinates: x, y, r, l)
    :param target_lbls: string identifier, must match data files for each feature (coordinates: r, z)
    :param design_ids: numeric identifier for each design (e.g., 0, 1, 2,...)
    :param design_locs: the location of each feature in the mask design file
    :return:
    """

    if design_locs is None:
        design_locs = [[0, 0]]

    if design_ids is None:
        design_ids = np.arange(1, len(design_locs) + 1)

    designs = {}
    for k, design_loc, design_lbl, target_lbl in zip(design_ids, design_locs, design_lbls, target_lbls):
        path_design = join(base_path, 'mask', '{}.xlsx'.format(design_lbl))
        path_target = join(base_path, 'mask', 'target-profile_{}.xlsx'.format(target_lbl))

        designs.update({k: GraycartFeature(did=k,
                                           dlbl=design_lbl,
                                           path_design=path_design,
                                           dxc=design_loc[0],
                                           dyc=design_loc[1],
                                           path_target=path_target,
                                           )
                        }
                       )

    return designs


def initialize_design_features(designs, design_spacing, dose_lbls, focus_lbls, process_flow, fem_dxdy,
                               target_radius, target_depth=None):

    dose, dose_step, focus, focus_step = parse_items_from_process_flow(process_flow,
                                                                       process_type='Expose',
                                                                       items=['Dose', 'Dose Step', 'Focus', 'Focus Step'])

    features = {}
    ij = 0
    for i, dose_lbl in enumerate(dose_lbls):
        for j, focus_lbl in enumerate(focus_lbls):
            for k, gcf in designs.items():

                if len(designs) > 1:
                    feature_label = "{}{}_{}".format(dose_lbl, focus_lbl, gcf.dlbl)
                else:
                    feature_label = "{}{}".format(dose_lbl, focus_lbl)

                feature_xc = fem_dxdy[0] * j
                feature_yc = fem_dxdy[1] * i
                feature_dose = dose + i * dose_step
                feature_focus = focus + j * focus_step

                features.update({feature_label: WaferFeature(graycart_feature=gcf,
                                                             fid=ij,
                                                             label=feature_label,
                                                             fxc=feature_xc,
                                                             fyc=feature_yc,
                                                             dose=feature_dose,
                                                             focus=feature_focus,
                                                             feature_spacing=design_spacing,
                                                             target_radius=target_radius,
                                                             target_depth=target_depth,
                                                             )
                                 }
                                )
                ij += 1

    return features


def parse_items_from_process_flow(process_flow, process_type, items):
    item_list = []
    for step, details in process_flow.items():
        if details['process_type'] == process_type:
            item_list = [details['details'][item] for item in items]
    return item_list