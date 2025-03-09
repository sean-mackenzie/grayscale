from os.path import join, isdir
from os import makedirs
import numpy as np
import pandas as pd

import matplotlib.tri as tri
import matplotlib.pyplot as plt


from graycart.GraycartFeature import ProcessFeature
from graycart.utils import process

sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF2C00'
sciorange = '#FF9500'
scipurple = '#845B97'
sciblack = '#474747'
scigray = '#9e9e9e'
sci_color_list = [sciblue, scigreen, scired, sciorange, scipurple, sciblack, scigray]

plt.style.use(['science', 'ieee', 'std-colors'])  # , 'std-colors'
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)


# ------------------------------------------------------------------------------------------------------------------
# PLOTTING FUNCTIONS - GraycartWafer


def plot_target_profile_and_process_flow_backout(dft, est_process_flow, path_save=None, save_type='.png'):
    # plot
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True,
                                   figsize=(size_x_inches * 1.125, size_y_inches))

    ax1.plot(dft.r, dft.z, color='k', label='Target')
    ax1.set_ylabel(r'$z \: (\mu m)$')
    ax1.legend()  # loc='center left', title='Profile'

    for est_step, est_prcss in est_process_flow.items():
        ax2.plot(est_prcss['df']['r'], est_prcss['df']['z_surf'],
                 label="Pred., Step {}: {} s, {}".format(est_step + 1,
                                                         int(np.round(est_prcss['time'], 1)),
                                                         est_prcss['recipe']),
                 )

    ax2.set_ylabel(r'$z \: (\mu m)$')
    ax2.set_xlabel(r'$r \: (\mu m)$')
    ax2.legend()  # loc='center right', title='Grayscale Mask')

    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, 'figs',
                         'target-profile-and-backout-process-flow_did-{}'.format(
                             est_process_flow[0]['did']) + save_type),
                    )
    else:
        plt.show()

    plt.close()


def plot_all_exposure_dose_to_depth(df, path_save=None):
    fig, ax = plt.subplots()

    focii = df.focus.unique()
    markers = ['o', 's', 'd']

    for focus, mrkr in zip(focii, markers):
        dff = df[df['focus'] == focus]
        fids = dff.fid.unique()

        for fid in fids:
            dfi = dff[dff['fid'] == fid]

            ax.scatter(dfi.exposure_dose, dfi.z, s=1, marker=mrkr, alpha=0.5, label='{}({})'.format(fid, focus))

    ax.set_xlabel('Dose')
    ax.set_ylabel('Depth')
    ax.legend()

    plt.tight_layout()
    if path_save is not None:
        plt.savefig(path_save)
    else:
        plt.show()
    plt.close()


def plot_features_by_process(gcw, px, py, did=None, normalize=False, save_fig=False, save_type='.png'):
    # get dataframe
    df = gcw.dfps

    # filter on 'did': Design ID
    if isinstance(did, (int, float)):
        df = df[df['did'] == did]
    elif isinstance(did, (list, np.ndarray)):
        df = df[df['did'].isin(did)]
    else:
        did = 'all'

    if len(df) < 5:
        raise ValueError("Design ID {} not in: {}".format(did, df.did.unique()))

    # get steps and Feature IDs
    df = df.sort_values(by=['step', 'fid'])
    steps = df.step.unique()
    fids = df.fid.unique()

    # plot
    fig, axs = plt.subplots(nrows=len(steps), sharex=True,
                            figsize=(size_x_inches * 1.25, size_y_inches * len(steps) / 1.875),
                            facecolor='white')

    if not isinstance(axs, np.ndarray):
        axs = [axs]

    for step, ax in zip(steps, axs):
        for fid in fids:
            # get slice
            dfds = df[(df['fid'] == fid) & (df['step'] == step)].reset_index()
            if len(dfds) < 5:
                continue

            # get params
            design_id = int(dfds.iloc[0].did)
            dose = int(dfds.iloc[0].dose)
            focus = int(dfds.iloc[0].focus)

            # normalize
            if normalize:
                ax.plot(dfds[px], dfds[py] / dfds[py].min(), linewidth=0.5,
                        label="({}, {}, {})".format(design_id, dose, focus))
                ax.set_ylabel(r'$z/H$')
            else:
                if 'z_surf' in dfds.columns:
                    p1, = ax.plot(dfds[px], dfds.z_surf, linewidth=0.5,
                                  label="({}, {}, {})".format(design_id, dose, focus))
                    ax.fill_between(dfds[px], y1=dfds.z_surf, y2=0, where=dfds.z_surf > 0, ec='none',
                                    fc=p1.get_color(), alpha=0.25)
                else:
                    ax.plot(dfds[px], dfds[py], linewidth=0.5, label="({}, {}, {})".format(design_id, dose, focus))

                # ax.plot(dfds[px], dfds[py], linewidth=0.5, label="({}, {}, {})".format(design_id, dose, focus))
                ax.set_ylabel(r'$z \: (\mu m)$')

            if len(fids) > 8:
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1.1), title=r'$(d_{ID}, I_{o}, f)$', fontsize='x-small')
            else:
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1.1), title=r'$(d_{ID}, I_{o}, f)$', fontsize='small')
            ax.set_title(gcw.processes[step].descriptor)

    axs[-1].set_xlabel(r'$r \: (\mu m)$')

    plt.tight_layout()
    if save_fig:
        plt.savefig(join(gcw.path_results, 'figs',
                         'features_by_process_px-{}_py-{}_did-{}_norm-{}'.format(px, py, did, normalize) + save_type))
    else:
        plt.show()
    plt.close()


def plot_features_diff_by_process(gcw, px, py, did=None, normalize=False, save_fig=False, save_type='.png'):
    # get dataframe
    df = gcw.dfps

    # filter on 'did': Design ID
    if isinstance(did, (int, float)):
        df = df[df['did'] == did]
    elif isinstance(did, (list, np.ndarray)):
        df = df[df['did'].isin(did)]
    else:
        did = 'all'

    if len(df) < 5:
        raise ValueError("Design ID {} not in: {}".format(did, df.did.unique()))

    # get steps and Feature IDs
    df = df.sort_values(by=['step', 'fid'])
    steps = df.step.unique()
    fids = df.fid.unique()

    if len(steps) > 1:

        # plot
        fig, axs = plt.subplots(nrows=len(steps) - 1,
                                sharex=True,
                                figsize=(size_x_inches * 1.25, size_y_inches * (len(steps) - 1) / 1.875),
                                facecolor='white')

        if not isinstance(axs, np.ndarray):
            axs = [axs]

        for i, ax in enumerate(axs):
            step1 = steps[i]
            step2 = steps[i + 1]

            for fid in fids:
                # get slice
                dfds1 = df[(df['fid'] == fid) & (df['step'] == step1)].reset_index()
                dfds2 = df[(df['fid'] == fid) & (df['step'] == step2)].reset_index()

                if np.min([len(dfds1), len(dfds2)]) < 5:
                    continue

                # compute the difference
                max_sampling_rate = process.get_max_sampling_rate(dfds1, dfds2, px)
                x_new, y1, y2 = process.uniformize_x_dataframes(dfs=[dfds1, dfds2],
                                                                xy_cols=[px, py],
                                                                num_points=None,
                                                                sampling_rate=np.ceil(max_sampling_rate),
                                                                )
                dy = y2 - y1

                # get params
                design_id = int(dfds2.iloc[0].did)
                dose = int(dfds2.iloc[0].dose)
                focus = int(dfds2.iloc[0].focus)

                # normalize
                if normalize:
                    ax.plot(x_new, dy - np.mean(dy), linewidth=0.5,
                            label="({}, {}, {})".format(design_id, dose, focus))
                    ax.set_ylabel(r'$\Delta z - \overline{\Delta z}$')
                else:
                    ax.plot(x_new, dy, linewidth=0.5, label="({}, {}, {})".format(design_id, dose, focus))
                    ax.set_ylabel(r'$\Delta z \: (\mu m)$')

                ax.legend(loc='upper left', bbox_to_anchor=(1, 1.1), title=r'$(d_{ID}, I_{o}, f)$')
                ax.set_title(gcw.processes[step2].descriptor)

        axs[-1].set_xlabel(r'$r \: (\mu m)$')

        plt.tight_layout()
        if save_fig:
            plt.savefig(join(gcw.path_results, 'figs',
                             'features_diff_by_process_px-{}_py-{}_did-{}_norm-{}'.format(px, py, did,
                                                                                          normalize) + save_type))
        else:
            plt.show()
        plt.close()


def plot_features_diff_by_process_and_material(gcw, px, py='z_surf', did=None, normalize=False, save_fig=False,
                                               save_type='.png'):
    # get dataframe
    df = gcw.dfps

    # filter on 'did': Design ID
    if isinstance(did, (int, float)):
        df = df[df['did'] == did]
    elif isinstance(did, (list, np.ndarray)):
        df = df[df['did'].isin(did)]
    else:
        did = 'all'

    if len(df) < 5:
        raise ValueError("Design ID {} not in: {}".format(did, df.did.unique()))

    # get steps and Feature IDs
    df = df.sort_values(by=['step', 'fid'])
    steps = df.step.unique()
    fids = df.fid.unique()

    if len(steps) > 1:

        # plot
        fig, axs = plt.subplots(nrows=len(steps) - 1, ncols=2,
                                sharex=True,
                                figsize=(size_x_inches * 1.75, size_y_inches * (len(steps) - 1) / 1.5),
                                facecolor='white')

        if np.size(axs) < 3:
            axs = [axs]
        elif not isinstance(axs, np.ndarray):
            axs = [axs]

        for i, ax in enumerate(axs):
            step1 = steps[i]
            step2 = steps[i + 1]

            # get process details
            p_type = gcw.processes[step2].process_type
            p_recipe = gcw.processes[step2].recipe
            p_time = gcw.processes[step2].time

            # average 'dz' (etch depth) across all features
            dz = []

            for fid in fids:
                # get slice
                dfds1 = df[(df['fid'] == fid) & (df['step'] == step1)].reset_index()
                dfds2 = df[(df['fid'] == fid) & (df['step'] == step2)].reset_index()

                if np.min([len(dfds1), len(dfds2)]) < 5:
                    continue

                # get params
                design_id = int(dfds2.iloc[0].did)
                dose = int(dfds2.iloc[0].dose)
                focus = int(dfds2.iloc[0].focus)

                # get sampling rate
                max_sampling_rate = process.get_max_sampling_rate(dfds1, dfds2, px)

                # ---

                # evaluate z_surf > 0 and z_surf < 0

                # average 'dz' (etch depth) across all features
                dzm = []

                for k, zdir in enumerate([1, -1]):
                    # get slice
                    dfds11 = dfds1[(dfds1[py] * zdir > 0)].reset_index()
                    dfds22 = dfds2[(dfds2[py] * zdir > 0)].reset_index()

                    min_length = np.min([len(dfds11), len(dfds22)])
                    mean_vals = np.max(np.abs([dfds11[px].mean(), dfds22[px].mean()]))
                    if min_length < 25 or mean_vals > 1000:
                        dzm.append(0)
                        # ax[k].set_yticklabels([])
                        # ax[k].set_yticks([])
                        continue

                    # compute the difference
                    x_new, y1, y2 = process.uniformize_x_wrapper(dfs=[dfds11, dfds22],
                                                                 xy_cols=[px, py],
                                                                 num_points=None,
                                                                 sampling_rate=np.ceil(max_sampling_rate),
                                                                 split_sampling_rate=500,
                                                                 )
                    # 'dy' = change in feature height
                    dy = y2 - y1
                    dzm.append(np.mean(dy))

                    # normalize
                    if normalize:
                        ax[k].plot(x_new, dy - np.mean(dy), '.', ms=1,
                                   label="({}, {}, {})".format(design_id, dose, focus))
                        ax[k].set_ylabel(r'$\Delta z - \overline{\Delta z}$')
                    else:
                        ax[k].plot(x_new, dy, '.', ms=1, label="({}, {}, {})".format(design_id, dose, focus))
                        ax[k].set_ylabel(r'$\Delta z \: (\mu m)$')

                    # ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1.1), title=r'$(d_{ID}, I_{o}, f)$')

                # append [average dz(photoresist), average dz(silicon)]
                dz.append(dzm)

            # average dz(photoresist) and dz(silicon)
            dzs = np.array(dz)
            dz_pr = np.mean(dzs[:, 0][dzs[:, 0] != 0])
            dz_si = np.mean(dzs[:, 1][dzs[:, 1] != 0])

            if not np.isnan(dz_pr):
                dz_pr_rate = int(np.round((dz_pr * 1000) / (p_time / 60), 0))  # units: nm/min
                ax[0].set_title(
                    'Step {}, {}: '.format(step2, p_recipe) + r'$\Delta z = $' + ' {} nm/min'.format(dz_pr_rate))

            if not np.isnan(dz_si):
                dz_si_rate = int(np.round((dz_si * 1000) / (p_time / 60), 0))  # units: nm/min
                ax[1].set_title(
                    'Step {}, {}: '.format(step2, p_recipe) + r'$\Delta z = $' + ' {} nm/min'.format(dz_si_rate))

        # set x-labels
        axs[-1][0].set_xlabel(r'$r \: (\mu m)$')
        axs[-1][1].set_xlabel(r'$r \: (\mu m)$')

        plt.tight_layout()
        if save_fig:
            plt.savefig(join(gcw.path_results, 'figs',
                             'features_diff_by_process-mat_px-{}_py-{}_did-{}_norm-{}'.format(px, py, did,
                                                                                              normalize) + save_type))
        else:
            plt.show()
        plt.close()


def plot_processes_by_feature(gcw, px, py, did=None, normalize=False, save_fig=False, save_type='.png'):
    # get dataframe
    df = gcw.dfps

    # filter on 'did': Design ID
    if isinstance(did, (int, float)):
        df = df[df['did'] == did]
    elif isinstance(did, (list, np.ndarray)):
        df = df[df['did'].isin(did)]
    else:
        did = 'all'

    if len(df) < 5:
        raise ValueError("Design ID {} not in: {}".format(did, df.did.unique()))

    # get Steps and Feature IDs
    df = df.sort_values(by=['step', 'fid'])
    steps = df.step.unique()
    fids = df.fid.unique()

    # plot
    fig, axs = plt.subplots(nrows=len(fids), sharex=True,
                            figsize=(size_x_inches * 1.125, size_y_inches * len(fids) / 1.5),
                            facecolor='white')

    if len(fids) == 1:
        axs = [axs]

    for fid, ax in zip(fids, axs):
        for step in steps:
            # get slice
            dfds = df[(df['fid'] == fid) & (df['step'] == step)].reset_index()
            if len(dfds) < 5:
                continue

            # get params
            design_id = int(dfds.iloc[0].did)
            dose = int(dfds.iloc[0].dose)
            focus = int(dfds.iloc[0].focus)

            # normalize
            if normalize:
                ax.plot(dfds[px], dfds[py] / dfds[py].min(), linewidth=0.5, label="{}".format(step))
                ax.set_ylabel(r'$z/H$')
            else:
                # calculate height at middle index of 1D array
                arr_length = int(np.round(len(dfds) / 2))
                avg_half_length = 25
                py_at_mid_px = dfds.iloc[arr_length - avg_half_length: arr_length + avg_half_length][py].mean()

                # fill between (optional)
                if 'z_surf' in dfds.columns:
                    p1, = ax.plot(dfds[px], dfds.z_surf, linewidth=0.5,
                                  label="{}, H={}".format(step, np.round(py_at_mid_px, 2)))
                    ax.fill_between(dfds[px], y1=dfds.z_surf, y2=0, where=dfds.z_surf > 0, ec='none',
                                    fc=p1.get_color(), alpha=0.25)
                else:
                    ax.plot(dfds[px], dfds[py], linewidth=0.5, label="{}, H={}".format(step, np.round(py_at_mid_px, 2)))

                ax.set_ylabel(r'$z \: (\mu m)$')

            ax.legend(loc='upper left', bbox_to_anchor=(1, 1.1), title='Step')
            ax.set_title(r'$(f_{ID}, d_{ID}, I_{o}, f)=$' + '({}, {}, {}, {})'.format(fid, design_id, dose, focus),
                         fontsize=6)

    axs[-1].set_xlabel(r'$r \: (\mu m)$')

    plt.tight_layout()
    if save_fig:
        plt.savefig(join(gcw.path_results, 'figs',
                         'processes_by_feature_px-{}_py-{}_did-{}_norm-{}'.format(px, py, did, normalize) + save_type))
    else:
        plt.show()
    plt.close()


def compare_target_to_features_by_process(gcw, px, py, did=None, normalize=False, save_fig=False, save_type='.png'):
    # get dataframe
    df = gcw.dfps

    # filter on 'did': Design ID
    if isinstance(did, (int, float)):
        df = df[df['did'] == did]
    elif isinstance(did, (list, np.ndarray)):
        df = df[df['did'].isin(did)]
    else:
        did = 'all'

    if len(df) < 5:
        raise ValueError("Design ID {} not in: {}".format(did, df.did.unique()))

    # get steps and Feature IDs
    df = df.sort_values(by=['step', 'fid'])
    steps = df.step.unique()
    fids = df.fid.unique()

    # plot
    fig, axs = plt.subplots(nrows=len(steps), sharex=True,
                            figsize=(size_x_inches * 1.25, size_y_inches * len(steps) / 1.875),
                            facecolor='white')

    if not isinstance(axs, np.ndarray):
        axs = [axs]

    for step, ax in zip(steps, axs):
        z_amplitude = []
        for fid in fids:
            # get slice
            dfds = df[(df['fid'] == fid) & (df['step'] == step)].reset_index()
            if len(dfds) < 5:
                continue

            # get params
            design_id = int(dfds.iloc[0].did)
            dose = int(dfds.iloc[0].dose)
            focus = int(dfds.iloc[0].focus)

            # normalize
            if normalize:
                ax.plot(dfds[px], dfds[py] / (dfds[py].min() - dfds[py].max()), linewidth=0.5,
                        label="({}, {}, {})".format(design_id, dose, focus))
                ax.set_ylabel(r'$z/H$')
            else:
                if py == 'z_surf':
                    p1, = ax.plot(dfds[px], dfds.z_surf, linewidth=0.5,
                                  label="({}, {}, {})".format(design_id, dose, focus))
                    ax.fill_between(dfds[px], y1=dfds.z_surf, y2=0, where=dfds.z_surf > 0, ec='none',
                                    fc=p1.get_color(), alpha=0.25)
                else:
                    ax.plot(dfds[px], dfds[py], linewidth=0.5, label="({}, {}, {})".format(design_id, dose, focus))

                ax.set_ylabel(r'$z \: (\mu m)$')

            z_amplitude.append(dfds[py].min())

        # --------------------------------------------------------------------------------------------------------
        # UNDER CONSTRUCTION

        # get target profile for this design
        gcwf = gcw.get_feature(fids[0], step)  # gcwf = gcw.features['a1']

        target_did = gcwf.did
        target_radius = gcwf.target_radius
        mean_amplitude = np.abs(np.mean(z_amplitude))
        mdft = gcwf.mdft.copy()

        """ Conditional 'units' errors... the result of a non-resized target profile. """
        if mdft['r'].max() < dfds[px].max() / 10:
            mdft['r'] = mdft['r'] * target_radius / 2
        if mdft['z'].abs().max() < dfds[py].abs().max() / 5:
            mdft['z'] = mdft['z'] / mdft['z'].abs().max() * mean_amplitude  # normalize then adjust amplitude

        if normalize:
            ax.plot(mdft['r'], mdft['z'] / mdft['z'].min(), linewidth=0.5, linestyle='dotted', color='k',
                    label="Target({})".format(target_did))
        else:
            ax.plot(mdft['r'], mdft['z'], linewidth=0.5, linestyle='dotted', color='k',
                    label="Target({})".format(target_did))

        # --------------------------------------------------------------------------------------------------------

        ax.legend(loc='upper left', bbox_to_anchor=(1, 1.1), title=r'$(d_{ID}, I_{o}, f)$')
        ax.set_title(gcw.processes[step].descriptor)

    axs[-1].set_xlabel(r'$r \: (\mu m)$')

    plt.tight_layout()
    if save_fig:
        plt.savefig(join(gcw.path_results, 'figs',
                         'compare_target-to-features_by_process_px-{}_py-{}_did-{}_norm-{}'.format(px, py, did,
                                                                                                   normalize) + save_type))
    else:
        plt.show()
    plt.close()


def estimated_target_profiles(gcw, px, py, etch_recipe_PR, etch_recipe_Si, target_depth, thickness_PR_budget,
                              include_target=True, save_fig=False, save_type='.png'):
    # get dataframe
    df = gcw.dfps

    # get steps and Feature IDs
    df = df.sort_values(by=['step', 'fid'])
    fids = df.fid.unique()

    for fid in fids:
        dff = df[(df['fid'] == fid)]
        steps = dff.step.unique()

        if len(steps) > 0:

            # plot
            if len(steps) == 1:
                fig, axs = plt.subplots(ncols=2,
                                        figsize=(size_x_inches * 2, size_y_inches),
                                        facecolor='white')
            else:
                fig, axs = plt.subplots(nrows=len(steps), ncols=2,
                                        sharex=True,
                                        figsize=(size_x_inches * 2, size_y_inches * (len(steps) - 1) / 1.25),
                                        facecolor='white')

            if np.size(axs) < 3:
                axs = [axs]
            elif not isinstance(axs, np.ndarray):
                axs = [axs]

            for step, ax in zip(steps, axs):

                # get process details
                p_type = gcw.processes[step].process_type
                p_recipe = gcw.processes[step].recipe
                p_time = gcw.processes[step].time
                p_materials = gcw.processes[step].materials

                # get slice
                dfds = df[(df['fid'] == fid) & (df['step'] == step)].reset_index()

                if len(dfds) < 5:
                    continue

                # get params
                design_id = int(dfds.iloc[0].did)
                dose = int(dfds.iloc[0].dose)
                focus = int(dfds.iloc[0].focus)

                # plot
                ax[0].plot(dfds[px], dfds[py], '.', ms=1, label="({}, {}, {})".format(design_id, dose, focus))
                ax[0].set_ylabel(r'$z \: (\mu m)$')
                ax[0].legend(loc='upper center')

                ax[1].plot(dfds[px], dfds[py], '.', ms=0.5,
                           label="Meas., Step {}: {} s, {}".format(step, p_time, p_recipe))

                # ---

                # estimate process flow and profile
                est_process_flow = process.estimate_process_to_achieve_target(dfds, px, py,
                                                                              p_materials,
                                                                              recipe_smOOth=etch_recipe_PR,
                                                                              recipe_SF6_O2=etch_recipe_Si,
                                                                              target_depth=target_depth,
                                                                              thickness_PR_budget=thickness_PR_budget,
                                                                              r_target=5,
                                                                              )

                z_amplitude = dfds[py].min()
                for est_step, est_prcss in est_process_flow.items():
                    ax[1].plot(est_prcss['df'][px], est_prcss['df'][py], '.', ms=0.5,
                               label="Pred., Step {}: {} s, {}".format(step + est_step + 1,
                                                                       int(np.round(est_prcss['time'], 1)),
                                                                       est_prcss['recipe']),
                               )

                    z_amplitude = est_prcss['z_min']

                # ------------------------------------------------------------------------------------------------------
                # UNDER CONSTRUCTION
                if include_target:

                    # get target profile for this design
                    gcwf = gcw.get_feature(fid, step)
                    target_radius = gcwf.target_radius
                    mdft = gcwf.mdft.copy()

                    """ Conditional 'units' errors... the result of a non-resized target profile. """
                    if mdft['r'].max() < dfds[px].max() / 10:
                        mdft['r'] = mdft['r'] * target_radius / 2
                    if mdft['z'].abs().max() < np.abs(z_amplitude) / 2:  # dfds[py].abs().max() / 2.5:
                        mdft['z'] = mdft['z'] / mdft['z'].abs().max() * np.abs(z_amplitude)  # norm then scale amplitude

                    ax[1].plot(mdft['r'], mdft['z'], linewidth=0.75, linestyle='dotted', color='k', label="Target")

                # ------------------------------------------------------------------------------------------------------

                ax[1].set_ylabel(r'$z \: (\mu m)$')
                ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

            # set x-labels
            axs[-1][0].set_xlabel(r'$r \: (\mu m)$')
            axs[-1][1].set_xlabel(r'$r \: (\mu m)$')

            plt.tight_layout()
            if save_fig:
                plt.savefig(join(gcw.path_results, 'figs',
                                 'est-process-and-profile_px-{}_py-{}_fid-{}_{}'.format(px, py, fid,
                                                                                        etch_recipe_Si) + save_type))
            else:
                plt.show()
            plt.close()


def compare_exposure_function_plots(gcp, path_save=None, save_type='.png'):
    num_points = 2 ** 8

    fig, ax = plt.subplots()

    for flbl, gpf in gcp.features.items():
        if isinstance(gpf, ProcessFeature):
            depth_space = np.linspace(gpf.exposure_func['depth_limits'][0],
                                      gpf.exposure_func['depth_limits'][1], num_points)
            f_dose = gpf.exposure_func['dose_func'](depth_space, *gpf.exposure_func['dose_popt'])
            ax.plot(depth_space, f_dose, linewidth=1, linestyle='-',
                    label='{}({}, {})'.format(flbl, gpf.dose, gpf.focus))

    ax.set_ylabel(r'$I_{o} \: (mJ)$')
    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.legend(title=r'$f_{ID}$' + '(Dose, Focus)')
    ax.set_title('Step {}, Post-{}'.format(gcp.step, gcp.process_type))

    plt.tight_layout()

    if path_save is not None:
        plt.savefig(join(path_save, 'figs', 'dose-from-depth-collection_step{}'.format(gcp.step) + save_type))
    else:
        plt.show()
    plt.close()


# ------------------------------------------------------------------------------------------------------------------
# PLOTTING FUNCTIONS - GraycartProcess


def plot_process_feature_profilometry(graycart_process, save_fig=False, save_type='.png'):
    """
    Plot all ProcessFeatures on the same figure.

    :param graycart_process:
    :return:
    """

    fig, ax = plt.subplots(
        figsize=(size_x_inches * 1.5, size_y_inches * 1),
        facecolor='white',
    )

    for f_lbl, pf in graycart_process.features.items():
        if not isinstance(pf, ProcessFeature):
            continue

        if 'z_surf' in pf.dfpk.columns:
            p1, = ax.plot(pf.dfpk.r, pf.dfpk.z_surf, '.', ms=1, label='{}, {}'.format(pf.dose, pf.focus))
            ax.fill_between(pf.dfpk.r, y1=pf.dfpk.z_surf, y2=0, where=pf.dfpk.z_surf > 0, ec='none', fc=p1.get_color(),
                            alpha=0.25)
        else:
            ax.plot(pf.dfpk.r, pf.dfpk.z, '.', ms=1, label='{}, {}'.format(pf.dose, pf.focus))

    ax.set_xlabel(r'$r \: (\mu m)$')
    ax.set_ylabel(r'$z \: (\mu m)$')

    ax.set_title(graycart_process.descriptor)
    ax.legend(title='Dose, Focus', loc='upper left', bbox_to_anchor=(1, 1), markerscale=3)

    plt.tight_layout()
    if save_fig:
        save_path = join(graycart_process.ppath, 'results')
        if not isdir(save_path):
            makedirs(save_path)
        plt.savefig(join(save_path, graycart_process.subpath + '_multi' + save_type))
    else:
        plt.show()
    plt.close()


def plot_process_feature_fit_profilometry(graycart_process, include_vhlines=True, save_fig=False, save_type='.png'):
    """
    Plot a figure for each ProcessFigure, including:
        * scan profile
        * function fitted to peak

    :param save_type:
    :param save_fig:
    :param include_vhlines:
    :param graycart_process:
    :return:
    """

    num_figures = graycart_process.num_process_features

    fig, ax = plt.subplots(nrows=num_figures, sharex=True,
                           figsize=(size_x_inches, size_y_inches * num_figures * 0.5),
                           facecolor='white',
                           )

    if not isinstance(ax, (list, np.ndarray)):
        ax = [ax]

    if graycart_process.num_process_features == graycart_process.num_features:
        ax_id = [pf.fid for pf in graycart_process.features.values()]
    else:
        ax_id = np.arange(len(ax))

    j = 0
    for f_lbl, pf in graycart_process.features.items():
        if not isinstance(pf, ProcessFeature):
            continue
        i = ax_id[j]

        # data
        ax[i].plot(pf.dfpk.index - pf.peak_properties['pk_idx'], -pf.dfpk['z'],
                   '.', ms=1, color='gray', alpha=0.35,
                   )

        # fit
        ax[i].plot(pf.peak_properties['res_x'] - pf.peak_properties['pk_idx'], pf.peak_properties['res_z'],
                   linewidth=0.5, color='b')

        # vertical and horizontal lines from scipy.find_peaks()
        if include_vhlines:
            ax[i].plot(0, pf.peak_properties['fit_z'][pf.peak_properties['fit_pk_idx']],
                       'o', ms=1.5, color='k')

            ax[i].vlines(x=0,
                         ymin=pf.peak_properties['fit_z'][pf.peak_properties['fit_pk_idx']] - pf.peak_properties[
                             "prominences"],
                         ymax=pf.peak_properties['fit_z'][pf.peak_properties['fit_pk_idx']],
                         color="r", linewidth=0.5,
                         label='{}, {}'.format(np.round(pf.peak_properties['pk_h'], 1),
                                               int(np.round(pf.peak_properties['pk_r'], -1))),
                         )

            ax[i].hlines(y=pf.peak_properties["width_heights"],
                         xmin=pf.peak_properties["left_ips"] - pf.peak_properties['pk_idx'],
                         xmax=pf.peak_properties["right_ips"] - pf.peak_properties['pk_idx'],
                         color="r", linewidth=0.5)

        ax[i].set_ylabel(r'$z \: (\mu m)$')
        ax[i].set_title(f_lbl + r'$(x_{c}=$' + str(int(np.round(pf.peak_properties["pk_xc"], -2))) + r'$ \: \mu m)$')
        ax[i].legend(title=r'$H, r$', loc='upper left', bbox_to_anchor=(1, 1))

        j += 1

    ax[-1].set_xlabel(r'$i \: (\#)$')

    plt.tight_layout()
    if save_fig:
        save_path = join(graycart_process.ppath, 'results')
        if not isdir(save_path):
            makedirs(save_path)
        plt.savefig(join(save_path, graycart_process.subpath + '_fit-peaks' + save_type))
    else:
        plt.show()
    plt.close()


def plot_tilt_correct_array(x, y, num_points, fit_func, popt, rmse, r_squared, save_id,
                            save_path=None, save_type='.png'):
    # x-y displacements between measurement range
    tilt_y = np.mean(y[-num_points:]) - np.mean(y[:num_points])
    tilt_x = np.mean(x[-num_points:]) - np.mean(x[:num_points])
    tilt_deg = np.rad2deg(np.arcsin(tilt_y / tilt_x))

    # get measurement range array
    x_edges = np.hstack([x[:num_points], x[-num_points:]])
    y_edges = np.hstack([y[:num_points], y[-num_points:]])

    # fit a function
    y_fit = fit_func(x, *popt)
    y_corr = y - fit_func(x, *popt)

    # plot
    fig, ax = plt.subplots(figsize=(size_x_inches * 1.25, size_y_inches * 0.875))

    ax.plot(x, y, color='b', alpha=0.5, label=r'$\Delta y=$' + '{}'.format(np.round(tilt_y, 2)))
    ax.scatter(x_edges, y_edges, s=3, color='yellow', alpha=0.125, label='Fit(x, y, n={})'.format(num_points))
    ax.plot(x, y_fit, color='k', linewidth=0.5, label=r'$y_{fit}$')
    ax.plot(x, y_corr, color='r', linewidth=0.5, alpha=0.5, label=r'$y_{corr}$')

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(alpha=0.125)

    ax.set_title('{}: rmse={}, r sq={}'.format(save_id, np.round(rmse, 3), np.round(r_squared, 4)))

    plt.tight_layout()
    if save_path:
        save_path = join(save_path, 'results')
        if not isdir(save_path):
            makedirs(save_path)
        plt.savefig(join(save_path, save_id + '_tilt-correction' + save_type))

    if r_squared < 0.9 or rmse > 0.25:
        plt.show()

    plt.close()


# ------------------------------------------------------------------------------------------------------------------
# PLOTTING FUNCTIONS - GraycartFeature

def plot_exposure_profile(gcf, path_save=None, save_type='.png'):
    # get data
    df = gcf.dfe
    did = gcf.did
    dose = gcf.dose
    feature_label = gcf.label

    # plot
    fig, ax = plt.subplots(figsize=(size_x_inches * 1.25, size_y_inches * 0.75), facecolor='white')

    # plot layers
    ax.plot(df.r, df.l, color='k',
            label=r'$l_{min}=$' + ' {}'.format(int(df.l.min())) + '\n' +
                  r'$l_{max}=$' + ' {}'.format(int(df.l.max())),
            )
    ax.set_xlabel(r'$r \: (\mu m)$')
    ax.set_ylabel('Design Layer (8-bit)')
    ax.set_ylim([0, 256])
    ax.set_yticks(np.arange(0, 256, 50))

    ax.grid(alpha=0.125)
    ax.legend(loc='lower left')
    ax.set_title('Feature: {}'.format(feature_label))

    # plot - exposure profile
    axr = ax.twinx()
    axr.plot(df.r, df.exposure_dose, linestyle='--', color='r',
             label=r'$I_{o, min}=$' + ' {}'.format(int(df.exposure_dose.min())) + '\n' +
                   r'$I_{o, max}=$' + ' {}'.format(int(df.exposure_dose.max()))
             )

    axr.set_ylabel('Exposure Dose (mJ)', color='r')
    axr.set_ylim([0, dose])
    axr.set_yticks(np.arange(0, dose + 1, 50))
    axr.legend(loc='upper right')

    # save
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, 'exposure-profile_did-{}_{}'.format(did, feature_label) + save_type))
    else:
        plt.show()
    plt.close()


def plot_exposure_profile_and_design_layers(gpf, path_save=None, save_type='.png',
                                            thickness_PR=None):
    # plot
    fig, ax = plt.subplots(figsize=(size_x_inches * 1.25, size_y_inches * 1.0625))

    ax.plot(gpf.dft.r, gpf.dft.z, color='k', label='Target')
    ax.plot(gpf.fold_dfpk.r, gpf.fold_dfpk.z, 'o', ms=0.5, color='gray', alpha=0.25, label='Measured')
    ax.set_xlabel(r'$r \: (\mu m)$')
    ax.set_ylabel(r'$z \: (\mu m)$')

    if thickness_PR is not None:
        ax.axhline(-thickness_PR, linestyle='--', color='gray', linewidth=0.5, label='Substrate')
        ax.set_ylim(top=0.25, bottom=-thickness_PR -0.25)
    ax.legend(loc='lower left', title='Profile', bbox_to_anchor=(0, 1.01), ncols=3,
              fontsize='small', title_fontsize='small', handlelength=1.0, handletextpad=0.3, columnspacing=0.75)

    axr = ax.twinx()
    axr.plot(gpf.dfd.r, gpf.dfd.l, linewidth=0.75, color='b', label='Original')
    axr.plot(gpf.dft.r, gpf.dft.l, linewidth=0.75, color='r', label='Corrected')
    axr.set_ylabel('Layer', color='r')
    axr.set_ylim(top=265, bottom=-10)
    axr.legend(loc='lower right', title='Grayscale Mask',  bbox_to_anchor=(1, 1.01), ncols=3,
               fontsize='small', title_fontsize='small', handlelength=1.0, handletextpad=0.3, columnspacing=0.75)

    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save,
                         'exposure-profile-and-design_step{}-{}_{}_fid{}'.format(gpf.step, gpf.process_type, gpf.label,
                                                                                 gpf.fid) + save_type),
                    )
    else:
        plt.show()
    plt.close()


def plot_overlay_feature_and_exposure_profiles(gcw, step, did, path_save=None, save_type='.png'):
    gcp = gcw.processes[step]

    fig, ax = plt.subplots(figsize=(size_x_inches * 1.5, size_y_inches))
    axr = ax.twinx()

    exposure_doses = []
    for gcf in gcp.features.values():
        if isinstance(gcf, ProcessFeature):
            if gcf.did == did:
                p1, = ax.plot(gcf.dfpk.r, gcf.dfpk.z, label='({}, {})'.format(gcf.dose, gcf.focus))
                axr.plot(gcf.mdfe.r, gcf.mdfe.exposure_dose, linestyle='--', color=p1.get_color())
                exposure_doses.append(gcf.dose)

    ax.set_xlabel(r'$r \: (\mu m)$')
    ax.set_ylabel(r'$z \: (\mu m)$')

    ax.grid(alpha=0.125)
    ax.legend(loc='upper left', bbox_to_anchor=(1.125, 1), title=r'$(I_{o}, f)$')

    axr.set_ylabel('Exposure Dose (mJ)', color='r')
    if len(exposure_doses) > 0:
        axr.set_ylim([0, np.max(exposure_doses)])
        axr.set_yticks(np.arange(0, np.max(exposure_doses) + 1, 50))

    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, 'overlay-profiles_step{}_did-{}'.format(step, did) + save_type))
    else:
        plt.show()
    plt.close()


def plot_exposure_dose_depth_relationship(gpf, path_save=None, save_type='.png'):
    """
    plotting.plot_exposure_dose_depth_relationship(self)
    """
    num_points = 2 ** 8

    fdf = gpf.fold_dfpk  # 'exposure profile'
    dfe = gpf.dfe  # 'dose profile'
    dfmap = gpf.exposure_func['dfmap']

    dose_space = np.linspace(gpf.exposure_func['dose_limits'][0], gpf.exposure_func['dose_limits'][1], num_points)
    f_depth = gpf.exposure_func['depth_func'](dose_space, *gpf.exposure_func['depth_popt'])

    depth_space = np.linspace(gpf.exposure_func['depth_limits'][0], gpf.exposure_func['depth_limits'][1], num_points)
    f_dose = gpf.exposure_func['dose_func'](depth_space, *gpf.exposure_func['dose_popt'])

    # ---

    # plotting
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(size_x_inches, size_y_inches * 1.25))

    # plot 1: overlay exposure and dose profiles (x-axis = microns)
    ax1.plot(fdf.r, fdf.z, '.', ms=0.5, color='gray', label='Data')
    ax1.plot(dfmap.profile_r, dfmap.z, linewidth=0.5, color='r', label='Interp.')
    ax1.axhspan(ymin=depth_space.min(), ymax=depth_space.max(), xmin=0, xmax=1, color='gray', ec=None, alpha=0.125)

    ax1.set_xlabel(r'$r \: (\mu m)$')
    ax1.set_ylabel(r'$z \: (\mu m)$')
    ax1.legend(loc='center right')
    ax1.set_title('Step{}: {} Post-{}'.format(gpf.step, gpf.label, gpf.process_type))

    ax1r = ax1.twinx()
    ax1r.plot(dfe.r, dfe.exposure_dose, linewidth=0.5, color='b', label='exposure')
    ax1r.set_ylabel('dose (mJ)', color='b')

    # plot 2: we want DOSE as a function of z
    ax2.plot(dfmap.z, dfmap.exposure_dose, 'o', ms=0.75, color='gray', alpha=0.5, label='Mapping', zorder=3.2)
    ax2.plot(depth_space, f_dose, linewidth=1, linestyle='-', color='k', label='Fit', zorder=3.1)
    ax2.axvspan(xmin=depth_space.min(), xmax=depth_space.max(), ymin=0, ymax=1, color='gray', ec=None, alpha=0.125)

    ax2.set_ylabel(r'$I_{o} \: (mJ)$')
    ax2.set_xlabel(r'$z \: (\mu m)$')
    ax2.set_xlim(right=0)
    ax2.legend()

    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, 'char-dose-depth_step{}-{}_{}_fid{}'.format(gpf.step, gpf.process_type, gpf.label,
                                                                                gpf.fid) + save_type),
                    )
    else:
        plt.show()
    plt.close()


def plot_exposure_functions(gpf, path_save=None, save_type='.png'):
    """
    plotting.plot_exposure_functions(gpf)
    """

    dose_space = np.linspace(gpf.exposure_func['dose_limits'][0],
                             gpf.exposure_func['dose_limits'][1],
                             gpf.bit_resolution,
                             )
    depth_popt = gpf.exposure_func['depth_popt']
    f_depth = gpf.exposure_func['depth_func'](dose_space, *depth_popt)
    depth_lbl = r'$ax^4 + bx^3 + cx^2 + dx + e$' + '\n' + \
                '({}, {}, {}, {}, {})'.format(np.round(depth_popt[0], 4),
                                              np.round(depth_popt[1], 4),
                                              np.round(depth_popt[2], 4),
                                              np.round(depth_popt[3], 4),
                                              np.round(depth_popt[4], 2),
                                              )

    depth_space = np.linspace(gpf.exposure_func['depth_limits'][0],
                              gpf.exposure_func['depth_limits'][1],
                              gpf.bit_resolution,
                              )
    dose_popt = gpf.exposure_func['dose_popt']
    f_dose = gpf.exposure_func['dose_func'](depth_space, *dose_popt)
    dose_lbl = r'$ax^4 + bx^3 + cx^2 + dx + e$' + '\n' + \
               '({}, {}, {}, {}, {})'.format(np.round(dose_popt[0], 2),
                                             np.round(dose_popt[1], 2),
                                             np.round(dose_popt[2], 2),
                                             np.round(dose_popt[3], 2),
                                             np.round(dose_popt[4], 1),
                                             )

    # ---

    # plotting
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(size_x_inches, size_y_inches * 1.25))

    # plot 1:
    ax1.plot(dose_space, f_depth, label=depth_lbl)
    ax1.set_xlabel(r'$I_{o} \: (mJ)$')
    ax1.set_xlim(left=0, right=gpf.dose)
    ax1.set_ylabel(r'$z \: (\mu m)$')
    ax1.legend()

    # plot 2: we want DOSE as a function of z
    ax2.plot(depth_space, f_dose, label=dose_lbl)
    ax2.set_ylabel(r'$I_{o} \: (mJ)$')
    ax2.set_xlabel(r'$z \: (\mu m)$')
    ax2.set_xlim(right=0)
    ax2.legend()

    plt.suptitle('Step{}: {} Post-{}'.format(gpf.step, gpf.label, gpf.process_type))
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, 'funcs-dose-depth_step{}-{}_{}_fid{}'.format(gpf.step, gpf.process_type, gpf.label,
                                                                                 gpf.fid) + save_type),
                    )
    else:
        plt.show()
    plt.close()


def plot_profile_to_target_error(gcf, path_save=None, save_type='.png'):
    dft = gcf.mdft.copy()
    dfpk = gcf.dfpk.copy()

    fig, ax = plt.subplots(figsize=(size_x_inches * 1.25, size_y_inches * 0.75))

    ax.plot(dft.r, dft.z, 'r-', alpha=1, label='Target')
    ax.plot(dfpk.r, dfpk.z, linestyle='--', color='b', label='Profile')
    ax.set_xlabel(r'$r \: (\mu m)$')
    ax.set_ylabel(r'$z \: (\mu m)$')
    ax.legend()

    ax.set_title(r'$\sigma_{z} = $' + ' {} '.format(np.round(gcf.target_rmse, 1)) + r'$\mu m,$' + ' ' +
                 '(' + r'$\sigma_{z}/h = $' + ' {}\%'.format(np.round(gcf.target_rmse_percent_depth, 1)) + ')',
                 )
    plt.suptitle(r'$V = $' + ' {} nL, '.format(int(np.round(gcf.volume, 0))) +
                 r'$\frac {V - V_{target}} {V_{target}}= $' +
                 ' {}\%'.format(np.round(gcf.target_volume_error / gcf.target_volume * 100, 1)))

    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save,
                         'profile-accuracy_step{}_{}_fid{}'.format(gcf.step, gcf.label, gcf.fid) + save_type),
                    )
    else:
        plt.show()
    plt.close()


# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 3D PLOTTING FUNCTIONS

def plot_3d_profile(df, px, py, pz, aspect_ratio_z, view_height, view_rotation, path_save):
    """

    :param df: dfpk3d
    :param px: 'rx'
    :param py: 'ry'
    :param pz: 'z'
    :param dxy: 5
    :param aspect_ratio_z: 1
    :param view_height: 35
    :param view_rotation: 120
    :return:
    """

    # 0. setup the plot
    ax = plt.figure().add_subplot(projection='3d')
    cmap = plt.cm.Spectral_r
    xlim = (-2.5, 2.5)
    ylim = (-2.5, 2.5)

    # 1. pre-process data
    df[px] = df[px] * 1e-3
    df[py] = df[py] * 1e-3
    df[pz] = df[pz] * -1

    # 2. filter data
    df = df[(df[px] > np.min(xlim)) & (df[px] < np.max(xlim))]
    df = df[(df[py] > np.min(ylim)) & (df[py] < np.max(ylim))]

    # 3. get data
    x = df[px].to_numpy()
    y = df[py].to_numpy()
    z = df[pz].to_numpy()
    rots = df['rot'].unique()
    num_rots = len(rots)

    # 4. get "flat" data (i.e., profiles for 'rot' == 0 and 90)
    fx = df[df['rot'].abs() < 5][px].to_numpy()
    fxz = df[df['rot'].abs() < 5][pz].to_numpy()

    fy = df[np.abs(df['rot'] - 90) < 5][py].to_numpy()
    fyz = df[np.abs(df['rot'] - 90) < 5][pz].to_numpy()

    # ---

    # 4. Create x- and y-profiles of 3D scan
    fx0 = np.ones_like(fx) * np.min(xlim)
    fy0 = np.ones_like(fy) * np.min(ylim)

    # 5. Create a custom triangulation.
    triang = tri.Triangulation(x, y)

    # ---

    # plot

    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(x) / 2 * aspect_ratio_z))

    ax.plot_trisurf(triang, z, cmap=cmap)

    ax.plot(fx, fx0, fxz, color='k')
    ax.plot(fy0, fy, fyz, color='k')

    ax.set(xlim=xlim, ylim=ylim)

    ax.view_init(view_height, view_rotation)  # height, rotation

    ax.set_xlabel(r'$X (mm)$')
    ax.set_ylabel(r'$Y (mm)$')
    ax.set_zlabel(r'$-Z \: (\mu m)$')
    # ax.set_title('angles = ({})'.format(rots))

    # plt.tight_layout()

    if path_save is not None:
        plt.savefig(path_save, dpi=450)
    else:
        plt.show()

    plt.close()

# ---