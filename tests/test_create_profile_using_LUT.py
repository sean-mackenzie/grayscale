import os
from os.path import join
import numpy as np
import pandas as pd
from scipy.special import erf

import gdspy

import matplotlib.pyplot as plt


def get_erf_profile(diameter, depth, num_points, x0, diameter_flat):
    """ diameter=4e-3, depth=65e-6, num_points=200, x0=1.5, diameter_flat=1e-3 """
    if isinstance(x0, (list, np.ndarray)):
        x1, x2 = x0[0], x0[1]
    else:
        x1, x2 = x0, x0

    erf_x = np.linspace(-x1, x2, num_points)
    erf_y = erf(erf_x)

    norm_erf_x = (erf_x - erf_x[0]) / (x1 + x2)
    norm_erf_y = (erf_y - erf_y[0])
    norm_erf_y = norm_erf_y / -norm_erf_y[-1]

    profile_x = norm_erf_x * diameter / 2
    profile_z = norm_erf_y * depth

    profile_x = profile_x * (diameter - diameter_flat) / diameter

    return profile_x, profile_z


def fit_erf_profile_norm(diameter, depth, num_points, x0, diameter_flat):
    """
    px, py, pyf, p12 = fit_erf_profile_norm(diameter, depth, num_points, x0, diameter_flat)

    """
    x1, x2 = x0[0], x0[1]
    ddx = diameter / 2 / num_points

    erf_x = np.linspace(-x1, x2, num_points)
    erf_y = erf(erf_x)

    norm_erf_x = (erf_x - erf_x[0]) / (x1 + x2)
    # norm_erf_x = (norm_erf_x - 1) * (diameter - diameter_flat) / diameter + 1
    norm_erf_x = norm_erf_x * (diameter - diameter_flat) / diameter

    norm_erf_y = (erf_y - erf_y[0])
    norm_erf_y = norm_erf_y / norm_erf_y[-1]
    norm_erf_y = np.flip(norm_erf_y)

    # -
    px = norm_erf_x
    py = norm_erf_y

    p12 = np.poly1d(np.polyfit(px, py, 12))
    pyf = p12(px)

    fit = (px.copy(), pyf, p12)

    if diameter_flat != 0:
        pxflat_outer = np.linspace(px[-1] + ddx, 1, int(len(px) * diameter_flat / diameter))
        pyflat_outer = np.zeros_like(pxflat_outer)

        px = np.hstack([px, pxflat_outer])
        py = np.hstack([py, pyflat_outer])

    return px, py, fit


def fit_flat_profile_norm(diameter, depth, num_points, x0, diameter_flat):
    """
    nfx, nfy, _, nf1 = fit_flat_profile_norm(diameter, depth, num_points, x0, diameter_flat)

    Note: x0 serves no purpose in this function.
    """
    x_intercept = (diameter - diameter_flat) / diameter
    ddx = diameter / 2 / num_points

    px = np.linspace(0, x_intercept, num_points)
    py = np.linspace(1, 0, num_points)

    def p1(x):
        return 1 / x_intercept * x + 1

    fit = (px.copy(), py.copy(), p1)

    if diameter_flat != 0:
        pxflat_outer = np.linspace(px[-1] + ddx, 1, int(len(px) * diameter_flat / diameter))
        pyflat_outer = np.zeros_like(pxflat_outer)

        px = np.hstack([px, pxflat_outer])
        py = np.hstack([py, pyflat_outer])

    return px, py, fit



"""
CREATE LINEAR RAMP INTENSITY OF RECTANGLES
"""

def create_linear_ramp_rectangles():
    # rectangles
    rints = np.arange(0, 256, 1, dtype=int)
    rw, rh = 10, 250
    rect_x0, rect_y0 = 0, 0

    # tick marks
    tint = 100
    tw, th = rw, 40
    tick_gap = 10
    ticks_every = 25
    tick_x0, tick_y0 = rect_x0, rect_y0 + rh + tick_gap
    lyr_tint = {"layer": tint, "datatype": 0}

    # labels
    lint = 100
    lbl_gap = 25
    ls = 25
    lbl_x0, lbl_y0 = tick_x0, tick_y0 + th + lbl_gap
    lyr_lbl = {"layer": lint, "datatype": 0}

    # ---

    # gdspy

    # The GDSII file is called a library, which contains multiple cells.
    lib = gdspy.GdsLibrary()

    # Geometry must be placed in cells.
    cell_name = '21'
    cell = lib.new_cell(cell_name)

    # ---

    rect_xi, rect_yi = rect_x0, rect_y0
    tick_xi, tick_yi = tick_x0, tick_y0
    lbl_xi, lbl_yi = lbl_x0, lbl_y0

    data = []
    for i, rint in enumerate(rints):
        rect_xi = rect_x0 + i * rw
        tick_xi = tick_x0 + i * rw
        lbl_xi = lbl_x0 + i * rw

        lyr_i = {"layer": rint, "datatype": 0}
        rect = gdspy.Rectangle((rect_xi, rect_yi), (rect_xi + rw, rect_yi + rh), **lyr_i)
        cell.add(rect)

        data.append([i, rint, rect_xi, rect_xi + rw])

        if rint % ticks_every == 0:
            tick = gdspy.Rectangle((tick_xi, tick_yi), (tick_xi + tw, tick_yi + th), **lyr_tint)
            cell.add(tick)

            htext = gdspy.Text(str(i), ls, (lbl_xi, lbl_yi), lyr_lbl)
            cell.add(htext)

    # add a rectangle at very end, to know exactly where ended
    rect = gdspy.Rectangle((rect_xi + rw, rect_yi), (rect_xi + rw * 2, rect_yi + rh), **lyr_tint)
    cell.add(rect)
    data.append([i + 1, tint, rect_xi + rw, rect_xi + rw * 2])

    tick = gdspy.Rectangle((tick_xi + rw, tick_yi), (tick_xi + rw + tw, tick_yi + th), **lyr_tint)
    cell.add(tick)

    # -

    # add a rectangle before rint == 0, to know exactly where started
    rect = gdspy.Rectangle((rect_x0 - rw, rect_yi), (rect_x0, rect_y0 + rh), **lyr_tint)
    cell.add(rect)
    data.append([-1, tint, rect_x0 - rw, rect_x0])

    tick = gdspy.Rectangle((tick_x0 - rw, tick_yi), (tick_x0, tick_y0 + th), **lyr_tint)
    cell.add(tick)

    # ---

    # export to excel
    df = pd.DataFrame(np.array(data), columns=['i', 'lyr', 'xi', 'xf'])
    df = df.sort_values('i')
    df.to_excel('linear_ramp_dXdI_10um_0-255.xlsx', index=False)

    # cell.write_svg('test_example{}.svg'.format(cell_name))

    # Save the library in a file called 'first.gds'.
    # lib.write_gds('linear_ramp_dXdI_10um.gds')

# ---

"""
CREATE LINEAR RAMP INTENSITY OF RINGS (CIRCLES)
"""

def create_linear_ramp_circles():
    # rectangles
    dI = 4
    rints = np.arange(0, 256, dI, dtype=int)

    x0, y0 = 0, 0
    dR = 3 * dI
    R0 = dR / dI * rints[-1] + 2 * dR
    initial_angle = 0
    final_angle = 359.99
    dTheta = 13
    tolerance = 1e-1

    # tick marks
    tint = 125
    tw, th = R0 * 2, 25
    tick_gap = 0
    tick_x0, tick_y0 = -R0, R0 + tick_gap
    lyr_tint = {"layer": tint, "datatype": 0}

    # labels
    lint = 125
    lbl_gap = 25
    ls = 50
    lbl_x0, lbl_y0 = tick_x0, tick_y0 + th + lbl_gap
    lyr_lbl = {"layer": lint, "datatype": 0}
    lbl_text = "Dia{}mm, dr{}um".format(np.round(R0 * 2, 3), dR)

    if dR == 5:
        save_id = 'linear_ramp_dRdI_5um_R2p56mm'
    elif dR == 4:
        save_id = 'linear_ramp_dRdI_4um_R2p05mm'
    elif dR == 3:
        save_id = 'linear_ramp_dRdI_3um_R1p54mm'
    elif dR == 10:
        save_id = 'linear_ramp_dRd2I_10um_R2p56mm'
    else:
        save_id = 'TEST_linear_ramp_dRdI_{}um'.format(dR)

    save_id = 'linear_ramp_dR{}um_dI{}_Tol1e1'.format(int(dR / dI), dI)

    # ---

    # MODIFIERS
    do_gdspy = True
    export_to_excel = True
    save_fig = True

    # gdspy

    # The GDSII file is called a library, which contains multiple cells.
    lib = gdspy.GdsLibrary()
    # Geometry must be placed in cells.
    cell_name = save_id
    cell = lib.new_cell(cell_name)

    data = []
    for i, rint in enumerate(rints):
        circ_ri = R0 - i * dR
        i_angle_i = initial_angle + i * dTheta
        f_angle_i = final_angle + i * dTheta

        data.append([i, rint, circ_ri, circ_ri - dR, i_angle_i, f_angle_i])
        data.append([i, rint, circ_ri - dR, 0, i_angle_i, f_angle_i])

        if do_gdspy:
            lyr_i = {"layer": rint, "datatype": 0}
            circ = gdspy.Round((x0, y0),  # center
                               circ_ri,  # outer radius
                               inner_radius=circ_ri - dR,
                               initial_angle=i_angle_i,
                               final_angle=f_angle_i,
                               tolerance=tolerance,
                               **lyr_i)
            cell.add(circ)

    # add a rectangle at edge to know where circle begins, and label
    tick = gdspy.Rectangle((tick_x0, tick_y0), (tick_x0 + tw, tick_y0 + th), **lyr_tint)
    cell.add(tick)
    tick = gdspy.Rectangle((tick_x0, -R0 - tick_gap - th), (tick_x0 + tw, -R0 - tick_gap), **lyr_tint)
    cell.add(tick)

    htext = gdspy.Text(lbl_text, ls, (lbl_x0, lbl_y0), lyr_lbl)
    cell.add(htext)

    # export to excel
    df = pd.DataFrame(np.array(data), columns=['i', 'lyr', 'ro', 'ri', 'thetai', 'thetaf'])
    if export_to_excel:
        df.to_excel(save_id + '.xlsx', index=False)

    fig, ax = plt.subplots()
    ax.plot(df.ro, df.lyr, lw=0.5)
    ax.set_ylabel('Layer (8-bit)')
    ax.set_xlabel('r (um)')
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if save_fig:
        plt.savefig(save_id + '.png', dpi=300, facecolor='w', bbox_inches='tight')
    plt.show()
    plt.close()

    # Save the library in a file called 'first.gds'.
    lib.write_gds(save_id + '.gds')

    # NOTE: the .svg is almost certainly TOO BIG TO EXPORT, so don't even try
    # a 180 MB .gds will become a 600 MB .svg, which will crash inkscape
    cell.write_svg('{}.svg'.format(save_id[:-4]))

# ---

"""
1. Use a grayscale LUT (z vs. dose) to convert profile to layer (0-255)
"""

# 1. Load LUT, load intensity profile, and plot the data that you have

def load_lut_and_intensity_profile():
    base_dir = '/Users/mackenzie/Desktop/zipper_paper/Fabrication/grayscale/rectangle_w1'
    fn_lut = 'analyses/step1_post-60s-AZ300MIF/800mJ_profile2/800mJ_profile2_step6_FitSurface+Dose.xlsx'
    fp_lut = join(base_dir, fn_lut)

    id_ = '800mJ_profile2'
    exposure_dose = 800  # mJ

    dir_intensity = 'MLA 150'
    fn_intensity = 'linear_ramp_dXdI_10um_0-255.xlsx'
    fp_intensity = join(base_dir, dir_intensity, fn_intensity)

    dfi = pd.read_excel(fp_intensity)
    dfi['dose'] = dfi['lyr'] / 255 * exposure_dose
    dfi = dfi.iloc[1:-1].reset_index(drop=True)
    dfi.head(3)

    zmin, zmax = 0, -7750  # units: nm

    # 1. load coefficients and make polynomial
    df_coeffs = pd.read_excel(fp_lut)
    coeffs = df_coeffs.coeff.to_numpy()
    p5 = np.poly1d(coeffs)

    # 2. fit polynominal to intensity profile
    p1 = np.poly1d(np.polyfit(dfi.xi, dfi.dose, deg=1))

    # 3. interpolate / apply LUT
    fx = np.linspace(zmin, zmax)
    z2x = p5(fx)
    x2dose = p1(z2x)

    # 4. create conversion LUT
    lut5 = np.poly1d(np.polyfit(fx * 1e-3, x2dose, 5))

    # ---

    # plot

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 3.5))

    ax1.plot(fx, p5(fx), label='z-to-x')
    ax1.set_xlabel('Exposure Depth (nm)')
    ax1.set_ylabel('x (um)')
    ax1.legend()
    ax1.grid(alpha=0.25)

    ax2.plot(z2x, p1(z2x), label='x-to-dose')
    ax2.set_xlabel('x (um)')
    ax2.set_ylabel('Exposure Dose (mJ)')
    ax2.legend()
    ax2.grid(alpha=0.25)

    ax3.plot(fx * 1e-3, x2dose, label='z-to-dose')
    ax3.set_xlabel('Exposure Depth (nm)')
    ax3.set_ylabel('Exposure Dose (mJ)')
    ax3.legend()
    ax3.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()

    # ---

    fig, ax3 = plt.subplots()

    ax3.plot(fx * 1e-3, x2dose, 'k-', label='LUT')
    ax3.plot(fx * 1e-3, lut5(fx * 1e-3), 'r--', label='Fit')
    ax3.set_xlabel('Exposure Depth (um)')
    ax3.set_ylabel('Exposure Dose (mJ)')
    ax3.legend()
    ax3.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()

# ---

"""
2. Generate the erf profile that you want
"""

# 2. Generate and plot erf profile

def generate_erf_in_silicon_and_rescale_to_photoresist():
    # 1. Generate erf in silicon
    diameter = 1.5e-3
    depth = 150e-6
    num_points = 32
    x0 = 1.25
    diameter_flat = diameter * 0.08

    px, py = get_erf_profile(diameter, depth, num_points, x0, diameter_flat)

    px = px + diameter_flat / 2
    px = np.flip(px)
    py = py * 1e6

    fig, ax = plt.subplots()
    ax.plot(px, py, '-o')
    plt.show()

    # ---

    # 2. Rescale erf to photoresist
    resist_thickness = 8.125
    depth_min, depth_max = 0, -7
    depth_offset = 0.5

    rx_ = px * 1e6
    ry_ = py / np.min(py) * depth_max - depth_offset

    rx_ = np.append(rx_, [0.25, 0])
    ry_ = np.append(ry_, [ry_[-1], ry_[-1]])

    # plot
    fig, ax = plt.subplots()

    ax.plot(rx_, ry_, '-o', lw=1)
    ax.plot(rx_[:5], ry_[:5], 'r-', lw=3)

    ax.set_ylabel('Depth (um)')
    ax.set_xlabel('Radius (um)')
    ax.set_ylim(top=0, bottom=-resist_thickness)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

# ---

"""
3. Do a "fake" generate_grayscale_mask to confirm that the pattern is correct
"""

def visualize_generate_grayscale_mask_from_lut_and_profile():
    # pre-this
    # dose = lut5(depth)      depth units are microns
    # rx = radial coordinates (meters)
    # ry = depth coordinates (microns)

    # convert
    rx = rx_
    ry = ry_

    # maximum dose (i.e., the dose setting to be used by MLA 150)
    max_dose = 800

    x0, y0 = 0, 0
    initial_angle = 0
    final_angle = 359.99
    num_revolutions = 1
    dTheta = 360 / len(rx) * num_revolutions

    # ---

    # ---

    # ---

    data = []
    for i in range(len(rx) - 1):
        # 1. diameter
        ring_outer_radius = rx[i]
        ring_inner_radius = rx[i + 1]

        # 2a. depth
        ring_depth_avg = np.mean([ry[i], ry[i + 1]])

        # 2b. dose (mJ)
        ring_dose = lut5(ring_depth_avg)

        # 2c. normalize dose and rescale to layer
        ring_dose_norm = ring_dose / max_dose
        ring_lyr = int(np.round(ring_dose_norm * 255))

        # -

        # other details
        i_angle_i = initial_angle + i * dTheta
        f_angle_i = final_angle + i * dTheta
        data.append(
            [i, ring_lyr, ring_dose, ring_depth_avg, ring_outer_radius, ring_inner_radius, i_angle_i, f_angle_i])
        data.append([i, ring_lyr, ring_dose, ring_depth_avg, ring_outer_radius, 0, i_angle_i, f_angle_i])

    # export to excel
    df = pd.DataFrame(np.array(data), columns=['i', 'lyr', 'dose', 'zm', 'ro', 'ri', 'thetai', 'thetaf'])

    # plot
    fig, ax1 = plt.subplots()

    ax1.plot(rx_, ry_, 'k-o', lw=1)
    ax1.set_ylabel('Depth (um)', fontsize=14, color='k')
    ax1.set_xlabel('r (um)', fontsize=14, color='k')
    ax1.set_ylim(top=0, bottom=-resist_thickness)
    ax1.grid(alpha=0.25)
    ax1.set_title('Radius = {}'.format(rx_[0]))

    ax2 = ax1.twinx()
    ax2.plot(df.ro, df.lyr, 'r-o', lw=1)
    ax2.set_ylabel('Layer (8-bit)', fontsize=14, color='r')
    # ax2.set_xlabel('r (um)')
    # ax2.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


# ---

"""
4. Generate grayscale mask using LUT and photoresist profile
"""

def generate_grayscale_mask_from_lut_and_profile():
    """
    diameter = 1.5e-3
    depth = 150e-6
    num_points = 32
    x0 = 1.25
    diameter_flat = diameter * 0.08

    resist_thickness = 8.125
    depth_min, depth_max = 0, -7
    depth_offset = 0.5

    max_dose = 800

    x0, y0 = 0, 0
    initial_angle = 0
    final_angle = 359.99
    num_revolutions = 1
    dTheta = 360 / len(rx) * num_revolutions
    """

    save_id = 'erf_dia1.5mm_x01.25_32lyrs_7.5umDepth_Tol1e1_dose800mJ'
    cell_name = save_id
    tolerance = 1e-1
    export_to_excel = True
    save_fig = True
    save_gds = True
    save_svg = True

    # ---
    R0 = rx[0]
    dR = np.round(rx[1] - rx[0], 1)
    # tick marks
    tint = 100
    tw, th = R0 * 2, 25
    tick_gap = 0
    tick_x0, tick_y0 = -R0, R0 + tick_gap
    lyr_tint = {"layer": tint, "datatype": 0}
    # labels
    lint = 100
    lbl_gap = 25
    ls = 50
    lbl_x0, lbl_y0 = tick_x0, tick_y0 + th + lbl_gap
    lyr_lbl = {"layer": lint, "datatype": 0}
    lbl_text = "Dia{}mm, dr{}um".format(np.round(R0 * 2, 3), dR)
    # ---

    # gdspy
    lib = gdspy.GdsLibrary()
    cell = lib.new_cell(cell_name)

    data = []
    for i in range(len(rx) - 1):
        # 1. diameter
        ring_outer_radius = rx[i]
        ring_inner_radius = rx[i + 1]

        # 2a. depth
        ring_depth_avg = np.mean([ry[i], ry[i + 1]])

        # 2b. dose (mJ)
        ring_dose = lut5(ring_depth_avg)

        # 2c. normalize dose and rescale to layer
        ring_dose_norm = ring_dose / max_dose
        ring_lyr = int(np.round(ring_dose_norm * 255))
        # -
        # -
        # other details
        i_angle_i = initial_angle + i * dTheta
        f_angle_i = final_angle + i * dTheta
        data.append(
            [i, ring_lyr, ring_dose, ring_depth_avg, ring_outer_radius, ring_inner_radius, i_angle_i, f_angle_i])
        data.append([i, ring_lyr, ring_dose, ring_depth_avg, ring_outer_radius, 0, i_angle_i, f_angle_i])
        # -
        # -
        # gdspy
        lyr_i = {"layer": ring_lyr, "datatype": 0}
        circ = gdspy.Round((x0, y0),  # center
                           ring_outer_radius,  # outer radius
                           inner_radius=ring_inner_radius,
                           initial_angle=i_angle_i,
                           final_angle=f_angle_i,
                           tolerance=tolerance,
                           **lyr_i)
        cell.add(circ)

    # add a rectangle at edge to know where circle begins, and label
    tick = gdspy.Rectangle((tick_x0, tick_y0), (tick_x0 + tw, tick_y0 + th), **lyr_tint)
    cell.add(tick)
    tick = gdspy.Rectangle((tick_x0, -R0 - tick_gap - th), (tick_x0 + tw, -R0 - tick_gap), **lyr_tint)
    cell.add(tick)

    htext = gdspy.Text(lbl_text, ls, (lbl_x0, lbl_y0), lyr_lbl)
    cell.add(htext)

    # ---

    # export to excel
    df = pd.DataFrame(np.array(data), columns=['i', 'lyr', 'dose', 'zm', 'ro', 'ri', 'thetai', 'thetaf'])
    if export_to_excel:
        df.to_excel(save_id + '.xlsx', index=False)

    # plot
    fig, ax1 = plt.subplots()

    ax1.plot(rx_, ry_, 'k-o', lw=1)
    ax1.set_ylabel('Depth (um)', fontsize=14, color='k')
    ax1.set_xlabel('r (um)', fontsize=14, color='k')
    ax1.set_ylim(top=0, bottom=-resist_thickness)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(df.ro, df.lyr, 'r-o', lw=1)
    ax2.set_ylabel('Layer (8-bit)', fontsize=14, color='r')

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_id + '.png', dpi=300, facecolor='w', bbox_inches='tight')
    plt.show()
    plt.close()

    if save_gds:
        # Save the library in a file called 'first.gds'.
        lib.write_gds(save_id + '.gds')

    if save_svg:
        # NOTE: the .svg is almost certainly TOO BIG TO EXPORT, so don't even try
        # a 180 MB .gds will become a 600 MB .svg, which will crash inkscape
        cell.write_svg('{}.svg'.format(save_id[:-4]))