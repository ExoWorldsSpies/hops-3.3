
import numpy as np
from hops.hops_tools.image_analysis import cartesian_to_polar


def select_calibration_stars(stars, x_length, y_length, psf, burn_limit, faint_target_mode=False,
                             moving_target_mode=False, max_comparisons=500, min_separation_factor=5.0):
    stars = np.array(stars)
    stars = stars[np.where(stars[:, 2] > 2)]  #### keep for good flux
    stars = stars[np.where(stars[:, 7] > 2)]  ###### keep for good SNR
    center_x = x_length / 2.0
    center_y = y_length / 2.0
    length = np.sqrt(center_x ** 2 + center_y ** 2)  # max dist
    min_flux = np.log10(np.min(stars[:, 2]) - 1)
    max_flux = np.log10(0.5 * burn_limit)
    if faint_target_mode or moving_target_mode:
        weight_distance = 1
        weight_flux = 4
    else:
        weight_distance = 2
        weight_flux = 1

    def _score(star):
        flux_val = star[2]
        log_flux = np.log10(flux_val) if flux_val < 0.5 * burn_limit else min_flux  ##saturated stars to min
        norm_flux = (log_flux - min_flux) / (max_flux - min_flux)
        norm_dist = 1.0 - (np.sqrt((star[0] - center_x) ** 2 + (star[1] - center_y) ** 2) / length)
        return -(weight_flux * norm_flux + weight_distance * norm_dist) / (weight_distance + weight_flux)

    stars_sorted = sorted(stars, key=_score)
    ref_id = 1 if (faint_target_mode or moving_target_mode) else 0
    ref_p = stars_sorted[ref_id]
    ref_star = {"x": ref_p[0], "y": ref_p[1], "flux": ref_p[7]}
    remaining = [s for i, s in enumerate(stars_sorted) if i != ref_id]
    remaining = np.array(remaining)
    min_flux_c = np.log10(np.min(remaining[:, 7]) - 1)
    max_flux_c = np.log10(np.max(remaining[:, 7]))
    w_dist, w_flux = 1, 3

    def _supp_score(star):  ####  to vote if stars are in the expected pos
        norm_flux = (np.log10(star[7]) - min_flux_c) / (max_flux_c - min_flux_c)
        norm_dist = 1.0 - (np.sqrt((star[0] - center_x) ** 2 + (star[1] - center_y) ** 2) / length)
        return -(w_flux * norm_flux + w_dist * norm_dist) / (w_dist + w_flux)

    remaining_sorted = sorted(remaining, key=_supp_score)
    comparisons = []
    for star in remaining_sorted:
        r, u = cartesian_to_polar(star[0], star[1], ref_star["x"], ref_star["y"])
        if r > min_separation_factor * psf:
            comparisons.append([r, u])
        if len(comparisons) >= max_comparisons:
            break
    if faint_target_mode:
        check_num = 2.5
    else:
        check_num = max(9.5, len(comparisons) / 10.0)

    return ref_star["x"], ref_star["y"], 0, comparisons, check_num


def angle_grid(psf, comparisons, rot_tol):
    mid_r = comparisons[len(comparisons) // 2][0]
    ustep = np.arcsin(psf / mid_r)
    small_angles = np.append(np.arange(-rot_tol, rot_tol, ustep),
                             np.arange(-rot_tol, rot_tol, ustep) + np.pi)
    large_angles = np.array([np.pi, 0.0])
    for ff in range(1, int(np.pi / ustep) + 1):
        large_angles = np.append(large_angles,
                                 [np.pi - ff * ustep, np.pi + ff * ustep, 0.0 - ff * ustep, 0.0 + ff * ustep])
    return small_angles, large_angles  # arrays with the candidate rotation angles



def check_alignment_candidate(
        fits_data, candidate_x, candidate_y, candidate_u, comparisons, check_num, centroids_snr, mean, std, psf, log
):
    y_length, x_length = fits_data.shape
    int_psf = int(max(1, round(psf)))
    check_lim = (mean + centroids_snr*std) * ((2*int_psf + 1)**2) #### expected for star detection
    score = 0.0
    for comp in comparisons:
        check_x = int(candidate_x + comp[0]*np.cos(candidate_u + comp[1]))   ##### project the comp star to the expected pos
        check_y = int(candidate_y + comp[0]*np.sin(candidate_u + comp[1]))
        if 0 < check_x < x_length and 0 < check_y < y_length:
            pixel_sum = np.sum(fits_data[check_y - int_psf: check_y + int_psf + 1,
                                         check_x - int_psf: check_x + int_psf + 1])
            score += 1.0 if pixel_sum > check_lim else -1.0  #vote
        else:
            score -= 1.0
        if abs(score) > check_num:
            break

        log('Check ref. star at: ', check_x, check_y, ', Test: ', score)

    return score

# def check_alignment_loop(fits_data, settings, comparisons, check_num, centroids_snr, mean, std, psf):
#     for (x, y, u, star) in settings:
#         score = check_alignment_candidate(fits_data, x, y, u, comparisons, check_num, centroids_snr, mean, std, psf)
#         if score >= check_num:
#             return {"x0": x, "y0": y, "u0": u, "f0": star[-1]}
#     return None


def update_rotation_angle(fits_data, x0, y0, u0, comparisons, psf, max_stars=20):
    y_length, x_length = fits_data.shape
    search_window = int(round(3*psf))
    meshx = np.arange(0, x_length) + 0.5
    meshy = np.arange(0, y_length) + 0.5
    fits_datax, fits_datay = np.meshgrid(meshx, meshy)
    test_cos, test_sin = [], []

    for comp in comparisons:
        check_x = x0 + comp[0]*np.cos(u0 + comp[1])
        check_y = y0 + comp[0]*np.sin(u0 + comp[1])
        if not (0 < check_x < x_length and 0 < check_y < y_length):
            continue
        y_min = int(max(int(check_y) - search_window, 0))
        y_max = int(min(int(check_y) + search_window + 1, y_length))
        x_min = int(max(int(check_x) - search_window, 0))
        x_max = int(min(int(check_x) + search_window + 1, x_length))
        window = fits_data[y_min:y_max, x_min:x_max]
        window_x = fits_datax[y_min:y_max, x_min:x_max]
        window_y = fits_datay[y_min:y_max, x_min:x_max]
        window_sum = np.sum(window)
        if window_sum == 0:
            continue

        centroid_x = np.sum(window*window_x)/window_sum
        centroid_y = np.sum(window*window_y)/window_sum
        _, measured_angle = cartesian_to_polar(centroid_x, centroid_y, x0, y0)
        diff = measured_angle - comp[1]
        if diff < 0:
            diff += 2*np.pi
        test_cos.append(np.cos(diff))
        test_sin.append(np.sin(diff))
        if len(test_cos) >= max_stars:
            break
    if not test_cos:
        return u0   #original rot angles unchanged
    med_cos = np.median(test_cos)
    med_sin = np.median(test_sin)
    u0_update = np.arccos(np.clip(med_cos, -1.0, 1.0))
    if med_sin < 0:
        u0_update = np.pi + (np.pi - u0_update)
    return u0_update


def fov_visibility(all_stars, frame_alignments, x_length, y_length, psf):
    all_stars = np.array(all_stars)
    first_good = next((fa for fa in frame_alignments if fa.get("align_x0")), None)
    if first_good is None:
        raise ValueError("No successfully aligned frame.")
    ref_x0 = first_good["align_x0"]
    ref_y0 = first_good["align_y0"]

    polar_coords = [cartesian_to_polar(star[0], star[1], ref_x0, ref_y0) for star in all_stars]
    in_fov = np.ones(len(polar_coords))   ### start with assuming all stars visible
    for fa in frame_alignments:
        frame_x0 = fa.get("align_x0")
        if not frame_x0:
            continue    ## skip frames that failed alignment
        frame_y0 = fa["align_y0"]
        frame_u0 = fa["align_u0"]
        frame_psf = fa.get("psf", psf)
        margin = 3*frame_psf
        in_fov_single = []
        for r, u in polar_coords:
            cart_x = frame_x0 + r*np.cos(frame_u0 + u)   #### project the star into this frame's coordinate system
            cart_y = frame_y0 + r*np.sin(frame_u0 + u)
            visible = (margin < cart_x < x_length - margin and margin < cart_y < y_length - margin)
            in_fov_single.append(1.0 if visible else 0.0)
        in_fov *= np.array(in_fov_single) ### multiply by 0 to mark any star outside this frame
    in_fov_id = np.where(in_fov)
    margin_ref = 3*psf
    fov_bounds = {"x_min": float(np.min(all_stars[in_fov_id, 0]) - margin_ref), "x_max": float(np.max(all_stars[in_fov_id, 0]) + margin_ref),
                  "y_min": float(np.min(all_stars[in_fov_id, 1]) - margin_ref), "y_max": float(np.max(all_stars[in_fov_id, 1]) + margin_ref)}

    return in_fov, fov_bounds


