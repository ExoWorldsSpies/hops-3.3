import numpy as np
from photutils.aperture import CircularAperture, aperture_photometry


def pixel_target_position(ref_x, ref_y, ref_u, r, u):
    x = ref_x + r*np.cos(ref_u + u)
    y = ref_y + r*np.sin(ref_u + u)
    return x, y


def measure_psf_photometry(norm, x_std, y_std):
    gauss_flux = 2*np.pi*norm*x_std*y_std
    gauss_flux_error = np.sqrt(np.abs(gauss_flux))
    return gauss_flux, gauss_flux_error


def measure_aperture_photometry(total_app_flux, sky_flux, sky_flux_unc, camera_gain):
    aperture_flux = total_app_flux - sky_flux
    aperture_flux_error = np.sqrt(np.abs(aperture_flux)/camera_gain + 2*(sky_flux_unc**2))
    return aperture_flux, aperture_flux_error


def compute_geometric_center(fits_data, star_x, star_y, psf, aperture_radius):
    x1 = int(star_x - 3*psf)
    x2 = x1 + int(6*psf) + 1
    y1 = int(star_y - 3*psf)
    y2 = y1 + int(6*psf) + 1
    area = fits_data[y1:y2, x1:x2]
    area_x, area_y = np.meshgrid(np.arange(x1, x2) + 0.5, np.arange(y1, y2) + 0.5)
    x_mean = np.sum(area*area_x)/np.sum(area)
    y_mean = np.sum(area*area_y)/np.sum(area)
    total_app_flux = aperture_photometry(fits_data, CircularAperture(np.array([x_mean - 0.5, y_mean - 0.5]), aperture_radius))['aperture_sum'][0]
    return x_mean, y_mean, total_app_flux


def compute_faint_target(fits_data, x, y, aperture_radius, mean, std):
    x_mean, y_mean = x, y
    floor = mean
    x_std = std
    y_std = std
    total_app_flux = aperture_photometry(fits_data, CircularAperture(np.array([x_mean - 0.5, y_mean - 0.5]), aperture_radius))['aperture_sum'][0]
    sky_flux = np.pi*(aperture_radius**2)*mean
    sky_flux_unc = np.sqrt(np.pi*(aperture_radius**2))*std
    norm = (total_app_flux - sky_flux)/(2*np.pi*x_std*y_std)
    return x_mean, y_mean, norm, floor, x_std, y_std, total_app_flux, sky_flux, sky_flux_unc


def relative_light_curve(gauss_flux, aperture_flux, gauss_flux_error, aperture_flux_error, active_mask):
    n_targets = gauss_flux.shape[0]
    gauss_lc = np.full_like(gauss_flux, np.nan)
    gauss_lc_error = np.full_like(gauss_flux, np.nan)
    aperture_lc = np.full_like(aperture_flux, np.nan)
    aperture_lc_error = np.full_like(aperture_flux, np.nan)
    gauss_comp_sum = np.sum(gauss_flux[active_mask][1:], axis=0)    ### sum for all active comparison stars
    aperture_comp_sum = np.sum(aperture_flux[active_mask][1:], axis=0)

    for target in range(n_targets):
        if target == 0:
            g_lc = gauss_flux[0]/gauss_comp_sum
            a_lc = aperture_flux[0]/aperture_comp_sum
            g_lc_err = np.sqrt((gauss_flux_error[0]/gauss_flux[0])**2 + (np.sqrt(np.sum(gauss_flux_error[active_mask][1:]**2, axis=0))/gauss_comp_sum)**2)*g_lc
            a_lc_err = np.sqrt((aperture_flux_error[0]/aperture_flux[0])**2 + (np.sqrt(np.sum(aperture_flux_error[active_mask][1:]**2, axis=0))/aperture_comp_sum)**2)*a_lc
        elif target in active_mask[0]:
            g_others = gauss_comp_sum - gauss_flux[target]  ## compare against all other active comparisons
            a_others = aperture_comp_sum - aperture_flux[target]
            g_lc = gauss_flux[target]/g_others
            a_lc = aperture_flux[target]/a_others
            g_lc_err = np.full_like(g_lc, np.nan)
            a_lc_err = np.full_like(a_lc, np.nan)
        else:
            continue
        gauss_lc[target] = g_lc
        gauss_lc_error[target] = g_lc_err
        aperture_lc[target] = a_lc
        aperture_lc_error[target] = a_lc_err
    return gauss_lc, gauss_lc_error, aperture_lc, aperture_lc_error


