
import numpy as np
import exoclock

from scipy.interpolate import interp1d

from .image_analysis import image_burn_limit, image_mean_std, image_psf

def mad_filter(datax, lim=5, iter=3):

    inliers = np.abs(datax) >= 0
    for iteration in range(iter):
        data_median = np.median(datax[inliers])
        data_mad = np.median(np.abs(datax[inliers] - data_median))
        new_inliers = np.abs(datax - data_median) < lim * data_mad
        inliers = new_inliers

    return inliers

def mean_std_from_median_mad(data, samples=None, madfilter=None):

    data = 1.0 * data.flatten()

    if samples:
        data = data[::int(len(data) / samples + 1)]

    if madfilter:
        data =  mad_filter(data, lim=madfilter)

    median = np.median(data)
    dx = data - median
    mad = np.sqrt(np.median(dx * dx))

    return median, mad * 0.6745

def combine_frames(frames, method='median'):
    stack = np.stack(frames, axis=0)
    if method == 'mean':
        return np.nanmean(stack, axis=0)
    return np.nanmedian(stack, axis=0)

def reduce_dark_frame(dark_frame, master_bias, dark_exp_time, bias_exp_time):
    return (dark_frame - master_bias) / (dark_exp_time - bias_exp_time)


def reduce_flat_frame(flat_frame, master_dark, master_bias, flat_exp_time, bias_exp_time):
    return flat_frame - master_bias - (flat_exp_time - bias_exp_time) * master_dark


def compute_master_flat(corrected_flat_frames, method='median', colour_camera_mode=False):

    if len(corrected_flat_frames) == 0:
        return 1.0

    if method == 'mean':
        normalized = [ff / np.nanmean(ff) for ff in corrected_flat_frames]
        master_flat = np.nanmean(np.stack(normalized), axis=0)
    else:
        normalized = [ff / np.nanmedian(ff) for ff in corrected_flat_frames]
        master_flat = np.nanmedian(np.stack(normalized), axis=0)

    if colour_camera_mode:
        master_flat[::2, ::2] = master_flat[::2, ::2] / np.nanmedian(master_flat[::2, ::2])
        master_flat[::2, 1::2] = master_flat[::2, 1::2] / np.nanmedian(master_flat[::2, 1::2])
        master_flat[1::2, ::2] = master_flat[1::2, ::2] / np.nanmedian(master_flat[1::2, ::2])
        master_flat[1::2, 1::2] = master_flat[1::2, 1::2] / np.nanmedian(master_flat[1::2, 1::2])
    else:
        master_flat = master_flat / np.nanmedian(master_flat)

    #     clean master flat

    bp_map = np.zeros_like(master_flat)
    bp_map = np.where(master_flat <= 0, 1, bp_map)
    bp_map = np.where(np.isnan(master_flat), 1, bp_map)
    bp_map = np.where(master_flat == np.inf, 1, bp_map)

    master_flat[np.int_(bp_map)] = 1
    bp_map[:, 0] = 0
    bp_map[:, -1] = 0

    for line in range(len(master_flat)):
        clean_pixs = np.where(bp_map[line] == 0)
        points = clean_pixs[0]
        values = master_flat[line][clean_pixs]
        line_model = interp1d(points, values)
        master_flat[line] = line_model(np.arange(len(master_flat[line])))

    return master_flat

def reduce_science_frame(science_frame, saturation, master_flat, master_dark, master_bias, science_exp_time,
                         bias_exp_time):

    dq_frame = np.where(science_frame == saturation, 1, 0)

    reduced_science_frame = (science_frame - master_bias -
                             (science_exp_time - bias_exp_time) * master_dark) / master_flat

    reduced_science_frame[np.where(np.isnan( reduced_science_frame))] = 0.0

    reduced_science_frame[np.where(dq_frame > 0)] = saturation

    return reduced_science_frame


def bin_frame(data_frame, binning):

    binning = int(binning)

    if binning <= 1:
        return data_frame

    new_frame_ysize = int(len(data_frame)/binning)
    new_frame_xsize = int(len(data_frame[0])/binning)

    new_frame = np.zeros((new_frame_ysize, len(data_frame[0])))

    for xx in range(binning):
        new_frame += data_frame[xx:binning * new_frame_ysize:binning]

    new_frame2 = np.zeros((new_frame_ysize, new_frame_xsize))
    for xx in range(binning):
        new_frame2 += new_frame[:, xx:binning * new_frame_xsize:binning]

    return new_frame2


def bin_and_crop_science_frame(
        fits_data, saturation, crop_x1, crop_x2, crop_y1, crop_y2,crop_edge_pixels, bin_fits
):

    dq_frame = np.where(fits_data == saturation, 1, 0)

    crop_x1 = int(max(0, crop_x1))
    crop_x2 = int(min(crop_x2, len(fits_data[0])))
    crop_y1 = int(max(0, crop_y1))
    crop_y2 = int(min(crop_y2, len(fits_data)))

    if crop_x2 == 0:
        crop_x2 = len(fits_data[0])
    if crop_y2 == 0:
        crop_y2 = len(fits_data)

    if not (np.array([crop_x1, crop_x2, crop_y1, crop_y2]) == np.array([0, len(fits_data[0]), 0, len(fits_data)])).all():
        fits_data = fits_data[crop_y1: crop_y2]
        fits_data = fits_data[:, crop_x1: crop_x2]
        dq_frame = dq_frame[crop_y1: crop_y2]
        dq_frame = dq_frame[:, crop_x1: crop_x2]

    crop_edge_pixels = int(crop_edge_pixels)
    if crop_edge_pixels > 0:
        fits_data = fits_data[crop_edge_pixels: -crop_edge_pixels, crop_edge_pixels: -crop_edge_pixels]
        dq_frame = dq_frame[crop_edge_pixels: -crop_edge_pixels, crop_edge_pixels: -crop_edge_pixels]

    bin_fits = int(bin_fits)
    if bin_fits > 1:
        fits_data = bin_frame(fits_data, bin_fits)
        saturation = saturation * bin_fits * bin_fits
        dq_frame = bin_frame(dq_frame, bin_fits)

    fits_data[np.where(dq_frame > 0)] = saturation

    return fits_data

####################################################################################################
# HOPS-SPECIFIC TASKS

def get_saturation(fits_header, key):
    if key and key in fits_header:
        return fits_header[key]

    else:
        return min(65535, int(2 ** abs(fits_header['BITPIX']) - 1))


def add_image_statistics_to_header(fits_data, fits_header, saturation, log):

    fits_header['BZERO'] = 0
    fits_header['BSCALE'] = 1

    fits_header['NAXIS1'] = len(fits_data[0])
    fits_header['NAXIS2'] = len(fits_data)
    if 'XBINNING' in fits_header:
        fits_header['XBINNING'] = log.get_param('bin_fits') * fits_header['XBINNING']
    else:
        fits_header['XBINNING'] = log.get_param('bin_fits')

    if 'YBINNING' in fits_header:
        fits_header['YBINNING'] = log.get_param('bin_fits') * fits_header['YBINNING']
    else:
        fits_header['XBINNING'] = log.get_param('bin_fits')

    mean, std = image_mean_std(fits_data, samples=10000, mad_filter=5.0)

    bpp = 100 * np.sum(fits_data - mean > 3 * std) / (len(fits_data) * len(fits_data[0]))

    fits_header[log.hops_saturation_key] = saturation * (log.get_param('bin_fits') ** 2)

    psf = image_psf(
        fits_data, fits_header, mean, std, 0.8 * fits_header[log.hops_saturation_key],
        centroids_snr=log.get_param('centroids_snr'),
        stars_snr=log.get_param('stars_snr'),
        psf_guess=log.get_param('psf_guess')
    )

    if np.isnan(psf):
        psf = 10
        skip = True
    else:
        skip = False

    fits_header[log.mean_key]= mean
    fits_header[log.std_key]= std
    fits_header[log.bpp_key]= bpp
    fits_header[log.psf_key]= psf
    fits_header[log.skip_key]= skip

    return fits_header


def add_timing_and_coordinates_to_header(fits_header, log):

    if log.get_param('observation_date_key') == log.get_param('observation_time_key'):
        observation_time = ' '.join(fits_header[log.get_param('observation_date_key')].split('T'))
    else:
        observation_time = ' '.join([fits_header[log.get_param('observation_date_key')].split('T')[0],
                                     fits_header[log.get_param('observation_time_key')]])

    exp_time = float(fits_header[log.get_param('exposure_time_key')])

    observation_time = exoclock.Moment(utc=observation_time)
    if log.get_param('time_stamp') == 'exposure start':
        pass
    elif log.get_param('time_stamp') == 'mid-exposure':
        observation_time = exoclock.Moment(jd_utc=observation_time.jd_utc() - 0.5 * exp_time / 24.0 / 3600.0)
    elif log.get_param('time_stamp') == 'exposure end':
        observation_time = exoclock.Moment(jd_utc=observation_time.jd_utc() - exp_time / 24.0 / 3600.0)
    else:
        raise RuntimeError('Not acceptable time stamp.')

    julian_date = observation_time.jd_utc()

    location_string = log.get_param('location').split(' ')
    observatory = exoclock.Observatory(exoclock.Degrees(location_string[0]),
                                       exoclock.Degrees(location_string[1]))
    ra_dec_string = log.get_param('target_ra_dec').split(' ')
    target = exoclock.FixedTarget(exoclock.Hours(ra_dec_string[0]), exoclock.Degrees(ra_dec_string[1]))

    airmass = observatory.airmass(target, observation_time)

    fits_header[log.hops_observatory_latitude_key] = observatory.latitude.deg_coord()
    fits_header[log.hops_observatory_longitude_key] = observatory.longitude.deg()
    fits_header[log.hops_target_ra_key] = target.ra.deg()
    fits_header[log.hops_target_dec_key] = target.dec.deg_coord()
    fits_header[log.hops_datetime_key] = observation_time.utc().isoformat()
    fits_header[log.hops_exposure_key] = exp_time
    fits_header[log.hops_filter_key] = log.get_param('filter')
    fits_header[log.time_key] = julian_date
    fits_header[log.airmass_key] = airmass

    return fits_header, observation_time


def stack_frames_calc(frames, headers, log, stack_count, t_eff):

    new_header = headers[-1].copy()
    new_header['STACKN'] = (len(headers), 'Number of stacked frames')


    exp_key = log.get_param('exposure_time_key')
    start_time = exoclock.Moment(jd_utc=headers[0][log.time_key])
    end_time = exoclock.Moment(jd_utc=headers[-1][log.time_key] + headers[-1][exp_key] / 24 / 3600)


    effective_exposure = (headers[-1][log.time_key] - headers[0][log.time_key]) * 24.0 * 3600.0 + headers[-1][exp_key]
    new_header[exp_key] = (round(effective_exposure, 2), 'Effective exposure time (seconds)')


    true_exposure = np.sum([ff[exp_key] for ff in headers])
    new_header['TRUEEXP'] = (round(true_exposure, 2), 'True exposure time (seconds)')


    new_header[log.hops_datetime_key] = (start_time.utc().isoformat(), 'Stack start')
    new_header[log.time_key] = (headers[0][log.time_key], 'Stack start JD')


    midpoint_jd = (headers[0][log.time_key] + headers[-1][exp_key] / 24 / 3600 + headers[-1][log.time_key]) / 2.0
    midpoint_time = exoclock.Moment(jd_utc=midpoint_jd)
    location_string = log.get_param('location').split(' ')
    observatory = exoclock.Observatory(exoclock.Degrees(location_string[0]), exoclock.Degrees(location_string[1]))
    ra_dec_string = log.get_param('target_ra_dec').split(' ')
    target = exoclock.FixedTarget(exoclock.Hours(ra_dec_string[0]), exoclock.Degrees(ra_dec_string[1]))
    airmass = observatory.airmass(target, midpoint_time)
    new_header[log.airmass_key] = airmass


    new_saturation = np.sum([ff[log.hops_saturation_key] for ff in headers])
    new_header[log.hops_saturation_key] = new_saturation


    dq_frame = np.zeros_like(frames[0])
    for nn in range(len(frames)):
        dq_frame += np.where(frames[nn] == headers[nn][log.hops_saturation_key], 1, 0)

    stacked = np.sum(frames, axis=0)
    stacked[np.where(dq_frame > 0)] = new_saturation

    mean, std = image_mean_std(stacked, samples=10000, mad_filter=5.0)
    new_header[log.mean_key] = mean
    new_header[log.std_key] = std


    psf = image_psf(
        stacked, headers[0], mean, std, 0.8 * new_saturation, centroids_snr=log.get_param('centroids_snr'),
        stars_snr=log.get_param('stars_snr'), psf_guess=log.get_param('psf_guess')
    )
    if np.isnan(psf):
        psf = 10
        skip = True
    else:
        skip = False
    new_header[log.psf_key] = psf
    new_header[log.skip_key] = skip


    bpp = 100 * np.sum(stacked - mean > 3 * std) / (len(stacked) * len(stacked[0]))
    new_header[log.bpp_key] = bpp


    new_header.add_comment(f'Effective exposure: {effective_exposure:.2f}s')
    new_header.add_comment(
        f'Time span: {start_time.utc().isoformat()} to {end_time.utc().isoformat()}')
    print(
        f"  Stack {stack_count + 1}: Effective exposure = {effective_exposure:.2f}s (target: {t_eff:.1f}s)")
    print(
        f"    Calculated: mean={mean:.1f}, std={std:.1f}, PSF={psf:.2f}, airmass={airmass:.3f}")


    return stacked, new_header, effective_exposure
