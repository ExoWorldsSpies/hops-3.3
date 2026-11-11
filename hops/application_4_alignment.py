
import os
import time
import numpy as np

import matplotlib.patches as mpatches

from hops.application_windows import MainWindow
from hops.hops_tools.files import get_fits_data_and_header, open_fits, save_fits, open_dict, save_dict
from hops.hops_tools.image_analysis import image_find_stars, cartesian_to_polar
from hops.hops_tools.centroids_and_stars import _find_centroids
from hops.hops_tools.reduction import mean_std_from_median_mad
from hops.hops_tools.alignment import select_calibration_stars, angle_grid, check_alignment_candidate
from hops.hops_tools.alignment import update_rotation_angle, fov_visibility
from hops.thirdparty import twirl

import sys
sys.setrecursionlimit(100000000)


class AlignmentWindow(MainWindow):

    def __init__(self, log):

        MainWindow.__init__(self, log, name='HOPS - Alignment', position=2)

        # set variables, create and place widgets

        self.shift_tolerance_p = self.log.get_param('shift_tolerance_p')
        self.rotation_tolerance = self.log.get_param('rotation_tolerance')
        self.min_calibration_stars_number = int(self.log.get_param('min_calibration_stars_number'))
        self.centroids_snr = self.log.get_param('centroids_snr')
        self.stars_snr = self.log.get_param('stars_snr')
        self.faint_target_mode = self.log.get_param('faint_target_mode')
        self.moving_target_mode = self.log.get_param('moving_target_mode')
        self.rotating_field_mode = self.log.get_param('rotating_field_mode')
        self.compress_reduced = True

        self.all_frames = open_dict(self.log.all_frames_path)
        self.science_files = []
        for science_file in self.all_frames:
            if not self.all_frames[science_file][self.log.skip_key]:
                self.science_files.append([self.all_frames[science_file][self.log.time_key], science_file])
            else:
                self.all_frames[science_file][self.log.align_x0_key] = False
                self.all_frames[science_file][self.log.align_y0_key] = False
                self.all_frames[science_file][self.log.align_u0_key] = False

                fits = open_fits( os.path.join(self.log.reduction_directory, science_file))
                if self.compress_reduced:
                    fits[1].header.set(self.log.align_x0_key, False)
                    fits[1].header.set(self.log.align_y0_key, False)
                    fits[1].header.set(self.log.align_u0_key, False)
                else:
                    fits[0].header.set(self.log.align_x0_key, False)
                    fits[0].header.set(self.log.align_y0_key, False)
                    fits[0].header.set(self.log.align_u0_key, False)

                save_fits(fits, os.path.join(self.log.reduction_directory, science_file), overwrite=True)

        self.science_files.sort()
        self.science_files = [ff[1] for ff in self.science_files]

        self.skip_time = 0
        self.science_counter = 0
        self.test_level = None
        self.redraw = None
        self.stars = None
        self.science_file = None
        self.fits_data = None
        self.fits_header = None
        self.fits_hdu = None
        self.psf = None
        self.stars_detected = None
        self.rotation_detected = None
        self.check_num = None
        self.target_x0 = None
        self.target_y0 = None
        self.target_r = None
        self.target_u = None
        self.x0 = None
        self.y0 = None
        self.u0 = None
        self.x0_fits0 = None
        self.y0_fits0 = None
        self.u0_fits0 = None
        self.y_length = None
        self.x_length = None
        self.fits_datax = None
        self.fits_datay = None
        self.burn_limit = None
        self.all_stars = None
        self.comparisons = None
        self.comparisons_snr = None
        self.small_angles = None
        self.large_angles = None
        self.settings_to_check = None
        self.setting_checking = 0

        self.robust_aligner = None
        self.use_hybrid_alignment = True
        self.use_robust_only = False

        # common definitions for all images

        fits_data, fits_header = get_fits_data_and_header(os.path.join(self.log.reduction_directory, self.science_files[0]))

        t0 = time.time()
        _ = mean_std_from_median_mad(fits_data)
        self.fr_time = int(200 * (time.time()-t0))

        self.shift_tolerance = int(max(fits_data.shape) * (self.shift_tolerance_p / 100.0))
        self.circles_diameter = 12 * fits_header[self.log.psf_key]

        # progress window

        self.progress_figure = self.FitsWindow(
            fits_data=fits_data, fits_header=fits_header, input_name=self.science_files[0]
        )
        self.progress_all_stars = self.Label(text='')
        self.progress_alignment = self.Progressbar(task="Aligning frames")
        self.progress_half_frame = self.CheckButton(text='Show smaller frame', initial=0)

        self.setup_window([
            [[self.progress_figure, 0, 2]],
            [[self.progress_all_stars, 0]],
            [[self.progress_alignment, 0], [self.progress_half_frame, 1]],
            [[self.Button(text='STOP ALIGNMENT & RETURN TO MAIN MENU', commands=[self.trigger_exit]), 0, 2]],
            []
        ])

        self.set_close_button_function(self.trigger_exit)

    def run_alignment(self):

        self.progress_figure.adjust_size()

        if self.log.get_param('alignment_complete'):
            if self.askyesno('Overwrite files', 'Alignment has been completed, do you want to run again?'):
                self.log.set_param('alignment_complete', False)
                self.log.save_local_log()
            else:
                self.log.set_param('proceed', True)

        if not self.log.get_param('alignment_complete'):

            self.progress_all_stars.set('Analysing first frame...')

            self.after(self.find_all_stars)

        else:
            self.close()

    def alignment_log(self, *text):
        # print(*text)
        pass

    def find_all_stars(self):

        if self.exit:
            self.after(self.align)

        else:

            fits_data, fits_header = get_fits_data_and_header(
                os.path.join(self.log.reduction_directory, self.science_files[0])
            )

            self.progress_figure.load_fits(fits_data, fits_header, self.science_files[0])

            stars = image_find_stars(fits_data, fits_header,
                                     mean=fits_header[self.log.mean_key], std=fits_header[self.log.std_key],
                                     burn_limit=fits_header[self.log.hops_saturation_key],
                                     centroids_snr=self.centroids_snr, stars_snr=self.stars_snr,
                                     psf=fits_header[self.log.psf_key],
                                     progress_window=self,
                                     verbose='shallow'
                                     )

            stars = np.array(stars)

            # self.log.save_local_log()

            self.all_stars = stars
            self.x_length = len(fits_data[0])
            self.y_length = len(fits_data)
            self.fits_datax, self.fits_datay = np.meshgrid(np.arange(0, self.x_length) + 0.5,
                                                           np.arange(0, self.y_length) + 0.5)
            self.psf = fits_header[self.log.psf_key]
            self.burn_limit = fits_header[self.log.hops_saturation_key]

            self.progress_all_stars.set('Choosing calibrating stars...')

            self.after(self.choose_calibration_stars)

    def choose_calibration_stars(self):
        if self.exit:
            self.after(self.align)
        else:

            self.x0, self.y0, self.u0, self.comparisons, self.check_num = select_calibration_stars(
                stars=self.all_stars, x_length=self.x_length, y_length=self.y_length, psf=self.psf,
                burn_limit=self.burn_limit, faint_target_mode=self.faint_target_mode,
                moving_target_mode=self.moving_target_mode
            )

            self.x0_fits0 = 1.0 * self.x0   ## store the reference-frame anchor for twirl later
            self.y0_fits0 = 1.0 * self.y0
            self.u0_fits0 = 1.0 * self.y0

            if self.faint_target_mode:
                self.min_calibration_stars_number = 3

            if self.target_x0:
                self.target_r, self.target_u = cartesian_to_polar(self.target_x0, self.target_y0, self.x0, self.y0)

            self.small_angles, self.large_angles = angle_grid(
                psf=self.psf, comparisons=self.comparisons, rot_tol=self.rotation_tolerance
            )

            self.progress_all_stars.set(' ')
            self.after(self.align)

    def align(self):

        if self.exit:
            self.after(self.plot_current)

        else:

            if self.science_counter == 0:
                self.progress_alignment.initiate(len(self.science_files))

            self.stars = []
            self.science_file = self.science_files[self.science_counter]
            self.alignment_log(self.science_file)
            self.fits_data, self.fits_header, self.fits_hdu = get_fits_data_and_header(
                os.path.join(self.log.reduction_directory, self.science_file), return_hdu=True
            )

            if self.science_counter == 0:
                t0 = time.time()
                _ = mean_std_from_median_mad(self.fits_data)
                self.fr_time = int(1000 * (time.time()-t0))

            self.progress_alignment.show_message(' ')
            self.progress_figure.load_fits(
                self.fits_data, self.fits_header, self.science_file, draw=False,
                show_half=self.progress_half_frame.get()
            )

            self.stars_detected = False
            self.rotation_detected = False
            self.test_level = 1
            self.redraw = 0
            self.skip_time = 0

            self.after(self.detect_stars)

    def detect_stars(self):

        if self.exit:
            self.after(self.plot_current)

        else:

            self.settings_to_check = []
            self.setting_checking = 0

            if self.test_level == 1:

                self.alignment_log('Test Level: 1')

                self.stars = _find_centroids(
                    self.fits_data, self.x0 - self.shift_tolerance, self.x0 + self.shift_tolerance,
                    self.y0 - self.shift_tolerance, self.y0 + self.shift_tolerance, self.fits_header[self.log.mean_key],
                    self.fits_header[self.log.std_key], 2 * self.fits_header[self.log.hops_saturation_key],
                    self.fits_header[self.log.psf_key], snr=self.centroids_snr
                )

                # self.stars = image_find_stars(self.fits_data, self.fits_header,
                #                               x_low=self.x0 - self.shift_tolerance,
                #                               x_upper=self.x0 + self.shift_tolerance,
                #                               y_low=self.y0 - self.shift_tolerance,
                #                               y_upper=self.y0 + self.shift_tolerance,
                #                               x_centre=self.x0,
                #                               y_centre=self.y0,
                #                               mean=self.fits_header[self.log.mean_key],
                #                               std=self.fits_header[self.log.std_key],
                #                               burn_limit=2*self.fits_header[self.log.hops_saturation_key],
                #                               psf=self.fits_header[self.log.psf_key],
                #                               centroids_snr=self.centroids_snr, stars_snr=self.stars_snr,
                #                               order_by_flux=False,
                #                               sky_inner_aperture=2.0, sky_outer_aperture=5.0,
                #                               star_limit=1,
                #                               )

                if len(self.stars)>0:

                    self.stars = np.array(self.stars)[:, 1:]

                    self.stars = sorted(self.stars, key=lambda x: (x[0] - self.x0) ** 2 + (x[1] - self.y0) ** 2)

                    for star in self.stars:
                        self.settings_to_check.append([star[0], star[1], self.u0, star])
                        if self.rotating_field_mode:
                            for rotation in self.large_angles + self.u0:
                                self.settings_to_check.append([star[0], star[1], rotation, star])

                self.check_star()

            elif self.test_level == 2:

                self.alignment_log('Test Level: 2')

                self.stars = image_find_stars(
                    self.fits_data, self.fits_header, mean=self.fits_header[self.log.mean_key],
                    std=self.fits_header[self.log.std_key], burn_limit=2*self.fits_header[self.log.hops_saturation_key],
                    psf=self.fits_header[self.log.psf_key], centroids_snr=self.centroids_snr, stars_snr=self.stars_snr,
                    sky_inner_aperture=2.0, sky_outer_aperture=5.0, verbose=True, order_by_flux=True
                )

                self.progress_all_stars.set(' ')

                if len(self.stars) > self.min_calibration_stars_number:

                    self.check_num = max(self.min_calibration_stars_number - 0.5, len(self.stars) / 10.0)

                    all_stars = np.array(self.all_stars)

                    all_stars_to_check_x = np.array([all_stars[0][0]])
                    all_stars_to_check_y = np.array([all_stars[0][1]])
                    for star in all_stars:
                        if min(np.sqrt(
                                (star[0] - all_stars_to_check_x) ** 2 + (star[1] - all_stars_to_check_y) ** 2
                        )) > 6 * self.fits_header[self.log.psf_key]:
                            all_stars_to_check_x = np.append(all_stars_to_check_x, star[0])
                            all_stars_to_check_y = np.append(all_stars_to_check_y, star[1])

                    new_stars_to_check_x = np.array([self.stars[0][0]])
                    new_stars_to_check_y = np.array([self.stars[0][1]])
                    for star in self.stars:
                        if min(np.sqrt(
                                (star[0] - new_stars_to_check_x) ** 2 + (star[1] - new_stars_to_check_y) ** 2
                        )) > 6 * self.fits_header[self.log.psf_key]:
                            new_stars_to_check_x = np.append(new_stars_to_check_x, star[0])
                            new_stars_to_check_y = np.append(new_stars_to_check_y, star[1])

                    # for ii in range(20):
                    #     circle = mpatches.Circle((all_stars_to_check_x[ii], all_stars_to_check_y[ii]), 20, ec='r', fill=False)
                    #     self.progress_figure.ax.add_patch(circle)
                    # for ii in range(20):
                    #     circle = mpatches.Circle((new_stars_to_check_x[ii], new_stars_to_check_y[ii]), 30, ec='g', fill=False)
                    #     self.progress_figure.ax.add_patch(circle)

                    if len(new_stars_to_check_y) > self.min_calibration_stars_number:

                        X = twirl.utils.find_transform(
                            np.array([[all_stars_to_check_x[ii], all_stars_to_check_y[ii]]
                                      for ii in range(min(len(all_stars_to_check_y)-1, 20))]),
                            np.array([[new_stars_to_check_x[ii], new_stars_to_check_y[ii]]
                                      for ii in range(min(len(new_stars_to_check_y)-1, 20))]),
                            n=20, tolerance=3 * self.fits_header[self.log.psf_key])

                        # for ii in twirl.utils.affine_transform(X)(np.array([[new_stars_to_check_x[ii], new_stars_to_check_y[ii]] for ii in range(20)])):
                        #     circle = mpatches.Circle((ii[0], ii[1]), 40, ec='b', fill=False)
                        #     self.progress_figure.ax.add_patch(circle)
                        #
                        # self.progress_figure.draw()

                        center, check = twirl.utils.affine_transform(X)(
                            np.array([[self.x0_fits0, self.y0_fits0],[self.x0_fits0 + 10, self.y0_fits0]])
                        )

                        for star in sorted(
                                self.stars, key=lambda x: np.sqrt((x[0] - center[0])**2 + (x[1] - center[1])**2)
                        )[:5]:

                            base_angle = cartesian_to_polar(check[0], check[1], center[0], center[1])[1]
                            if self.rotating_field_mode:
                                angle_pool = self.large_angles + base_angle
                            else:
                                angle_pool = self.small_angles + base_angle

                            for rotation in angle_pool:
                                self.settings_to_check.append([star[0], star[1], rotation, star])

                self.after(self.check_star)

            elif self.test_level == 3:
                if self.askyesno('HOPS - Alignment', 'Stars not detected.\n'
                                                     'Do you want to skip this frame?'):
                    pass
                else:
                    if self.stars:
                        for star in self.stars:
                            for rotation in self.large_angles:
                                self.settings_to_check.append([star[0], star[1], rotation, star])

                self.after(self.check_star)

    def check_star(self):

        if self.exit:
            self.after(self.plot_current)

        else:

            if self.setting_checking < len(self.settings_to_check):

                x, y, u, star = self.settings_to_check[self.setting_checking]
                self.alignment_log('Checking star at: ', x, y, ', with rotation:', u)

                score = check_alignment_candidate(
                    fits_data=self.fits_data, candidate_x=x, candidate_y=y, candidate_u=u,
                    comparisons=self.comparisons, check_num=self.check_num, centroids_snr=self.centroids_snr,
                    mean=self.fits_header[self.log.mean_key], std=self.fits_header[self.log.std_key],
                    psf=self.fits_header[self.log.psf_key], log=self.alignment_log)

                if score >= self.check_num:
                    self.stars_detected = True
                    if self.test_level > 1:
                        self.rotation_detected = True
                    self.x0 = x
                    self.y0 = y
                    self.u0 = u
                    # self.f0 = star[-1]
                    self.after(self.plot_current)
                else:
                    self.setting_checking += 1
                    self.after(self.check_star)
            else:
                if self.test_level == 1:
                    self.test_level = 2
                    self.progress_all_stars.set('Analysing frame...')
                    self.progress_alignment.show_message('Testing large shift & rotation...')
                    self.progress_figure.load_fits(
                        self.fits_data, self.fits_header, self.science_file, draw=True,
                        show_half=self.progress_half_frame.get()
                    )
                    self.after(self.detect_stars)
                elif self.test_level == 2:
                    self.test_level = 3
                    self.progress_alignment.show_message('Testing all shifts & rotations...')
                    self.progress_figure.load_fits(
                        self.fits_data, self.fits_header, self.science_file, draw=True,
                        show_half=self.progress_half_frame.get()
                    )
                    self.after(self.detect_stars)
                else:
                    self.plot_current()

    def plot_current(self):

        if self.exit:
            self.after(self.save)

        else:

            if self.stars_detected:

                self.u0 = update_rotation_angle(
                    fits_data=self.fits_data, x0=self.x0, y0=self.y0, u0=self.u0,comparisons=self.comparisons,
                    psf=self.fits_header[self.log.psf_key]
                )

                self.alignment_log(self.x0, self.y0, self.u0)

                if self.compress_reduced:
                    self.fits_hdu[1].header.set(self.log.align_x0_key, self.x0)
                    self.fits_hdu[1].header.set(self.log.align_y0_key, self.y0)
                    self.fits_hdu[1].header.set(self.log.align_u0_key, self.u0)
                else:
                    self.fits_hdu[0].header.set(self.log.align_x0_key, self.x0)
                    self.fits_hdu[0].header.set(self.log.align_y0_key, self.y0)
                    self.fits_hdu[0].header.set(self.log.align_u0_key, self.u0)

                save_fits(self.fits_hdu, os.path.join(self.log.reduction_directory, self.science_file), overwrite=True)

                circle = mpatches.Circle((self.x0, self.y0), self.circles_diameter, ec='r', fill=False)
                self.progress_figure.ax.add_patch(circle)
                for ii in self.comparisons[:int(self.check_num + 0.5)]:
                    circle = mpatches.Circle((self.x0 + ii[0] * np.cos(self.u0 + ii[1]),
                                              self.y0 + ii[0] * np.sin(self.u0 + ii[1])),
                                             self.circles_diameter, ec='y', fill=False)
                    self.progress_figure.ax.add_patch(circle)

                if self.target_x0:

                    target_x = self.x0 + self.target_r * np.cos(self.u0 + self.target_u)
                    target_y = self.y0 + self.target_r * np.sin(self.u0 + self.target_u)
                    dd = len(self.fits_data[0]) * 0.1
                    arrow = mpatches.Arrow(target_x - dd, target_y - dd, 0.9 * dd, 0.9 * dd, width=dd/3, color='r')

                    text = self.log.get_param('target_name')
                    self.progress_figure.ax.text(target_x - 0.4 * dd, target_y - 0.5 * dd,
                                                 text, color='r', va='top', ha='left')
                    self.progress_figure.ax.add_patch(arrow)

            else:

                if self.compress_reduced:
                    self.fits_hdu[1].header.set(self.log.align_x0_key, False)
                    self.fits_hdu[1].header.set(self.log.align_y0_key, False)
                    self.fits_hdu[1].header.set(self.log.align_u0_key, False)
                else:
                    self.fits_hdu[0].header.set(self.log.align_x0_key, False)
                    self.fits_hdu[0].header.set(self.log.align_y0_key, False)
                    self.fits_hdu[0].header.set(self.log.align_u0_key, False)

                save_fits(self.fits_hdu, os.path.join(self.log.reduction_directory, self.science_file), overwrite=True)

                time.sleep(1)

            self.all_frames[self.science_file][self.log.align_x0_key] = self.x0
            self.all_frames[self.science_file][self.log.align_y0_key] = self.y0
            self.all_frames[self.science_file][self.log.align_u0_key] = self.u0

            if not self.x0:
                self.all_frames[self.science_file][self.log.skip_key] = True

            self.progress_figure.draw()
            if self.skip_time == 0:
                self.progress_alignment.update()
            else:
                self.progress_alignment.update(skip=time.time() - self.skip_time)
                self.skip_time = 0
            self.science_counter += 1

            if self.science_counter >= len(self.science_files):
                self.progress_all_stars.set('Aligning all stars in all frames...')
                self.after(self.save)
            else:

                if len(self.jobs) > self.jobs_completed + 1:
                    self.fr_time += 10
                else:
                    self.fr_time -= 2
                    self.fr_time = max(self.fr_time, 2)

                self.after(self.align, self.fr_time)

    def save(self):

        if self.exit:
            self.after(self.check_visibility)
        else:
            save_dict(self.all_frames, self.log.all_frames_path)
            self.after(self.check_visibility)

    def check_visibility(self):

        if self.exit:
            self.close()

        else:

            frame_alignments = []
            for science_file in self.science_files:
                meta = self.all_frames[science_file]
                frame_alignments.append(
                    {'align_x0': meta[self.log.align_x0_key], 'align_y0': meta[self.log.align_y0_key],
                     'align_u0': meta[self.log.align_u0_key], 'psf': meta[self.log.psf_key]})

            in_fov, fov_bounds = fov_visibility(
                all_stars=self.all_stars, frame_alignments=frame_alignments, x_length=self.x_length,
                y_length=self.y_length, psf=self.psf)

            all_stars_dict = {'all_stars': self.all_stars,
                              'x-length': self.x_length,
                              'y-length': self.y_length,
                              'psf': self.psf,
                              'burn_limit': self.burn_limit,
                              'in_fov': in_fov
                              }
            save_dict(all_stars_dict, self.log.all_stars_path)

            self.log.set_param('min_x', fov_bounds['x_min'])
            self.log.set_param('min_y', fov_bounds['y_min'])
            self.log.set_param('max_x', fov_bounds['x_max'])
            self.log.set_param('max_y', fov_bounds['y_max'])
            self.log.set_param('alignment_complete', True)
            self.log.set_param('alignment_version', self.log.version)
            self.log.save_local_log()
            self.log.set_param('proceed', True)

            self.close()
