import numpy as np
import cv2


class Line():
    def __init__(self):
        # n iteration that we want to average the line information
        self.n = 16
        # was the line detected in the last iteration?
        self.detected = False
        # numbers of the bad detection in a row
        self.bad_detection = 0
        # polynomial coefficients of the last n iterations
        self.recent_fits = []
        # x values of the last n fits of the line
        self.recent_fitsx = []
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # average x values of the fitted line over the last n iterations
        self.best_fitx = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # polynomial coefficients for the previous fit
        self.previous_fit = [np.array([False])]
        # x values of the last n fits of the most recent fit
        self.current_fitx = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        self.radius_of_curvature_meter = None
        # the x value with max y-value
        self.x_of_ymax = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between best and current fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # threshod for the difference in fit coefficients
        self.thresh = np.array([1, 1, 1], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def add_to_recent_fits(self):
        self.recent_fits.append(self.current_fit)
        if len(self.recent_fits) > self.n:
            self.recent_fits.pop(0)

        self.best_fit = np.mean(self.recent_fits, 0)

        self.recent_fitsx.append(self.current_fitx)
        if len(self.recent_fitsx) > self.n:
            self.recent_fitsx.pop(0)

        self.best_fitx = np.mean(self.recent_fitsx, 0)

    def update(self, reset):
        if reset is True:
            self.recent_fits = []
            self.recent_fitsx = []
            self.diffs = np.array([0, 0, 0], dtype='float')
            self.bad_detection = 0
        else:
            if self.detected is True:
                self.diffs = np.absolute((self.current_fit - self.previous_fit) / self.previous_fit)
            else:
                self.diffs = np.absolute((self.current_fit - self.previous_fit) / self.previous_fit) * 3

        if np.all(self.diffs < self.thresh):
            self.add_to_recent_fits()
            self.detected = True
            self.bad_detection = 0
        else:
            self.detected = False
            self.bad_detection += 1


        self.previous_fit = self.current_fit
