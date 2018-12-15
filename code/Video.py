from Line import *
from undist import undist
from binary_threshold import binary_threshold
from perspective_transform import perspective_transform
import cv2

class Video(object):
    def __init__(self):
        self.left_line = Line()
        self.right_line = Line()
        self.mtx = None
        self.dist = None
        self.sobel_kernel_size = 3
        self.x_grad_threshold = (30, 150)
        self.hls_s_channel_threshold = (125, 255)
        self.perspective_M = None
        self.perspective_Minv = None
        # Reset the line detection after too many bad curvature transform
        self.reset_lane_detect = True

    def sliding_window(self, image):
        binary_warped = np.copy(image)
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and  visualize the result
        output = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(output, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (0, 255, 0), 2)
            cv2.rectangle(output, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        self.left_line.allx = leftx
        self.left_line.ally = lefty
        self.left_line.current_fitx = left_fitx
        self.left_line.current_fit = left_fit
        self.right_line.allx = rightx
        self.right_line.ally = righty
        self.right_line.current_fitx = right_fitx
        self.right_line.current_fit = right_fit

    def search_around_poly(self, image):
        # Choose the width of the margin around the previous polynomial to search
        margin = 100

        binary_warped = np.copy(image)

        left_fit = self.left_line.best_fit
        right_fit = self.right_line.best_fit

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Fetch best fit as the initialized arguments
        left_fit = self.left_line.best_fit
        right_fit = self.right_line.best_fit

        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                       left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                             left_fit[1] * nonzeroy + left_fit[
                                                                                 2] + margin)))

        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                        right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                               right_fit[1] * nonzeroy + right_fit[
                                                                                   2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        self.left_line.allx = leftx
        self.left_line.ally = lefty
        self.left_line.current_fitx = left_fitx
        self.left_line.current_fit = left_fit
        self.right_line.allx = rightx
        self.right_line.ally = righty
        self.right_line.current_fitx = right_fitx
        self.right_line.current_fit = right_fit

    def calc_radius_curvature(self):
        left_x = self.left_line.allx
        left_y = self.left_line.ally
        right_x = self.right_line.allx
        right_y = self.right_line.ally
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)
        left_fit_cr = np.polyfit(left_y * ym_per_pix, left_x * xm_per_pix, 2)
        right_fit_cr = np.polyfit(right_y * ym_per_pix, right_x * xm_per_pix, 2)
        y_eval = np.max(left_x)
        y_eval_meter = y_eval * ym_per_pix

        left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
        left_curverad_cr = ((1 + (
                    2 * left_fit_cr[0] * y_eval_meter + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad_cr = ((1 + (
                    2 * right_fit_cr[0] * y_eval_meter + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        left_x_of_ymax = left_fit_cr[0] * y_eval_meter ** 2 + left_fit_cr[1] * y_eval_meter + left_fit_cr[2]
        right_x_of_ymax = right_fit_cr[0] * y_eval_meter ** 2 + right_fit_cr[1] * y_eval_meter + right_fit_cr[2]

        self.left_line.radius_of_curvature = left_curverad
        self.right_line.radius_of_curvature = right_curverad
        self.left_line.radius_of_curvature_meter = left_curverad_cr
        self.right_line.radius_of_curvature_meter = right_curverad_cr
        self.left_line.x_of_ymax = left_x_of_ymax
        self.right_line.x_of_ymax = right_x_of_ymax

    def annotate(self, image):
        # Create an image to draw the lines on
        color_warp = np.zeros_like(image).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        left_fit = self.left_line.best_fit
        right_fit = self.right_line.best_fit
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.perspective_Minv, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        annotated_img = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        left_curve_meter = self.left_line.radius_of_curvature_meter
        right_curve_meter = self.right_line.radius_of_curvature
        mean_curve_meter = (left_curve_meter + right_curve_meter) / 2.0

        left_lane_pos = self.left_line.x_of_ymax
        right_lane_pos = self.right_line.x_of_ymax
        # Calculate the center position in meters
        center_lane_pos_m = (left_lane_pos + right_lane_pos) / 2.0

        # Car center position in pixel (middle of the image)
        car_center_position = image.shape[1] / 2.0
        # Car center position in meter
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        car_center_position_m = car_center_position * xm_per_pix

        center_dist = (car_center_position_m - center_lane_pos_m)
        center_dist_direction = ''
        if center_dist > 0:
            center_dist_direction = 'right'
        else:
            center_dist_direction = 'left'
        center_dis_abs = abs(center_dist)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated_img, 'Radius of Curvature = %.3f m' % mean_curve_meter, (50, 50), font, 1, (255, 255, 255), 2)
        cv2.putText(annotated_img, 'left curvature: %.3f m' % left_curve_meter, (50, 100), font, 1, (255, 255, 255), 2)
        cv2.putText(annotated_img, 'right curvature: %.3f m' % right_curve_meter, (50, 150), font, 1, (255, 255, 255), 2)
        cv2.putText(annotated_img, 'Vehicle Position = {0:.2f} (m) {1} of the center'.format(center_dis_abs, center_dist_direction), (50, 200), font, 1, (255, 255, 255), 2)

        return annotated_img

    def process(self, image):
        # Undistort the image
        undist_img, self.mtx, self.dist = undist(image)

        # Binary threshold the undistorted image
        bin_threshold_img = binary_threshold(undist_img, self.sobel_kernel_size, self.x_grad_threshold, self.hls_s_channel_threshold)

        # Perspective transform to bird-eye view
        perspective_trans_img, self.perspective_M, self.perspective_Minv = perspective_transform(bin_threshold_img)

        # If reset detection been activated, redo the sliding window search
        # Else just search in a margin around the pervious line position
        if self.reset_lane_detect:
            self.sliding_window(perspective_trans_img)
        else:
            self.search_around_poly(perspective_trans_img)

        self.calc_radius_curvature()
        self.left_line.update(self.reset_lane_detect)
        self.right_line.update(self.reset_lane_detect)

        max_bad_detection = 6
        if self.left_line.bad_detection > max_bad_detection or self.right_line.bad_detection > max_bad_detection:
            self.reset_lane_detect = True
        else:
            self.reset_lane_detect = False

        result = self.annotate(undist_img)
        return result
