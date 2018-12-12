## Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_Images/Original_Test_Image.jpg "Original Calibration Image"
[image2]: ./output_Images/Undistort_Test_Image.jpg "Undistorted Calibration Image"
[image3]: ./output_Images/Original_Image.jpg "Original Image"
[image4]: ./output_Images/Undistort_Image.jpg "Undistorted Image"
[image5]: ./output_Images/Binary_Threshold_Image.jpg "Binary Example"
[image6]: ./output_Images/Perspective_Transform_Image.jpg "Warp Example"
[image7]: ./output_Images/Fit_Polynomial_Warped_Image.jpg "Fit Visual"
[image8]: ./output_Images/Add_Figure_LaneLines_Image.jpg "Output"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

Here you are!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
* Read in calibration images
* Generate object points
* Find the image points
* Get camera calibration matrix and distortion coefficients via cv2.calibrateCamera function
* Distort the image below via cv2.undistort funct

![alt text][image1] "Original Calibration Image"
![alt text][image2] "Undistorted Calibration Image"

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
* Read in one of the test images
* Apply the matrix and coefficients generated in Step 1
![alt text][image3]
![alt text][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
* Threshold x gradient(for grayscaled image)
* Threshold color S channel
* Combine the two binary threshold into one for the final binary image result

![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
* Transform the image from the car camera's perspective to a bird-eye-view perspective
* Hard-code the source and destination polygon coordinates(as the following table shows) and obtain the matrix M that maps them on each other
* Masking the image only for the region of interest(lane area)
* Warp the image to the new bird-eye-view perspective

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
* Divide the image into 'nwindows' windows(steps)
* For each step, find the boundaries of current window based on the starting point and 'margin' parameter
* For each step, find all the activated pixels
* Fit a polynomial to all the relevant pixels in the sliding windows
* Set the area to search for activated pixels based on 'margin' parameter, using function search_around_poly() in the corresponding jupyter notebook cell

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
* Using the poly_fit coefficients to calculate the left_curve and right_curve curvature and offset for the vehicle position in the Image
* Convert the pixel scale to meters for curature and offset(vehicle position to the center) by using the xm_per_pix &ym_per_pix and offset calculation algorithm in the corresponding jupyter notebook cell

Result:
Vehicle is 0.236450 m left of center

Left lane curve radius:  2192.1294185762163
Right lane curve radius:  2306.8113055038393

Left lane curve radius in real world: 719.075795 m
Right lane curve radius in real world: 747.408772 m

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
* Warp the lane lines back on undistorted image
* Combine lane lines with undistorted image
* Combine figure(Curvature&Vehicle Position) with undistorted image

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Nothing special.
