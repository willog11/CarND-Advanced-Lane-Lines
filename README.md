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

[image1]: ./output_images/corners.jpg "Corners"
[image2]: ./output_images/undistored_image.jpg "Undistorted image"
[image3]: ./output_images/road_image_undistorted.jpg "Undistorted road image"
[image4]: ./output_images/raw_binary_image.jpg "Final thresholded image"
[image5]: ./output_images/color_extraction_image.jpg "Colour extraction stages"
[image6]: ./output_images/thresholded_image.jpg "Combined thresholded image"
[image7]: ./output_images/corrected_image_0.jpg "Corrected raw image"
[image8]: ./output_images/corrected_binary_image.jpg "Corrected binary image"
[image9]: ./output_images/histogram_of_points.jpg "Histogram of points"
[image10]: ./output_images/lanes_found.jpg "Lines found"
[image11]: ./output_images/final_test_image.jpg "Final resulting image"
[video1]: ./project_video_result.avi "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step can be found in calib_camera.py. The pipeline for this script is very simple, it consists of the following:

1. Get object and image points of a checker-board pattern corners
2. Calibrate the camera using these points

**Get object and image points**

1. Feed in multiple images taken by the same camera of a checkerboard pattern (6x9 was used)
2. For each of these images convert them to grayscale and using *cv2.findChessboardCorners(..)* find the corners
3. Append the resulting object and image points to lists and draw the points for visualization purposes
4. Store these lists for use later

~~~
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

# If found, add object points, image points
if ret == True:
	objpoints.append(objp)
	imgpoints.append(corners)

	# Draw and display the corners
	cv2.drawChessboardCorners(img, (9,6), corners, r
~~~

The following is the resultant image:

![alt text][image1]


**Calibrate the camera**

The process is very straight forward to calibrate the camera: 

1. Read an image of the checkerboard pattern
2. Using *cv2.calibrateCamera(..)*, calibrate the cameras
3. Perform image undistortion on the original image and verify the result
4. Save the calibration result for use later

~~~
img = cv2.imread('camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
dst = cv2.undistort(img, mtx, dist, None, mtx)
	
# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
cam_pickle = {}
cam_pickle["mtx"] = mtx
cam_pickle["dist"] = dist
pickle.dump( cam_pickle, open( "cam_pickle.p", "wb" ) )
~~~

The following is the resultant image:

![alt text][image2]

### Pipeline (single images)

The code implemented to detect the lanes of the image and all supporting functionality can be found in detect_lines.py

#### 1. Provide an example of a distortion-corrected image.

This step is very similar to above. However first the calibration must be read back using pickle and the undistortion function is to be applied to the new image set

~~~
def pipeline(img, first_frame, left_fit, right_fit, ploty):
    visualize_preproc_steps = False
    
    undistort_img                       = undistort_image(img, False)
	
def undistort_image(img, visualize=False):
	# Read in the saved camera matrix and distortion coefficients
	cam_pickle = pickle.load( open( "cam_pickle.p", "rb" ) )
	mtx = cam_pickle["mtx"]
	dist = cam_pickle["dist"]

	undistort_img = cv2.undistort(img, mtx, dist, None, mtx)
~~~

The resulting images show a slight image distortion correction. The resulting undistorted image will be used for further processing to find the lanes.

![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. This process was broken down into the following steps:

1. Use of HLS colour space to binarize images to show yellow and white lanes only
2. Using various image thresholding techniques, only show lane markings
3. Combine the resulting data to give the final filtered image

~~~
def pipeline(img, first_frame, left_fit, right_fit, ploty):

	.....
	
	color_binary                        = extract_hls_features(undistort_img, False)
	thresh_binary                       = get_thresholded_img(undistort_img, True)

	combined                            = np.zeros_like(thresh_binary)
	combined[(color_binary == 1) | (thresh_binary == 1)] = 1
~~~

This results in the following raw image:

![alt text][image4]

**Usage of HLS color space**

HLS is used to give only yellow and white lines as grayscale is known to perform very poor on yellow lines and areas of bright sun and concrete.  Thus the following logic was implemented:

1. Convert the image to HLS
2. Find the white colours on the HLS image by thresholding the L colour space
3. Find the yellow colours on the HLS image by thresholding the H and S colour space
4. Finally combine the above steps to give a resulting binary image

~~~
def white_color_enhance(img):
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(img, lower, upper)
    return  white_mask

def yellow_color_enhance(img):
    lower = np.uint8([  10, 0, 100])
    upper = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(img, lower, upper)
    return yellow_mask

def white_yellow_enhance(img, white_mask, yellow_mask):
    overall_mask = cv2.bitwise_or(white_mask, yellow_mask)
    return overall_mask

def extract_hls_features(img, visualize=False):
    """Applies the HSL transform"""
    hls_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    white_mask = white_color_enhance(hls_image)
    yellow_mask = yellow_color_enhance(hls_image)
    overall_mask = white_yellow_enhance(hls_image, white_mask, yellow_mask)
    overall_mask[(overall_mask == 255)] = 1
~~~

![alt text][image5]

**Image thresholding techniques**

Various image thresholding techniques were applied to filter out everything in the image but lines. It was a simple 4 stage process:

1. Convert the frame to grayscale
2. Apply gradient thresholding to find edges
3. Apply magnitude thresholding to give only areas of strong intensity
4. Apply directional threholding to find point segments in a particular direction
5. Finally combine all 3 previous steps to create a binarized image

~~~
def get_thresholded_img(img, visualize=False):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grad_binary = grad_threshold(gray, orient='x', thresh_min=20, thresh_max=100)
    mag_binary = mag_threshold(gray, sobel_kernel=3, mag_thresh=(30, 100))
    dir_binary = dir_threshold(gray, sobel_kernel=11, thresh=(0.7, 1.3))
    
    combined = np.zeros_like(dir_binary)
    combined[(grad_binary==1) | ((mag_binary == 1) & (dir_binary == 1))] = 1
	

def grad_threshold(img, orient='x', thresh_min=0, thresh_max=255):

    # Apply x or y gradient with the OpenCV Sobel() function and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    mag = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*mag/np.max(mag))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    dir_grad = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(img)
    binary_output[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1

    return binary_output
~~~

The result of the following process can be found below:

![alt text][image6]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The image correction was performed in the *correct_image(..)*, a source image array was defined on straight lane markings on the raw image in the dataset provided. This was then transformed to a selected destination location where the lines will be corrected straight using the following code:

~~~
def pipeline(img, first_frame, left_fit, right_fit, ploty):
	...
	corrected_img, M                    = correct_image(combined, False)

def correct_image(img, visualize=False):
            
    # Creating mask area
    ROI_TOP_LEFT_X = 575
    ROI_TOP_RIGHT_X = 570
    ROI_BOTTOM_LEFT_X = 255
    ROI_BOTTOM_RIGHT_X = 220
    ROI_TOP_Y = 460
    ROI_BOTTOM_Y = 50
    imshape = img.shape

    src = np.float32([(ROI_BOTTOM_LEFT_X,imshape[0]-ROI_BOTTOM_Y),
                      (ROI_TOP_LEFT_X, ROI_TOP_Y), 
                      (imshape[1]-ROI_TOP_RIGHT_X, ROI_TOP_Y), 
                      (imshape[1]-ROI_BOTTOM_RIGHT_X,imshape[0]-ROI_BOTTOM_Y)])
    
    offset = 250
    img_size = (imshape[1],imshape[0])
    
    dst = np.float32([[offset, imshape[0]],
                      [offset, 0], 
                      [imshape[1]-offset, 0], 
                      [imshape[1]-offset, imshape[0]]])
    M = cv2.getPerspectiveTransform(src, dst)
    corrected_img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
~~~

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 255, 670      | 250, 720        | 
| 575, 460      | 250, 0      |
| 710, 460     | 1030, 0      |
| 1060, 670      | 1030, 720        |

The following resulting images confirms that the correction was good (top image is original before binarization and the bottom is the resulting combined image):

![alt text][image7]

![alt text][image8]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The find lines functionality is broke into 2 parts. The first of which looks for initial lines and the second part looks for the next set of line based on the previous  frame. 

~~~
def pipeline(img, first_frame, left_fit, right_fit, ploty):
...

	if first_frame == True:
        left_fit, right_fit, ploty          = find_initial_lanes(corrected_img, True)
    else:
        left_fit, right_fit, ploty          = find_next_lanes(corrected_img, left_fit, right_fit, False)
~~~

The logic of this is based upon the material provided in the lessons. A breakdown of which can be found below.

For *find_initial_lanes(...)* the following logic was followed:

1. A histogram of all pixels is taken of the bottom half of the image and split into left and right sides
2. A sliding window is created to find areas of points and is then passed over the entire image starting  from the bottom
3. Areas which have points (non zero pixels) are then added to a list
4. These areas are then checked to ensure the number of points found falls greater than the minimum amount required
5. Finally a second order polyline fit is created to represent f(y). This is used rather than f(x) as there could be multiple potential values for y for any single value of x
6. This polyline fit is then used later for drawing of lines

In the following images can be found the histogram of points (top) and resulting line detections (bottom):

![alt text][image9]

![alt text][image10]

In the next frames the line locations will already be known thus the previous polynomials are used as a starting point with a search window applied to find the current set of lines. The rest of the functionality remains the same.

~~~
def find_next_lanes(img, left_fit, right_fit, visualize=False):
	margin = 50
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
~~~


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Now that the lines were found, the curvature and lateral offset relative to center can be found.

~~~
def pipeline(img, first_frame, left_fit, right_fit, ploty):
	...
	mean_curve_m, lat_off_rel_ctr       = calc_curvature_offset(left_fit, right_fit, ploty)
~~~

The logic is very straight forward as there are many tutorials present to learn how to calculate the radius of curvature, such as [here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php)

From here the mean curvature between both lines and lateral offset relative to center was calculated.

~~~
def calc_curvature_offset(left_fit, right_fit, ploty):
	## Calculate the mean curvature and lateral offset relative to center of vehicle
    curve_mean = ((left_curverad_m + right_curverad_m) / 2)
        
    img_width = 1280
    center_img_m = (img_width/2 + 0.5) * xm_per_pix
    
    left_lane_m = np.max(left_fitx)* xm_per_pix
    right_lane_m = np.max(right_fitx)*xm_per_pix
    lane_midpt   = (left_lane_m + right_lane_m) / 2
    veh_pos_rel_center = lane_midpt - center_img_m
	
	return curve_mean, veh_pos_rel_center
~~~

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Finally the resulting line detection, curvature and lateral offset need to be redrawn back to the original image. The inverse of the perspective transform matrix is applied to the corrected image after the lanes are drawn. After which the information of curvature and lateral offset is drawn to the top of the frame.

~~~
def pipeline(img, first_frame, left_fit, right_fit, ploty):
	...
	result_image                        = draw_result_raw_image(img, corrected_img, 
                                                                M, left_fit, right_fit, ploty, 
                                                                mean_curve_m, lat_off_rel_ctr, False)
																
def draw_result_raw_image(img, binary_img, M, left_fit, right_fit, ploty, mean_curve_m, lat_off_rel_ctr, visualize=False):
    
    Minv = np.linalg.inv(M)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    # Write some Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,'Mean Radius of Curvature: {0:.2f}m'.format(mean_curve_m),(350,50), font, 1,(255,255,255),2)
    cv2.putText(result,' Lateral Offset Relative to Center {0:.2f}m'.format(lat_off_rel_ctr),(320,100), font, 1,(255,255,255),2)
~~~

This results in the following image:

![alt text][image11]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The following link is the target video for the project to be tested upon

[Demo Video][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

It is very clear even before testing on the extra videos what the fallbacks\issues with this architecture are:

* Tar seams and cracks will be very problematic and need to be handled in a targeted manner. This is a known issue of any lane detection system.
* The detected lines should be coupled and assumed to be mostly parallel with the exceptions of entry\exits. This will stablize further the detection.
* Tracking can be implemented to further improve on performance
* Scenes with extreme curvature and effects from hill such as pitch\roll of the vehilce will severely impact on the performance of the algorithm

Lane detection\sensing is a huge area where teams of engineers can spend years implementing a robust algorithm. For the purpose of this project I kept the implementation simple and clear to follow to show what is possible with not too many lines of code. This was a very interesting project as it brings in many aspects imaging and computer vision, from calculating the intrinsic calibration to applying many computer vision techniques to create a good lane detection algorithm.