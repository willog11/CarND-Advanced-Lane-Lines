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
[video1]: ./project_video.mp4 "Video"

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

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

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

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
