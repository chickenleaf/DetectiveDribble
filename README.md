# Documentation:

The `DetectiveDribble` class is designed to detect and track a basketball in a given video file. It utilizes computer vision techniques provided by OpenCV to:
 1. locate the basketball
    - `def track_basketball()`
 2. detect/count dribbles
     - `bounce_count`
 3. estimate its velocity
     - `def calculate_dribble_velocity()`
 4. frequency accurately in real-time
     - `def calculate_dribble_frequency()`
 5. detect dribble direction
     - `def detect_dribble_direction()`

## Analysis Methods And Algorithms:

### 1. Basketball Detection and Tracking:

#### Color Segmentation and Thresholding::

The system utilizes the HSV color space to enhance color representation, specifically targeting the yellow color associated with the basketball. Each frame undergoes conversion from BGR to HSV color space. Subsequently, a predefined HSV range is applied to create a binary mask isolating the yellow regions indicative of the basketball.

- Convert the frame to HSV color space.
  ```
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  ```
- Define the lower and upper bounds for yellow color in HSV.
  ```
  lower_yellow = np.array([20, 100, 100])
  upper_yellow = np.array([30, 255, 255])
  ```
- Threshold the HSV image to create a binary mask isolating the yellow regions.
  ```
  mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
  ```

#### Noise Reduction:

To refine the binary mask and eliminate noise, morphological operations including erosion and dilation are employed. These operations aid in smoothing the edges of the detected object while effectively filling gaps within contours, ensuring the integrity of the basketball's representation.

- Perform morphological operations (erosion followed by dilation) on the binary mask to remove noise and refine the mask.
  ```
  mask = cv2.erode(mask, None, iterations=2)
  mask = cv2.dilate(mask, None, iterations=2)
  ```

#### Contour Detection:

Contours within the binary mask are identified using suitable algorithms. The largest contour is then selected as the representation of the basketball object, ensuring accurate tracking throughout the video.
```
contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

#### Basketball Detection:

- Select the largest contour detected as the potential basketball region.
  ```
  max_contour = max(contours, key=cv2.contourArea)
  ```
- Fit a circle around this contour to define the basketball's position and size.
  ```
  cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
  cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)
  ```

#### Movement Tracking:

Velocity estimation is achieved by calculating the displacement between consecutive frames, providing insight into the basketball's movement. Bounces are detected through sudden changes in velocity and reversal of it's direction, signifying impact with the ground and enabling precise tracking of the basketball's trajectory.

### 2. Bounce Count:

The bounce count is detected based on changes in the vertical velocity of the basketball. Here's the logic and algorithm used for bounce detection:

#### Logic:
##### Track the Basketball:

- Keep track of the position of the basketball in consecutive frames.
- Estimate the velocity of the basketball based on its position changes.
- Detect changes in velocity sign to identify dribbles (change in direction).
- Finally count bounces by also comparing the velocity change with a predefined threshold (`bounce_thresh = 20`).

##### Bounce Detection:

Bounce detection occurs when the vertical velocity changes sign from positive to negative, indicating the ball has reached its peak and is now descending.
By comparing the sign of the current and previous vertical velocities, bounces are detected, and the bounce count is incremented accordingly.

- Check if the sign of the vertical component of the estimated velocity has changed from positive to negative, indicating an upward movement followed by a downward movement.
- When this change occurs, increment the bounce count, indicating a bounce.

```
# Check if the sign of the velocity has changed
if np.sign(est_vel[1]) < 0 and np.sign(prev_est_vel[1]) > 0:
```  

##### Display Tracking Information:

Display the bounce count along with other tracking information.

### 3. Dribble Frequency Calculation:

#### Time Tracking:

Recording the initiation time of dribbling allows for accurate calculation of the elapsed time since the onset of the activity.

#### Bounce Count:

Throughout the video, the system tracks and counts the number of bounces detected, a fundamental metric for determining dribble frequency.

#### Frequency Calculation:

By dividing the bounce count by the elapsed time, the system computes the dribble frequency, providing valuable insights into the pace and intensity of gameplay. In scenarios where no dribbling has occurred, a default value of 0.0 is returned.

`dribble_frequency = bounce_count / time_elapsed` 

### 4. Dribble Velocity Calculation:

#### Previous Estimation:

The system retrieves the previously estimated velocity of the basketball, offering crucial information regarding its speed and direction.
```
est_vel = [position[0] - prev_position[0], position[1] - prev_position[1]]
prev_est_vel = est_vel.copy()
```

### 5. Dribble Direction Detection:

#### Vertical Component Analysis:

By analyzing the vertical component of the basketball's velocity, the system determines the direction of dribble. A negative component indicates an upward movement, while a positive component signifies a downward motion. In cases where the vertical component is zero, the direction is labeled as "Unknown," ensuring comprehensive analysis of dribbling patterns.

```
if prev_est_vel[1] < 0:
        return "Up"
    elif prev_est_vel[1] > 0:
        return "Down"
    else:
        return "Unknown"
```

## Challenges Faced During Implementation:

### 1. Color Segmentation Parameters:
- Tuning the HSV color range for accurate basketball detection.
- Iteratively adjusting parameters to handle varying lighting conditions.
  
### 2. Noise Reduction:
- Balancing erosion and dilation operations to remove noise without affecting basketball contour.
- Fine-tuning parameters to achieve optimal noise reduction.
  
### 3. Velocity Estimation:
- Accurately estimating basketball velocity while considering noise and sudden movements.
- Implementing robust velocity estimation logic to handle varying scenarios.
  
### 4. Real-time Performance:
- Ensuring efficient processing of video frames while performing multiple calculations.
- Optimizing code to maintain real-time performance without lag or delays.
  - Data Encapsulation into a class.
  - Parallelization: Utilize parallel processing or concurrency where applicable to leverage multiple CPU cores.
  - Code Profiling: Identify bottlenecks using profiling tools and optimize those sections.   

## Code Documentation:
- Each function is documented with clear explanations of parameters, return types, and purpose.
- Inline comments provide detailed explanations of key steps and logic within functions.
- Descriptive variable names enhance code readability and maintainability.
- Algorithmic decisions and potential adjustments are explained within comments for future reference and improvement.
