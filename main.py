import cv2
import numpy as np
import time

# Global variables
start_time = None
prev_position = None
prev_est_vel = [0, 0]  # Previous estimated velocity
bounce_count = 0
bounce_thresh = 20  # Adjust as needed

# Function to detect the basketball and draw a circle around it
def track_basketball(frame, bounce_count):
    """
    Function to detect the basketball in a given frame and draw a circle around it.

    Parameters:
    - frame (numpy.ndarray): Input frame in BGR color space.
    - bounce_count (int): Count of detected bounces.

    Returns:
    - frame (numpy.ndarray): Frame with circle drawn around the detected basketball.
    - prev_est_vel (list): Previous estimated velocity of the basketball.
    - bounce_count (int): Updated count of detected bounces.

    This function detects a basketball in the provided frame using color segmentation in the HSV color space.
    It then applies morphological operations to remove noise, finds contours, and identifies the largest contour
    as the basketball. If the radius of the detected circle meets a minimum size threshold, it draws a circle
    around it and tracks its movement to detect bounces. The function returns the modified frame, previous
    estimated velocity, and the updated bounce count.
    """
    global prev_position, prev_est_vel
    
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range of yellow color in HSV (for basketball)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Threshold the HSV image to get only yellow colors (for basketball)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Morphological operations to remove noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Proceed if at least one contour was found
    if contours:
        # Find the largest contour
        max_contour = max(contours, key=cv2.contourArea)
        
        # Get the minimum enclosing circle
        ((x, y), radius) = cv2.minEnclosingCircle(max_contour)
        
        # Proceed if the radius meets a minimum size
        if radius > 10:
            # Draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)
            
            # Dribble detection logic
            position = np.array([x, y])
            if prev_position is not None:
                # Estimate velocity
                est_vel = [position[0] - prev_position[0], position[1] - prev_position[1]]

                # Check if the sign of the velocity has changed
                if np.sign(est_vel[1]) < 0 and np.sign(prev_est_vel[1]) > 0:
                    # Check for bounces from large change in velocity
                    # dvy = abs(est_vel[1] - prev_est_vel[1])
                    # if dvy > bounce_thresh:
                    bounce_count += 1

                # Update previous state trackers
                prev_est_vel = est_vel.copy()
                
            prev_position = position
    
    return frame, prev_est_vel, bounce_count


# Function to calculate dribble frequency
def calculate_dribble_frequency(bounce_count):
    """
    Function to calculate the dribble frequency based on the count of detected bounces.

    Parameters:
    - bounce_count (int): Count of detected bounces.

    Returns:
    - dribble_frequency (float): Calculated dribble frequency.

    This function calculates the dribble frequency using the count of detected bounces.
    It computes the time elapsed since the start of dribbling, if available, and divides
    the bounce count by this time to obtain the dribble frequency. If no dribbling has occurred
    yet (i.e., no start time is available), it returns 0.0 as the dribble frequency.
    """
    global start_time
    if start_time is None:
        start_time = time.time()
    else:
        end_time = time.time()
        time_elapsed = end_time - start_time
        if time_elapsed > 0:
            dribble_frequency = bounce_count / time_elapsed
            return dribble_frequency
    return 0

# Function to calculate dribble velocity
def calculate_dribble_velocity(prev_est_vel):
    """
    Function to calculate the dribble velocity based on the previous estimated velocity.

    Parameters:
    - prev_est_vel (list): Previous estimated velocity of the basketball.

    Returns:
    - dribble_velocity (list): Calculated dribble velocity.

    This function simply returns the previous estimated velocity of the basketball as the dribble velocity.
    """
    return prev_est_vel

# Function to detect dribble direction
def detect_dribble_direction(prev_est_vel):
    """
    Function to detect the dribble direction based on the previous estimated velocity.

    Parameters:
    - prev_est_vel (list): Previous estimated velocity of the basketball.

    Returns:
    - direction (str): Detected dribble direction ("Up", "Down", or "Unknown").

    This function determines the dribble direction based on the vertical component of the
    previous estimated velocity. If the vertical component is negative, it indicates an upward
    movement and returns "Up". If the vertical component is positive, it indicates a downward
    movement and returns "Down". If the vertical component is zero, it returns "Unknown".
    """
    if prev_est_vel[1] < 0:
        return "Up"
    elif prev_est_vel[1] > 0:
        return "Down"
    else:
        return "Unknown"
    
# Function to display basketball tracking information
def display_frame(frame, bounce_count, dribble_velocity, prev_est_vel):
    """
    Function to display basketball tracking information on the frame.

    Parameters:
    - frame (numpy.ndarray): Input frame with basketball tracking information.
    - bounce_count (int): Count of detected bounces.
    - dribble_velocity (list): Dribble velocity information.
    - prev_est_vel (list): Previous estimated velocity of the basketball.

    Returns:
    - frame (numpy.ndarray): Frame with displayed basketball tracking information.

    This function displays various basketball tracking information on the input frame, including:
    - Dribble count
    - Dribble frequency
    - Dribble velocity
    - Dribble direction

    It calculates dribble frequency using the count of detected bounces and the elapsed time,
    displays dribble velocity, and detects dribble direction based on the previous estimated velocity.
    """
    # Display dribble count on frame
    cv2.putText(frame, f'Dribbles: {bounce_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Calculate dribble frequency per unit time
    dribble_frequency = calculate_dribble_frequency(bounce_count)
    cv2.putText(frame, f"Dribble Frequency: {dribble_frequency:.2f} dribbles/s", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate dribble velocity
    dribble_velocity_text = f"Dribble Velocity:{dribble_velocity}"
    cv2.putText(frame, dribble_velocity_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Detect and display dribble direction
    dribble_direction = detect_dribble_direction(prev_est_vel)
    cv2.putText(frame, f'Direction: {dribble_direction}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

# Open the video file
video_file = "WHATSAAP ASSIGNMENT.mp4"
cap = cv2.VideoCapture(video_file)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Loop through the video frames
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Check if the frame was read successfully
    if not ret:
        break
    
    # Track the basketball and draw a circle
    output_frame, dribble_velocity, bounce_count = track_basketball(frame, bounce_count)
    
    # Display information on the frame
    output_frame = display_frame(output_frame, bounce_count, dribble_velocity, prev_est_vel)

    # Display the resulting frame
    cv2.imshow('Basketball Tracker', output_frame)
    
    # Slow down the video by introducing a delay
    delay = 40  # Delay in milliseconds (25 frames per second)
    if cv2.waitKey(delay) & 0xFF == ord('q'):  # Exit if 'q' is pressed
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
