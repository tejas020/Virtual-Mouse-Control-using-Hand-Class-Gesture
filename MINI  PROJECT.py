#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import pyautogui

# Initialize variables for low-pass filtering
prev_x, prev_y = 0, 0
alpha = 0.5  # Filter coefficient

# Function to detect fingers and control mouse
def detect_fingers(frame, bg_subtractor):
    global prev_x, prev_y
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(gray)
    # Apply thresholding
    _, thresh = cv2.threshold(fg_mask, 50, 255, cv2.THRESH_BINARY)
    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.erode(thresh, kernevl, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    # Find contours
    contours, _ = cv2.findContours(bthresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Check if any contours are found
    if contours:
        # Filter contours by area
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
        if contours:
            # Find the contour with the maximum area
            max_contour = max(contours, key=cv2.contourArea)
            # Smooth the contour
            epsilon = 0.01 * cv2.arcLength(max_contour, True)
            approx = cv2.approxPolyDP(max_contour, epsilon, True)
            # Find the convex hull of the smoothed contour
            hull = cv2.convexHull(approx, returnPoints=False)
            # Find convexity defects
            if len(hull) > 3:
                try:
                    defects = cv2.convexityDefects(approx, hull)
                    # If defects are found
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(approx[s][0])
                            end = tuple(approx[e][0])
                            far = tuple(approx[f][0])
                            # Draw lines on fingers
                            cv2.line(frame, start, end, [0, 255, 0], 2)
                            cv2.circle(frame, far, 5, [0, 0, 255], -1)
                            # Calculate distance from start to far point
                            distance = np.sqrt((start[0] - far[0]) ** 2 + (start[1] - far[1]) ** 2)
                            # If distance is small enough, perform click
                            if distance < 20:
                                pyautogui.click()
                            # If distance is large enough, move the mouse
                            elif distance > 20:
                                # Low-pass filter to smooth movement
                                prev_x = alpha * prev_x + (1 - alpha) * far[0]
                                prev_y = alpha * prev_y + (1 - alpha) * far[1]
                                pyautogui.moveTo(int(prev_x), int(prev_y))
                except Exception as e:
                    print(f"Error: {e}")
    return frame

# Main function
def main():
    global prev_x, prev_y
    # Initialize background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    # Open webcam
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Detect fingers and control mouse pointer
        frame = detect_fingers(frame, bg_subtractor)
        # Display the frame
        cv2.imshow('Finger Detection', frame)
        # Check for key press
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):  # Press 'q' to exit
            break
    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()

