"""
Finger Counting Game with Face and Hand Detection

This code implements a finger counting game using face and hand detection. 
It uses OpenCV, Mediapipe, and Pyfirmata libraries to detect faces, facial landmarks, and hand gestures. 
The game displays equations on the screen, and the player has to show the correct number of fingers to 
solve each equation. The code tracks the player's hand gestures, counts the number of fingers, 
and compares it to the correct answer. It also generates random positions for target windows and
checks if the player's hand is within the target windows. The game tracks the player's score and 
displays it on the screen. The code is structured using classes for face detection, face mesh detection, 
and hand detection. It also includes helper functions to generate random equations, check overlapping windows, 
and create target windows.

Author: Thanh Long, Nhi Pham, Hoang Lan
Date: 22/5/2023
"""

# Import libraries
import random
import cv2
import cvzone
import mediapipe as mp
import pyfirmata
import numpy as np
import math
import time
import os
import random

import HandTrackingModule as htm

class FaceDetector:
    """
    FaceDetector Class

    This class encapsulates the functionality for detecting faces in an image using OpenCV's
    face detection algorithm. It provides methods to initialize the face detection model, perform
    face detection on an image, and draw bounding boxes around the detected faces.
    
    """
    def __init__(self, minDetectionCon=0.5):
        """
        Initializes the FaceDetector object with a minimum detection confidence threshold.
        """
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        """
        Detects faces in the given image using the FaceDetection model.

        Args:
            img (numpy.ndarray): The image in BGR format.
            draw (bool, optional): Whether to draw bounding boxes around the detected faces. 
                                   Defaults to True.

        Returns:
            tuple: A tuple containing the image with or without bounding boxes and a list of bounding boxes.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)
                bboxInfo = {"id": id, "bbox": bbox, "score": detection.score, "center": (cx, cy)}
                bboxs.append(bboxInfo)
                if draw:
                    img = cv2.rectangle(img, bbox, (255, 0, 255), 2)

                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        return img, bboxs
    
class FaceMeshDetector:
    """
    FaceMeshDetector Class

    This class encapsulates the functionality for detecting facial landmarks using the FaceMesh
    model. It provides methods to initialize the face mesh detection model, perform face mesh
    detection on an image, and draw the detected facial landmarks.

    """
    def __init__(self, staticMode=False, maxFaces=10, minDetectionCon=0.5, minTrackCon=0.5):
        """
        Initializes the FaceMeshDetector object with the given parameters.

        Args:
            staticMode (bool, optional): Whether to use static image mode. Defaults to False.
            maxFaces (int, optional): Maximum number of faces to detect. Defaults to 2.
            minDetectionCon (float, optional): Minimum detection confidence threshold. Defaults to 0.5.
            minTrackCon (float, optional): Minimum tracking confidence threshold. Defaults to 0.5.
        """
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode,
                                                 max_num_faces=self.maxFaces,
                                                 min_detection_confidence=self.minDetectionCon,
                                                 min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        """
        Detects face meshes in the given image using the FaceMesh model.

        Args:
            img (numpy.ndarray): The image in BGR format.
            draw (bool, optional): Whether to draw the face meshes on the image. Defaults to True.

        Returns:
            tuple: A tuple containing the image with or without face meshes and a list of face mesh coordinates.
        """
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return img, faces

    def findDistance(self,p1, p2, img=None):
        """
        Calculates the Euclidean distance between two points and optionally visualizes them on the image.

        Args:
            p1 (tuple): First point (x1, y1).
            p2 (tuple): Second point (x2, y2).
            img (numpy.ndarray, optional): The image in BGR format. Defaults to None.

        Returns:
            tuple: A tuple containing the distance, information about the points, and the image (if provided).
        """
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length,info, img
        else:
            return length, info
        
def check_overlap(window, windows):
    x, y = window
    for existing_window in windows:
        x_existing, y_existing = existing_window
        # Check if the given window overlaps with any existing window
        if abs(x - x_existing) < window_size[0] and abs(y - y_existing) < window_size[1]:
            return True
    return False

def create_window(num_windows, offset):
    windows = []  # List to store the generated windows
    # Generate non-overlapping random window positions
    while len(windows) < num_windows:
        x = random.randint(offset, wCam - window_size[0])
        y = random.randint(offset, hCam - window_size[1])
        new_window = (x, y)
        if not check_overlap(new_window, windows):
            windows.append(new_window)
    return windows

def check_point_in_window(point, window):
    x, y = point
    x_window, y_window = window
    # Check if the given point is inside the specified window
    if x_window <= x <= x_window + window_size[0] and y_window <= y <= y_window + window_size[1]:
        return True
    return False

def generate_random_equation():
    equation = ''
    result = 0
    num_constant = 5

    # Generate num_constant random integer numbers
    numbers = [random.randint(0, num_constant-1) for _ in range(num_constant)]

    # Generate random operators and build the equation
    for i in range(num_constant-1):
        operator = random.choice(['+', '-'])
        equation += str(numbers[i]) + operator
        if operator == '+':
            result += numbers[i+1]  # Use the next number for addition
        elif operator == '-':
            result -= numbers[i+1]  # Use the next number for subtraction

    equation += str(numbers[num_constant-1])
    result += numbers[0]  # Add the first number to the result

    return equation, result

def list_random_equation_result(no_equations):
    list_equation = []
    list_result = []
    for i in range(0, int(no_equations)):
        # Generate a random equation until the result is between 0 and 5
        equation, result = generate_random_equation()
        while not (0 <= result <= 5):
            equation, result = generate_random_equation()
        list_equation.append(equation)
        list_result.append(result)
    return list_equation, list_result

# Specify the serial port for communication with Arduino
port = "COM7"

# Connect to the Arduino board
board = pyfirmata.Arduino(port)

# Set up the servo pins on the Arduino board
servo_pinX = board.get_pin('d:10:s') #pin 9 Arduino
servo_pinY = board.get_pin('d:9:s') #pin 10 Arduino

# Initialize the servo position
servoPos = [70, 150]

# Set the webcam resolution
wCam, hCam = 640, 480

window_size = (50, 50)  # Desired window size
num_windows = 5  # Number of windows to generate

# Create a list of window positions for target detection
windows = create_window(num_windows, offset=40)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, wCam)
cap.set(4, hCam)

# Create instances of detections
detector_face = FaceDetector()
detector_mesh = FaceMeshDetector()
detector = htm.handDetector(detectionCon=0.75)
totalFingers = -1

finger_countDown = 0

# Set the initial time and interval for showing equations
start_time = time.time()
question_interval = 7

# Set the number of equations to generate
no_equation = 15

# Generate a list of random equations and results
list_equation, list_result = list_random_equation_result(no_equation)

# Create a video capture object to read from the webcam
cap = cv2.VideoCapture(0)

# Initialize variables for tracking time and frame rate
prev_frame_time = 0
new_frame_time = 0

# Initialize variables
current_equation_index = 0
start_time = time.time()

# Initialize the state of the rectangles
rectangles = [False] * num_windows

# Initialize variables for tracking points and counted boxes
point = 0
counted_boxes = set()

while finger_countDown < 4:
    
    # print('finger_countDown', finger_countDown)
    
    # Read a frame from the webcam
    ret, img = cap.read()
    
    # Flip the image horizontally
    img = cv2.flip(img, 1)
    show_totalFingers = True
    img = detector.findHands(img)
    
    # Detect faces in the image
    img, bboxs = detector_face.findFaces(img)
    img, faces = detector_mesh.findFaceMesh(img, draw = False)
    
    cv2.putText(img, "COUNT DOWN YOUR FINGER 3,2,1", (50, 430), cv2.FONT_HERSHEY_PLAIN, 2, (0,255, 0), 3)
    lmList, bbox = detector.findPosition(img, draw=False)
    
    if faces:
        # Get the specific facial landmarks for distance calculation
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        
        # Get the specific facial landmarks for distance calculation
        w, _ = detector_mesh.findDistance(pointLeft, pointRight)
        W = 6.3
        
        # Calculate the distance between the facial landmarks
        f = 1000
        d = (W * f) / w

        # Display the distance on the image
        cvzone.putTextRect(img, f'Distance: {int(d)}cm',
                        (face[10][0] - 100, face[10][1] - 50),
                        scale=2)
        
        # Display the calculated distance on the image
        middle = 75
        difference = 10
        far_thresh = middle + difference
        close_thresh = middle - difference
        
        # Classify the distance and display the corresponding text on the image
        if d < close_thresh:
            cvzone.putTextRect(img, 'FURTHER', (20, 90), scale=2,colorR=(0,0,255))
        if d > far_thresh:
            cvzone.putTextRect(img, 'CLOSER', (20, 90), scale=2,colorR=(255,0,0))
        if close_thresh <= d <= far_thresh:
            cvzone.putTextRect(img, 'STAY!', (20, 90),scale=2,colorR=(0,255,0))
            
    # Calculate the number of fingers show
    if lmList:
        fingersUp = detector.fingersUp()
        totalFingers = fingersUp.count(1)
        print("totalFingers", totalFingers)
        if totalFingers == 3 and finger_countDown == 0:
            finger_countDown += 1
        if totalFingers == 2 and finger_countDown == 1:
            finger_countDown += 1
        if totalFingers == 1 and finger_countDown == 2:
            finger_countDown += 1
        if totalFingers == 0 and finger_countDown == 3:
            finger_countDown += 1
    else:
        totalFingers = -1
        show_totalFingers = not show_totalFingers
    # Show the total number of fingers in a rectangle
    if show_totalFingers:
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)    
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
        
    start_play_time = time.time()
    # Display the image
    cv2.imshow("Image", img)
    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        start_time = time.time()
        break
        
# Main loop
while True:   
    # Get the current time and calculate the frame rate
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time+0.001)
    prev_frame_time = new_frame_time
    # converting the fps into integer
    fps = int(fps)
    
    # Flag to indicate if an equation should be shown
    show_equation = False
    current_time = time.time()

    # Calculate the elapsed time
    elapsed_time = int(current_time - start_time)
    # print("elapsed_time", elapsed_time)
    
    duration = 60
    game_time = int(duration - (current_time - start_play_time))
    
    # Check if it's time to show an equation
    if elapsed_time >= question_interval:
        show_equation = True

    if show_equation:
        servoPos = [70, 150]
        servo_pinX.write(servoPos[0])
        servo_pinY.write(servoPos[1])
        
        # Choose a random equation and its corresponding result
        index = random.randint(0, no_equation-1)
        question = list_equation[index]
        answer = list_result[index]
        show_totalFingers = True

        # Wait for the user to show the correct number of fingers
        while totalFingers != answer:
            # Read a frame from the webcam
            ret, img = cap.read()
            
            current_time = time.time()
            game_time = int(duration-(current_time - start_play_time))
            
            # Get the current time and calculate the frame rate
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time+0.001)
            prev_frame_time = new_frame_time
            
            # converting the fps into integer
            fps = int(fps)
            
            # Flip the image horizontally
            img = cv2.flip(img, 1)
                        
            cvzone.putTextRect(img, f'Time left: {int(game_time)}', (150, 50), scale=2, colorR=(0,0,255))
            
            # Draw the equation on the image
            cvzone.putTextRect(img, question, (200, 450), scale = 2, colorR = (0, 255, 0))
            cvzone.putTextRect(img, f'Point: {int(point)}', (380, 90),scale=2,colorR=(0,255,0))
            cv2.putText(img, f'FPS: {int(fps)}', (575,20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

            # Detect hands in the image
            img = detector.findHands(img)
            lmList, bbox = detector.findPosition(img, draw=False)
            show_totalFingers = True
            
            # Calculate the number of fingers show
            if lmList:
                fingersUp = detector.fingersUp()
                totalFingers = fingersUp.count(1)
            else:
                totalFingers = 6
                show_totalFingers = not show_totalFingers
                
            # Show the total number of fingers in a rectangle
            if show_totalFingers:
                cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)    
                cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
                
            if game_time == 0:
                break
            
            # Display the image
            cv2.imshow("Image", img)
            
            # Check for 'q' key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        # Reset the start time to show the next equation
        start_time = time.time()    
    else:
        # Read a frame from the webcam
        ret, img = cap.read()
        
        # Flip the image horizontally
        img = cv2.flip(img, 1)
        cvzone.putTextRect(img, f'Time left: {int(game_time)}', (150, 50), scale=2, colorR=(0,0,255))
        
        # Detect faces in the image
        img, bboxs = detector_face.findFaces(img)
        img, faces = detector_mesh.findFaceMesh(img, draw = False)
        
        for window in counted_boxes:
            x,y = window
            cv2.rectangle(img, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)
        
        # print(counted_boxes)
        
        if bboxs:
            # Get the coordinates of the face bounding box
            fx, fy = bboxs[0]["center"][0], bboxs[0]["center"][1]
            pos = [fx, fy]

            # Convert the coordinates to servo degrees
            servoX = np.interp(fx, [0, wCam], [0, 180])
            servoY = np.interp(fy, [0, hCam], [0, 180])
            
            # Update the servo positions
            servoPos[0] = servoX
            servoPos[1] = servoY-95
            if servoPos[1] <= 0:
                servoPos[1] = 0.5

            # print('servoPos', servoPos)
            
            if faces:
                # Get the specific facial landmarks for distance calculation
                face = faces[0]
                pointLeft = face[145]
                pointRight = face[374]
                
                # Get the specific facial landmarks for distance calculation
                w, _ = detector_mesh.findDistance(pointLeft, pointRight)
                W = 6.3
                
                # Calculate the distance between the facial landmarks
                f = 1000
                d = (W * f) / w

                # Display the distance on the image
                cvzone.putTextRect(img, f'Distance: {int(d)}cm',
                                (face[10][0] - 100, face[10][1] - 50),
                                scale=2)
                
                # Display the calculated distance on the image
                middle = 75
                difference = 10
                far_thresh = middle + difference
                close_thresh = middle - difference
                
                # Classify the distance and display the corresponding text on the image
                if d < close_thresh:
                    cvzone.putTextRect(img, 'FURTHER', (20, 90), scale=2,colorR=(0,0,255))
                if d > far_thresh:
                    cvzone.putTextRect(img, 'CLOSER', (20, 90), scale=2,colorR=(255,0,0))
                if close_thresh <= d <= far_thresh:
                    cvzone.putTextRect(img, 'STAY!', (20, 90),scale=2,colorR=(0,255,0))
                    
                # Clamp the servo positions within the valid range
                if servoPos[0] < 0:
                    servoPos[0] = 0
                elif servoPos[0] > 180:
                    servoPos[0] = 180
                if servoPos[1] < 0:
                    servoPos[1] = 0
                elif servoPos[1] > 180:
                    servoPos[1]= 180
                
            # Draw and illustrate how the program detect face 
            cv2.circle(img, (fx, fy), 60, (0, 0, 255), 2)
            cv2.circle(img, (fx, fy), 5, (0, 0, 255), cv2.FILLED)
            cv2.putText(img, str(pos), (fx+15, fy-15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            cv2.line(img, (0, fy), (wCam, fy), (0, 0, 0), 2)  # x line
            cv2.line(img, (fx, hCam), (fx, 0), (0, 0, 0), 2)  # y line
            cv2.putText(img, "TARGET LOCKED", (380, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        
            # Write the servo positions to the Arduino board
            servo_pinX.write(servoPos[0])
            servo_pinY.write(servoPos[1])
                
            # Check if the point is inside each window and update the state of the rectangles
            for i, window in enumerate(windows):
                x, y = window
                if check_point_in_window((fx, fy), window):
                    rectangles[i] = True
                    servoPos = [70, 150]
                    # Write the servo positions to the Arduino board
                    servo_pinX.write(servoPos[0])
                    servo_pinY.write(servoPos[1])
                    cv2.rectangle(img, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)
                    if window not in counted_boxes:
                        point += 1
                        counted_boxes.add(window)
                else:
                    rectangles[i] = False

            # Draw the rectangles on the frame for the windows where rectangles[i] is True
            for i, window in enumerate(windows):
                if rectangles[i]:
                    x1, y1 = window
                    x2, y2 = window[0] + window_size[0], window[1] + window_size[1]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Check if the target is not detected
        else:
            # Display "NO TARGET" message
            cv2.putText(img, "NO TARGET", (380, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            
            # Draw a red circle at the center of the frame
            cv2.circle(img, (320, 240), 60, (0, 0, 255), 2)
            cv2.circle(img, (320, 240), 5, (0, 0, 255), cv2.FILLED)
            
            # Draw horizontal and vertical lines on the image
            cv2.line(img, (0, 240), (wCam, 240), (0, 0, 0), 2)  # x line
            cv2.line(img, (320, hCam), (320, 0), (0, 0, 0), 2)  # y line

        # Display the number of points on the image
        cvzone.putTextRect(img, f'Point: {int(point)}', (380, 90),scale=2,colorR=(0,255,0))
        cv2.putText(img, f'FPS: {int(fps)}', (575,20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        
        # Check if the player has reached 5 points
        if point == 5:
            # Display "YOU WON!!!" message
            while True:
                ret, img = cap.read()
                
                # Flip the image horizontally
                img = cv2.flip(img, 1)
                
                # Detect faces in the image
                img, bboxs = detector_face.findFaces(img)
                img, faces = detector_mesh.findFaceMesh(img)
                
                for window in windows:
                    x, y = window
                    cv2.rectangle(img, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)
                cvzone.putTextRect(img,'YOU WON!!!', (150, 230),scale=4,colorR=(0,255,0))
                cv2.imshow("Image", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        # Check if the time is over
        if game_time == 0:
            # Display "GAME OVER!!!" message
            while True:
                ret, img = cap.read()
                
                # Flip the image horizontally
                img = cv2.flip(img, 1)
                
                # Detect faces in the image
                img, bboxs = detector_face.findFaces(img)
                img, faces = detector_mesh.findFaceMesh(img)
                
                # Display each generated window on the frame
                for window in windows:
                    x, y = window
                    cv2.rectangle(img, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)
                cvzone.putTextRect(img,'GAME OVER!', (150, 230),scale=4,colorR=(0,0,255))
                cv2.imshow("Image", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break 
            break

        # Display the image
        cv2.imshow("Image", img)
        
        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()