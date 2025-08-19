"""
Face Tracking and Distance Measurement Program

This program uses computer vision techniques to track a person's face in a video stream
and measures the distance between the person and the camera. It utilizes OpenCV for face
detection and visualization, pyfirmata library for Arduino communication, and cvzone library
for displaying annotations on the video. The program controls two servos connected to an
Arduino board to adjust the position of the camera based on the detected face.

Author: Nhi Pham, Lan Pham, Long Nguyen
Date: ...
"""

# Import libraries
import cv2
import cvzone
import mediapipe as mp
import pyfirmata
import numpy as np
import math

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
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
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

# Initialize the video capture object
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

# Set the resolution of the video capture
ws, hs = 1280, 720
cap.set(3, ws)
cap.set(4, hs)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Camera couldn't Access!!!")
    exit()

# Specify the serial port for communication with Arduino
port = "COM7"

# Connect to the Arduino board
board = pyfirmata.Arduino(port)

# Set up the servo pins on the Arduino board
servo_pinX = board.get_pin('d:10:s') #pin 9 Arduino
servo_pinY = board.get_pin('d:9:s') #pin 10 Arduino

# Create instances of face detection and face mesh detection classes
detector = FaceDetector()
detector_mesh = FaceMeshDetector(maxFaces=1)

# Initialize the servo position
servoPos = [90, 90]

while True:
    # Read a frame from the video capture
    success, img = cap.read()
    # Mirror the image horizontally
    img = cv2.flip(img, 1)
    
    # Detect faces and face meshes in the image
    img, bboxs = detector.findFaces(img, draw=False)
    img, faces = detector_mesh.findFaceMesh(img, draw=False)

    if bboxs and faces:
        # Get the coordinates of the face bounding box
        fx, fy = bboxs[0]["center"][0], bboxs[0]["center"][1]
        pos = [fx, fy]
        
        # Convert the coordinates to servo degrees
        servoX = np.interp(fx, [0, ws], [0, 180])
        servoY = np.interp(fy, [0, hs], [0, 180])
        
        # Update the servo positions
        servoPos[0] = servoX
        servoPos[1] = servoY-90
        if servoPos[1] <= 0:
            servoPos[1] = 0.5

        
        # Get the specific facial landmarks for distance calculation
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        # Drawing
        # cv2.line(img, pointLeft, pointRight, (0, 200, 0), 3)
        # cv2.circle(img, pointLeft, 5, (255, 0, 255), cv2.FILLED)
        # cv2.circle(img, pointRight, 5, (255, 0, 255), cv2.FILLED)
        
        # Get the specific facial landmarks for distance calculation
        w, _ = detector_mesh.findDistance(pointLeft, pointRight)
        W = 6.3
        
        # Calculate the distance between the facial landmarks
        f = 1000
        d = (W * f) / w
 
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
            cvzone.putTextRect(img, 'FURTHER', (50, 200), scale=5,colorR=(0,0,255))
        if d > far_thresh:
            cvzone.putTextRect(img, 'CLOSER', (50, 200), scale=5,colorR=(255,0,0))
        if close_thresh <= d <= far_thresh:
            cvzone.putTextRect(img, 'STAY!', (50, 200),scale=5,colorR=(0,255,0))
        
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
        cv2.circle(img, (fx, fy), 80, (0, 0, 255), 2)
        cv2.putText(img, str(pos), (fx+15, fy-15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2 )
        cv2.line(img, (0, fy), (ws, fy), (0, 0, 0), 2)  # x line
        cv2.line(img, (fx, hs), (fx, 0), (0, 0, 0), 2)  # y line
        cv2.circle(img, (fx, fy), 15, (0, 0, 255), cv2.FILLED)
        cv2.putText(img, "TARGET LOCKED", (850, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3 )

    else:
        cv2.putText(img, "NO TARGET", (880, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        cv2.circle(img, (640, 360), 80, (0, 0, 255), 2)
        cv2.circle(img, (640, 360), 15, (0, 0, 255), cv2.FILLED)
        cv2.line(img, (0, 360), (ws, 360), (0, 0, 0), 2)  # x line
        cv2.line(img, (640, hs), (640, 0), (0, 0, 0), 2)  # y line

    # Display the servo positions on the image
    cv2.putText(img, f'Servo X: {int(servoPos[0])} deg', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(img, f'Servo Y: {int(servoPos[1])} deg', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    # Write the servo positions to the Arduino board
    servo_pinX.write(servoPos[0])
    servo_pinY.write(servoPos[1])

    cv2.imshow("Image", img)
    
    # Check for the 'q' key press to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()