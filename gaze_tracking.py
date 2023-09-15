
from __future__ import division
import os
import cv2
import dlib
import numpy as np

class GazeTracking:
    """
    Tracks eye gaze
    """

    def __init__(self):
        self.frame = None
        self.l_eye = None
        self.r_eye = None
        self.calibration = Calibration()

        self.face_detector = dlib.get_frontal_face_detector()

        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self.predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_exist(self):
        """
        Checks pupil locations exist
        """

        return (self.r_eye and self.l_eye and self.r_eye.pupil and self.l_eye.pupil and
               isinstance(self.l_eye.pupil.x, (int, float)) and
               isinstance(self.l_eye.pupil.y, (int, float)) and
               isinstance(self.r_eye.pupil.x, (int, float)) and
               isinstance(self.r_eye.pupil.y, (int, float)))
  
    def update(self, frame):
        """
        Refresh and analyze frame

        Args:
            frame (numpy.ndarray): frame to analyze
        """
        self.frame = frame

        frame, arr_of_faces = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), self.face_detector(frame)

        if arr_of_faces:
            

            def create_eye(dir):
                return Eye(frame, self.predictor(frame, arr_of_faces[0]), dir, self.calibration)
            
            self.l_eye, self.r_eye = create_eye('left'), create_eye('right')
        else:
            self.l_eye, self.r_eye = None, None

    def pupil_pos(self):
        """
        Calculates pupil coordinates

        Returns:
            Function to generate pupil coordinates
        """
        if not self.pupils_exist:
            return None
        return lambda options: (options.origin[0] + options.pupil.x,
                                options.origin[1] + options.pupil.y)
    
    def pupil_left_pos(self):
        """
        Calculates left pupil coordinates

        Returns:
            Left pupil coordinates
        """
        return self.pupil_pos()(self.l_eye) if self.pupils_exist else None


    def pupil_right_pos(self):
        """
        Calculates right pupil coordinates

        Returns:
            Right pupil coordinates
        """
        return self.pupil_pos()(self.r_eye) if self.pupils_exist else None


class Calibration:
    """
    Calibrates pupil detection algorithm by estimating optimal
    binarization threshold value for person and camera
    """

    def __init__(self):
        self.nb_frames = 20
        self.thresholds = {"left": [], "right": []}

    def complete(self):
        return all(
            len(self.thresholds[side]) >= self.nb_frames
            for side in ["left", "right"]
        )

    def threshold(self, side):
        """
        Estimates threshold value depending on eye

        Args:
            side (string): indicates left or right eye

        Returns:
            Threshold value
        """
        eye_threshold = self.thresholds[side]
        return int(sum(eye_threshold) / len(eye_threshold))

    @staticmethod
    def iris_area(frame):
        """
        Estimates percent area of iris relative to eye surface

        Args:
            frame (frame object): current frame

        Returns:
            Percent area
        """
        cropped_frame = frame[5:-5, 5:-5]
        nb_pixels = cropped_frame.size
        nb_black = nb_pixels - cv2.countNonZero(cropped_frame)
        return nb_black / nb_pixels

    @staticmethod
    def estimate_optimal_threshold(eye_frame):
        """
        Calculates optimal threshold to binarize frame of given eye

        Args:
            eye_frame (numpy.ndarray): frame of given eye

        Returns:
            Optimal threshold value
        """
        mean_iris_area = 0.48
        thresholds = np.arange(5, 100, 5)
        trials = {}

        for threshold in thresholds:
            iris_frame = Pupil.image_processing(eye_frame, threshold)
            trials[threshold] = Calibration.iris_area(iris_frame)

        optimal_threshold, _ = min(
            trials.items(), key=(lambda p: abs(p[1] - mean_iris_area))
        )
        return optimal_threshold

    def evaluate(self, eye_frame, side):
        """
        Analyzes given image to enhance calibration

        Args:
            eye_frame (numpy.ndarray): frame of given eye
            side (string): indicates left or right eye
        """
        self.thresholds[side].append(self.estimate_optimal_threshold(eye_frame))

class Eye:
    """
    Creates isolated eye frame to initiate pupil detection
    """

    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

    def __init__(self, original_frame, landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None

        self.analyze(original_frame, landmarks, side, calibration)

    def isolate(self, frame, landmarks, points):
        """
        Isolate an eye in given frame

        Args:
            frame (numpy.ndarray): frame containing face
            landmarks (dlib.full_object_detection): facial landmarks
            points (list): 68 Multi-PIE landmarks eye points
        """
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        region = region.astype(np.int32)
        self.landmark_points = region

        black_frame = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
        mask = np.full((frame.shape[0], frame.shape[1]), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        margin = 5
        min_x, max_x = np.min(region[:, 0]) - margin, np.max(region[:, 0]) + margin
        min_y, max_y = np.min(region[:, 1]) - margin, np.max(region[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        self.center = (self.frame.shape[1]/2, self.frame.shape[0]/2)

    def analyze(self, original_frame, landmarks, side, calibration):
        """Detects and isolates the eye in a new frame, sends data to the calibration
        and initializes Pupil object.

        Args:
            original_frame (numpy.ndarray): webcam frame
            landmarks (dlib.full_object_detection): facial landmarks
            side (string): indicates left or right eye
            calibration (Calibration object): binarization threshold value object
        """
        if side == 'left': points = self.LEFT_EYE_POINTS
        elif side == 'right': points = self.RIGHT_EYE_POINTS
        else: return

        self.isolate(original_frame, landmarks, points)

        if not calibration.complete(): calibration.evaluate(self.frame, side)

        self.pupil = Pupil(self.frame, calibration.threshold(side))

class Pupil:
    """
    Detects iris and estimates pupil position
    """

    def __init__(self, eye_frame, threshold):
        self.threshold = threshold
        self.x, self.y = None, None
        self.detect_iris(eye_frame)

    @staticmethod
    def image_processing(eye_frame, threshold):
        """
        Isolates iris from frame of given eye

        Args:
            eye_frame (numpy.ndarray): frame of given eye
            threshold (int): optimal binarization threshold value

        Returns:
            Frame object singularly representing iris
        """
        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        new_frame = cv2.erode(new_frame, np.ones((3, 3), np.uint8), iterations=3)
        _, new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)

        return new_frame

    def detect_iris(self, eye_frame):
        """
        Identifies iris and estimates position via centroid calculation

        Arguments:
            eye_frame (numpy.ndarray): frame of given eye
        """
        self.iris_frame = self.image_processing(eye_frame, self.threshold)

        contours, _ = cv2.findContours(
            self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )[-2:]
        contours = sorted(contours, key=cv2.contourArea)

        try:
            moments = cv2.moments(contours[-2])
            self.x, self.y = int(moments["m10"] / moments["m00"]), int(
                moments["m01"] / moments["m00"]
            )
        except (IndexError, ZeroDivisionError):
            pass