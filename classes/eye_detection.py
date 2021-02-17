import cv2
import dlib
import os

from scipy.spatial import distance as dist
from .gaze_tracking import gaze_tracker


face_cascade = cv2.CascadeClassifier('./classes/assets/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./classes/assets/haarcascade_eye.xml')
cwd = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.abspath(os.path.join(cwd, "./assets/shape_predictor_68_face_landmarks.dat"))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)


def EAR_to_prob(ear):
    # plot ear to scores. 0.38 seems like it's the maximum for "openness"
    if ear < 0.10:
        ear = ear - 0.6
    elif ear < 0.15:
        ear = ear - 0.5
    elif ear < 0.21:
        ear = ear - 0.3
    return 0.45 - ear

def detect_attention(frame, gazehub, visual=True):
    
    faces = detector(frame)
    scores = []
    gaze_factors = []
        
    for e, face in enumerate(faces):
        
        # perform gaze detection
        if e not in gazehub.hub.keys():
            gaze = gaze_tracker.GazeTracking()
            gazehub.add_gaze(gaze, e)
        face_landmarks = predictor(frame, face)
        current_gaze = gazehub.get_gaze(e)
        current_gaze.refresh(frame, face_landmarks)
        gaze_factors.append(current_gaze.gaze_to_factor())
        if visual:
            current_gaze.annotated_frame(frame)
        
        # calculate EAR
        leftEye = []
        rightEye = []

        for n in range(36,42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            if visual:
                cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
            
        for n in range(42,48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            if visual:
                cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
    
        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)
        
        EAR = (left_ear+right_ear)/2
        EAR = round(EAR,2)
        scores.append(EAR_to_prob(EAR))
        
        
        
    ear_avg = sum(scores) / len(scores) if scores else 0.0
    gaze_avg = (sum(gaze_factors) / len(gaze_factors)) if gaze_factors else 0.0
    avg = 1 * gaze_avg
    avg = avg - ear_avg
    return min(1.0, max(avg, 0.0))

def calculate_EAR(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	return ear

def is_looking_into_camera(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 2, minSize=(200, 200))
    eyes = []

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 2, minSize=(1, 1))
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    return (len(eyes)/2)/len(faces)

if __name__ == '__main__':
    pass
