import argparse
import sys
import time

import cv2
import numpy as np
from PIL import ImageGrab
from PyQt5.QtWidgets import QApplication

import classes.Feedback as fb
from classes.Feedback import FeedbackWindow
from classes.classification import EmotionClassifier
from classes.eye_detection import detect_attention
from classes.gaze_tracking.gaze_tracker import GazeHub
from classes.plotting import MainWindow

classifier = EmotionClassifier()


def main(path, vis_interval, plot_interval):
    # if path == 0 VideoCapture captures over camera
    cap = cv2.VideoCapture(int(path)) if path == '0' else cv2.VideoCapture(path)
    gazehub = GazeHub()
    plot = MainWindow()
    plot.show()
    fb.createCsvFile()
    if vis_interval == 0.0:
        capture(cap, plot, time.time(), plot_interval, gazehub)
    else:
        capture_with_interval(cap, plot, time.time(), vis_interval, plot_interval, gazehub)

    # create Feedback Window
    app = QApplication(sys.argv)
    window = FeedbackWindow("Analysed_Data.csv")
    window.show()
    app.exec_()


def capture(cap, plot, start_time, plot_interval, gazehub):
    ret, plot_counter = True, 0
    ret, frame = cap.read()
    while ret:
        stamp = time.time() - start_time
        attention_score = detect_attention(frame, gazehub)
        cnn_out = classifier.predict(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            break
        if stamp > plot_counter:
            plot.update(stamp, attention_score)
            plot_counter += plot_interval
        fb.appendCsvFile(stamp, attention_score, cnn_out)
        cv2.imshow('frame', frame)
        ret, frame = cap.read()


def capture_with_interval(cap, plot, start_time, vis_interval, plot_interval, gazehub):
    ret, interval_counter, plot_counter = True, 0, 0
    ret, frame = cap.read()
    attention_score = detect_attention(frame, gazehub)

    while ret:
        # printscreen_pil = ImageGrab.grab()
        # printscreen_numpy = np.array(printscreen_pil.getdata(), dtype='uint8') \
        #     .reshape((printscreen_pil.size[1], printscreen_pil.size[0], 3))
        # cv2.imshow('window', printscreen_numpy)
        # cv2.imwrite('faces_detected.jpg', printscreen_numpy)
        #
        # image = printscreen_numpy
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #
        # faceCascade = cv2.CascadeClassifier('classes/assets/haarcascade_frontalface_default.xml')
        # faces = faceCascade.detectMultiScale(
        #     gray,
        #     scaleFactor=1.3,
        #     minNeighbors=3,
        #     minSize=(30, 30)
        # )
        #
        # print("[INFO] Found {0} Faces.".format(len(faces)))
        #
        # for i, (x, y, w, h) in enumerate(faces):
        #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #     roi_color = image[y:y + h, x:x + w]
        #     print("[INFO] Object found. Saving locally.")
        #     cv2.imwrite('face' + str(i) + '.jpg', roi_color)

        stamp = time.time() - start_time
        cnn_out = classifier.predict(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            break
        if stamp > interval_counter:
            attention_score = detect_attention(frame, gazehub)
            fb.appendCsvFile(stamp, attention_score, cnn_out)
            interval_counter += vis_interval
        if stamp > plot_counter:
            plot.update(stamp, attention_score)
            plot_counter += plot_interval
        cv2.imshow('frame', frame)
        ret, frame = cap.read()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('v_path', type=str, default='0', help='set path to video')
    parser.add_argument('--vis_inter', type=float, default=0.0, help='set the visual analysis interval')
    parser.add_argument('--plot_inter', type=float, default=0.1, help='set the plotting analysis interval')
    args = parser.parse_args()

    main(args.v_path, args.vis_inter, args.plot_inter)
