from PyQt5.QtWidgets import QApplication
from classes.Feedback import FeedbackWindow
import classes.Feedback as fb
import os
import sys
import argparse


def main(csvFile):

    app = QApplication(sys.argv)
    window = FeedbackWindow(os.path.abspath('Saves/'+csvFile))
    window.show()
    app.exec_()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default='0', help='set name of csv meeting file')
    args = parser.parse_args()

    main(args.path)
