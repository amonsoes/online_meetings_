from PyQt5 import QtWidgets, uic
from PyQt5 import QtGui, QtCore
from pyqtgraph import PlotWidget
import pyqtgraph as pg
import sys
import random

pg.setConfigOption('background', '#222222')

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        self.history_x = []
        self.history_y = []
        self.app = QtWidgets.QApplication(sys.argv)
        self.traces = dict()
        self.phase = 0.0
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('./classes/mainwindow.ui', self) # load ui
    
    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtWidgets.QApplication.instance().exec_()

    def plot(self, stamps, values, pen):
        self.graphWidget.plot(stamps, values, pen=pen)
            
    def update(self, x, y, name='visual'):
        if len(self.history_x) > 100:
            self.history_x.pop(0)
            self.history_y.pop(0)
        self.history_x.append(x)
        self.set_progressbar(y)
        self.history_y.append(y)
        self.graphWidget.plot(self.history_x, self.history_y, clear=True)
        self.phase += 0.1
        QtGui.QApplication.processEvents()
        """
        self.set_plotdata(name, self.x, y)
        
        """
    
    def set_progressbar(self, value):
        self.progressBar.setValue(value*100)
        
    def set_progressbar_2(self, value):
        self.progressBar_2.setValue(value*100)
    
    def set_plotdata(self, name, x, y):
        if name in self.traces:
            self.traces[name].setData(x, y)
        else:
            self.traces[name] = self.graphWidget.plot('y')

    def animation(self, func):
        timer = QtCore.QTimer()
        timer.timeout.connect()
        timer.start(100)
        self.start()

    def run_app(self):
        x = 0
        while True:
            self.update(x, random.random())
            x += 1

if __name__ == '__main__':         
    main = MainWindow()
    main.show()
    main.run_app()