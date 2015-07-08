__author__ = 'Daguerreo'

import sys
from PyQt4 import QtGui, QtCore # importiamo i moduli necessari
# a
class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.setWindowTitle('Box example')
        cWidget = QtGui.QWidget(self)

        hBox = QtGui.QHBoxLayout()
        hBox.setSpacing(5)

        randomLabel = QtGui.QLabel('Enter the information', cWidget)
        hBox.addWidget(randomLabel)

        textEdit = QtGui.QTextEdit(cWidget)
        if textEdit.isReadOnly() is True:
            textEdit.setReadOnly(False)
        hBox.addWidget(textEdit)

        vBox = QtGui.QVBoxLayout()
        vBox.setSpacing(2)
        hBox.addLayout(vBox)

        button1 = QtGui.QPushButton('Go!', cWidget)
        vBox.addWidget(button1)
        button2 = QtGui.QPushButton('Reset', cWidget)
        vBox.addWidget(button2)

        cWidget.setLayout(hBox)
        self.setCentralWidget(cWidget)

app = QtGui.QApplication(sys.argv)
main = MainWindow()
main.show()
sys.exit(app.exec_())