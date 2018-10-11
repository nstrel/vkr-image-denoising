from PyQt5.QtWidgets import QLineEdit, QFileDialog
from PyQt5.QtCore import pyqtSignal


def get_existing_directory():
    return QFileDialog().getExistingDirectory()


class LineSelector(QLineEdit):
    selected = pyqtSignal(str)

    def __init__(self, getter=get_existing_directory,
                 contents='', parent=None):
        QLineEdit.__init__(self, contents, parent)
        self.getter = getter

    def mouseDoubleClickEvent(self, event):
        line = self.getter()
        self.setText(line)
        self.selected.emit(line)
