from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from gui.status import SetStatus, NetStatus

class Info(QWidget):
    """ Информация о загруженном AE и сете """
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.net_status = NetStatus()
        self.set_status = SetStatus()

        lo = QHBoxLayout()
        lo.addWidget(self.net_status)
        lo.addWidget(self.set_status)

        self.setLayout(lo)