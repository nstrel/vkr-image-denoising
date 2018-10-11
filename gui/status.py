import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class StatusWidget(QWidget):
    changed = pyqtSignal()

    def __init__(self, keys, label=None, parent=None):
        QWidget.__init__(self, parent)

        self.fields = dict.fromkeys(keys)

        self.initUI(keys, label)
        self.changed.connect(self.on_changed)
        self.update_()

    def initUI(self, keys, label):
        self.info = QTextEdit()
        self.info.setReadOnly(True)

        lo = QVBoxLayout()
        if label:
            lo.addWidget(QLabel(label))
        lo.addWidget(self.info)
        self.setLayout(lo)

    def update_(self, active=False, fields=None):
        self.active = active
        if active and fields:
            self.fields.update(fields)
        else:
            keys = self.fields.keys()
            self.fields = dict.fromkeys(keys)
        self.changed.emit()

    def report(self):
        if self.active:
            report = 'Status: active\n'
            for key, val in self.fields.items():
                if isinstance(val, list):
                    val = '\n' + '\n'.join([' - ' + str(v) for v in val])
                report += '%s: %s\n' % (str(key), str(val))
        else:
            report = 'Status: not active\n'
        return report

    def on_changed(self):
        report = self.report()
        self.info.setText(report)


class SetStatus(StatusWidget):
    def __init__(self, parent=None):
        super(self.__class__, self).__init__(
            ['Directory', 'Count', 'Image size'],
            'Set status', parent
        )


class NetStatus(StatusWidget):
    def __init__(self, parent=None):
        super(self.__class__, self).__init__(
            ['File', 'Input shape', 'Layer info', 'Error'],
            'Net status', parent
        )
