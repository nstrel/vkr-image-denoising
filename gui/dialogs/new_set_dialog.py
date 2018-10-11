from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from gui.dialogs.line_selector import LineSelector
from util import current_directory
from training_set import create_training_set

class newSet(QDialog):

    params = pyqtSignal(dict)

    default = {
        'start': 0.2,
        'stop': 0.3,
        'num': 2
    }

    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('New training set')
        self.initUI()

    def initUI(self):

        validator = QIntValidator(10, 500)
        # Папка с изображениями для генерации
        self.source = LineSelector()
        # Папка для сохранения выборки
        self.destination = LineSelector()
        # К какому размеру привести изображение
        use_scale = QCheckBox('Resize')
        self.scale = QLineEdit()
        self.scale.setValidator(validator)
        self.scale.setEnabled(False)
        use_scale.toggled[bool].connect(
            self.scale.setEnabled
        )
        # На изображения какого размера порезать
        use_crop = QCheckBox('Crop')
        self.crop = QLineEdit()
        self.crop.setValidator(validator)
        self.crop.setEnabled(False)
        use_crop.toggled[bool].connect(
            self.crop.setEnabled
        )

        gauss = QGroupBox()
        gauss.setTitle('noise')
        self.start_lvl = QLineEdit()
        self.start_lvl.setValidator(QDoubleValidator(0.0, 0.5, 6))
        self.stop_lvl = QLineEdit()
        self.stop_lvl.setValidator(QDoubleValidator(0.0, 0.5, 6))
        self.levels = QLineEdit()
        self.levels.setValidator(QIntValidator(1, 5))
        gauss_lo = QFormLayout()
        gauss_lo.addRow('start level', self.start_lvl)
        gauss_lo.addRow('stop level', self.stop_lvl)
        gauss_lo.addRow('level count', self.levels)
        gauss.setLayout(gauss_lo)

        flo = QFormLayout()
        flo.addRow('source', self.source)
        flo.addRow('destination', self.destination)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )

        buttons.accepted.connect(self.on_accept)
        buttons.rejected.connect(self.reject)

        flo.addRow(use_scale, self.scale)
        flo.addRow(use_crop, self.crop)

        lo = QVBoxLayout()
        lo.addLayout(flo)
        lo.addWidget(gauss)
        lo.addWidget(buttons)
        self.setLayout(lo)

    def on_accept(self):
        value = self.value()
        self.params.emit(value)
        self.accept()

    def value(self):

        value = self.default

        value['source'] = self.source.text() or current_directory()
        value['destination'] = self.destination.text() or current_directory()

        if self.scale.isEnabled():
            scale = self.scale.text()
            if scale:
                value['scale_sz'] = int(scale)

        if self.crop.isEnabled():
            crop = self.crop.text()
            if crop:
                value['crop_sz'] = int(crop)

        start = self.start_lvl.text()
        if start:
            value['start'] = float(start)

        stop = self.stop_lvl.text()
        if stop:
            value['stop'] = float(stop)

        levels = self.levels.text()
        if levels:
            value['num'] = int(levels)

        return value
