from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class TrainCanvas(FigureCanvas):

    figure_changed = pyqtSignal()

    def __init__(self, parent=None, params=None):
        if not params:
            params = {}

        self.figure = Figure(**params)
        self.axes = self.figure.add_subplot(111)
        self.progress = None

        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(
            self,
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        FigureCanvas.updateGeometry(self)

        self.figure_changed.connect(
            self.on_figure_changed
        )


    def update_figure(self, value):
        #self.axes.plot(self.count, value)
        if isinstance(self.progress, np.ndarray):
            self.progress = np.append(self.progress, value)
        else:
            self.progress = np.asarray(value)
        self.axes.cla()
        self.axes.plot(self.progress, '*-')
        self.figure_changed.emit()

    def clear_figure(self):
        self.progress = None
        self.axes.cla()
        self.figure_changed.emit()

    def on_figure_changed(self):
        self.draw()


class Train(QWidget):
    """Обучение"""

    # сигналы
    train = pyqtSignal(dict)
    stop = pyqtSignal()

    # значения по-умолчанию
    params = {
        'epochs': 1,
        'batch_size': 8,
    }

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        valid_float = QDoubleValidator(0.0, 1.0, 6, self)

        # Для обучения нужно задать как минимум одну эпоху
        self.epochs = QLineEdit()
        self.epochs.setValidator(QIntValidator(bottom=1))
        self.epochs.setPlaceholderText('1')

        # Размер мини-пакета
        self.batch = QLineEdit()
        self.batch.setValidator(QIntValidator(1, 64))
        self.batch.setPlaceholderText('8')

        # Останов по достижению значения ошибки
        # во время обучения
        use_stop = QCheckBox('Stop value', self)
        self.stop_value = QLineEdit()
        self.stop_value.setValidator(valid_float)
        self.stop_value.setEnabled(False)
        use_stop.toggled[bool].connect(
            self.stop_value.setEnabled
        )

        # Метод обучения
        self.optimizer = QComboBox(self)
        self.optimizer.addItems([
            'None', 'Momentum', 'Adadelta'
        ])
        self.optimizer.setCurrentIndex(0)
        self.optimizer.activated[str].connect(
            self.optimizer_chosen
        )
        self.optimizer.activated[str].connect(
            self.optimizer_chosen
        )

        # Константы методов
        # Скорость обучения
        self.learning_rate = QLineEdit()
        self.learning_rate.setValidator(valid_float)
        # Момент
        self.momentum = QLineEdit(self)
        self.momentum.setValidator(valid_float)
        self.momentum.setEnabled(False)
        # Ро
        self.rho = QLineEdit(self)
        self.rho.setValidator(valid_float)
        self.rho.setEnabled(False)

        self.progress = TrainCanvas()

        # Кнопки
        # Отправить парметры обучения
        self.train_btn = QPushButton('train')
        self.train_btn.clicked.connect(
            self.on_train
        )
        # Прервать обучение
        self.stop_btn = QPushButton('stop')
        self.stop_btn.clicked.connect(
            self.on_stop
        )
        self.stop_btn.setEnabled(False)
        # Очистить график обучения
        self.clear_btn = QPushButton('clear')
        self.clear_btn.clicked.connect(
            self.on_clear
        )

        flo = QFormLayout()
        flo.addRow('Epochs', self.epochs)
        flo.addRow('Batch size', self.batch)
        flo.addRow(use_stop, self.stop_value)
        flo.addRow('Optimizer', self.optimizer)
        flo.addRow('Learning rate', self.learning_rate)
        flo.addRow('Momentum', self.momentum)
        flo.addRow('Rho', self.rho)

        btn_lo = QHBoxLayout()
        btn_lo.addStretch()
        btn_lo.addWidget(self.clear_btn)
        btn_lo.addWidget(self.train_btn)
        btn_lo.addWidget(self.stop_btn)

        lo = QVBoxLayout()
        lo.addLayout(flo)
        lo.addWidget(self.progress)
        lo.addLayout(btn_lo)

        self.setLayout(lo)

    @pyqtSlot()
    def on_train(self):
        """
        Составить словарь параметров обучения и
        отправить в сигнале train(dict)
        """
        value = self.value()
        self.train.emit(value)
        self.progress.clear_figure()

    @pyqtSlot()
    def on_stop(self):
        """Отправить сигнал stop()"""
        self.stop.emit()

    @pyqtSlot()
    def on_clear(self):
        """Очистить график"""
        self.progress.clear_figure()

    @pyqtSlot(str)
    def optimizer_chosen(self, optimizer):
        if optimizer == 'Momentum':
            self.learning_rate.setEnabled(True)
            self.momentum.setEnabled(True)
            self.rho.setEnabled(False)
        elif optimizer == 'Adadelta':
            self.learning_rate.setEnabled(False)
            self.momentum.setEnabled(False)
            self.rho.setEnabled(True)
        else:  # optimizer = 'None'
            self.learning_rate.setEnabled(True)
            self.momentum.setEnabled(False)
            self.rho.setEnabled(False)

    def value(self):
        value = self.params

        epochs = self.epochs.text()
        if epochs:
            value['epochs'] = int(epochs)

        batch = self.batch.text()
        if batch:
            value['batch_size'] = int(batch)

        if self.stop_value.isEnabled():
            stop_value = self.stop_value.text()
            if stop_value:
                value['stop'] = float(stop_value)

        value['optimizer'] = self.optimizer.currentText()

        constants = {}
        def get_constant(line, name, type, default):
            if line.isEnabled():
                arg = line.text()
                if arg:
                    arg = type(arg)
                else:
                    arg = default
                constants[name] = arg

        get_constant(self.learning_rate, 'learning_rate', float, 0.001)
        get_constant(self.momentum, 'momentum', float, 0.8)
        get_constant(self.rho, 'rho', float, 0.95)

        value['constants'] = constants
        return value