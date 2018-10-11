import numpy as np
from PyQt5.QtCore import *


class TrainThread(QThread):
    """
    Обучение сети в происходит отдельном треде, чтобы основное окно не висело.
    """

    iteration = pyqtSignal(float)
    train_finished = pyqtSignal(float, bool)

    def __init__(self):
        QThread.__init__(self)
        self.train = None
        self.epochs = 0
        self.size = 0
        self.stop = 0
        self.break_loop = False

    def __del__(self):
        self.wait()

    def set_params(self, train, size, epochs, err):
        """
        :param train: функция для обучения
        :param size: размер обучающей выборки
        :param epochs: число эпох
        :param err: закончить обучение, если ошибка меньше err
        """
        self.train = train
        self.epochs = epochs
        self.size = size
        self.stop = err
        self.break_loop = False

    def run(self):
        indices = list(range(self.size))
        for i in range(self.epochs):

            np.random.shuffle(indices)

            avg_err = 0
            for index in indices:
                err = self.train(index)
                avg_err += abs(err)
                self.iteration.emit(err)
                if self.break_loop:
                    self.train_finished.emit(avg_err, False)
                    return
            avg_err /= self.size

            print('\t%d;\t%f;' % (i, avg_err))

            if self.stop and avg_err < self.stop:
                break

        self.train_finished.emit(avg_err, True)

    def interrupt(self):
        self.break_loop = True
