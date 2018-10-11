import sys

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

# установка локали для корректного преобразования float-ов
QLocale.setDefault(QLocale.c())

from gui.info import Info
from gui.train import Train
from gui.apply import Apply
from gui.dialogs.new_ae_dialog import newAE
from gui.dialogs.new_set_dialog import newSet


class Window(QMainWindow):
    """Класс главного окна приложения"""

    # Сигналы
    new_ae = pyqtSignal(list)
    load_ae = pyqtSignal(str)
    save_ae = pyqtSignal(str)
    close_ae = pyqtSignal()

    new_set = pyqtSignal(dict)
    load_set = pyqtSignal(str)
    close_set = pyqtSignal()

    status = pyqtSignal(str)

    def __init__(self, parent=None):  # flags=None пройдет?
        QMainWindow.__init__(self, parent)

        # Вкладки
        self.info = Info()
        self.train = Train()
        self.apply = Apply()

        tabs = QTabWidget()
        tabs.addTab(self.info, 'info')
        tabs.addTab(self.train, 'train')
        tabs.addTab(self.apply, 'apply')
        self.setCentralWidget(tabs)

        self.init_menu()
        self.init_connections()

        self.show()

    def init_menu(self):
        menubar = self.menuBar()

        # Инициализация действий с автокодировщиком
        aeMenu = menubar.addMenu('&Encoder')

        # Инициализация
        self.newAE = QAction('&New...', self)
        self.newAE.setShortcut('Ctrl+N')
        self.newAE.triggered.connect(
            self.on_new_ae
        )
        aeMenu.addAction(self.newAE)
        # загрузка
        self.loadAE = QAction('&Load...', self)
        self.loadAE.setShortcut('Ctrl+L')
        self.loadAE.triggered.connect(
            self.on_load_ae
        )
        aeMenu.addAction(self.loadAE)
        # сохранение
        self.saveAE = QAction('&Save', self)
        self.saveAE.setShortcut('Ctrl+S')
        self.saveAE.triggered.connect(
            self.on_save_ae
        )
        aeMenu.addAction(self.saveAE)
        # сохранение как
        self.saveasAE = QAction('&Save as...', self)
        self.saveasAE.setShortcut('Ctrl+Shift+S')
        self.saveasAE.triggered.connect(
            self.on_saveas_ae
        )
        aeMenu.addAction(self.saveasAE)
        # закрытие
        self.closeAE = QAction('&Close', self)
        self.closeAE.setShortcut('Ctrl+W')
        self.closeAE.triggered.connect(
            self.on_close_ae
        )
        aeMenu.addAction(self.closeAE)

        setMenu = menubar.addMenu('&Set')
        # генерация тренировочной выборки
        self.newSet = QAction('&New...', self)
        self.newSet.triggered.connect(
            self.on_new_set
        )
        setMenu.addAction(self.newSet)
        # загрузка тренировочной выборки
        self.loadSet = QAction('&Load...', self)
        self.loadSet.triggered.connect(
            self.on_load_set
        )
        setMenu.addAction(self.loadSet)
        # закрытие тренировочной выборки
        self.closeSet = QAction('&Close', self)
        self.closeSet.triggered.connect(
            self.on_close_set
        )
        setMenu.addAction(self.closeSet)

    @pyqtSlot()
    def on_new_ae(self):
        """
        Вызывается диалог для получения параметров и
        параметры передаются в сигнале new_ae(list)
        """
        dialog = newAE()
        if dialog.exec_() == QDialog.Accepted:
            params = dialog.value()
            self.new_ae.emit(params)

    @pyqtSlot()
    def on_load_ae(self):
        """
        Вызывается диалог для получения существующего
        имени файла и имя передается в сигнале load_ae(str)
        """
        dialog = QFileDialog()
        fname = dialog.getOpenFileName()
        if fname and fname[0]:
            self.load_ae.emit(fname[0])

    @pyqtSlot()
    def on_save_ae(self):
        """
        Передается сигнал save_ae()
        """
        self.save_ae.emit('')

    @pyqtSlot()
    def on_saveas_ae(self):
        """
        Вызывается диалог для получения имени файла
        и имя передается в сигнале save_ae(str)
        """
        dialog = QFileDialog()
        fname = dialog.getSaveFileName()
        if fname and fname[0]:
            self.save_ae.emit(fname[0])

    @pyqtSlot()
    def on_close_ae(self):
        """
        Передается сигнал close_ae()
        """
        self.close_ae.emit()

    @pyqtSlot()
    def on_new_set(self):
        """
        Вызывается диалог для получения параметров и
        параметры передаются в сигнале new_set(list)
        """
        dialog = newSet()
        if dialog.exec_() == QDialog.Accepted:
            params = dialog.value()
            self.new_set.emit(params)

    @pyqtSlot()
    def on_load_set(self):
        """
        Вызывается диалог для получения пути и
        путь передается в сигнале load_set(str)
        """
        dialog = QFileDialog()
        path = dialog.getExistingDirectory()
        if path:
            self.load_set.emit(path)

    @pyqtSlot()
    def on_close_set(self):
        """
        Передается сигнал close_set()
        """
        self.close_set.emit()

    @pyqtSlot(str)
    def on_status(self, status):
        print(status)
        self.statusBar().showMessage(status)

    def init_connections(self):
        self.status[str].connect(
            self.on_status
        )


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Window()
    sys.exit(app.exec_())
