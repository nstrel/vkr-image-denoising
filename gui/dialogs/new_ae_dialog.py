import typing
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from gui.dialogs.layer_dialog import LayerDialog

class LayerListModel(QAbstractListModel):
    data_changed = pyqtSignal(QModelIndex, QModelIndex)

    def __init__(self, layers, parent=None):
        QAbstractListModel.__init__(self, parent)
        self.layers = layers

    def rowCount(self, parent: QModelIndex = ...):
        return len(self.layers)

    def data(self, index: QModelIndex, role: int = ...):
        if role == Qt.DisplayRole:
            layer = self.layers[index.row()]
            info = '\n'.join('{}: {}'.format(k, v) for k, v in layer.items())
            return QVariant(info)
        elif role == Qt.EditRole:
            return self.layers[index.row()]
        else:
            return QVariant()

    def setData(self, index: QModelIndex, value: typing.Any, role: int = ...):
        if role == Qt.EditRole:
            self.layers[index.row()] = value
            self.data_changed.emit(index, index)
            return True
        return False

    def removeRows(self, row: int, count: int, parent: QModelIndex = ...):
        if row < 0 or row > self.rowCount():
            return

        self.beginRemoveRows(QModelIndex(), row, row + count - 1)
        while count != 0:
            del self.layers[row]
            count -= 1
        self.endRemoveRows()

    # def flags(self, index: QModelIndex):
    #    if not index.isValid():
    #        return Qt.NoItemFlags
    #    return Qt.ItemIsEditable | Qt.ItemIsEnabled

    def addLayer(self, layer):
        self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount())
        self.layers.append(layer)
        self.endInsertRows()


class LayerListView(QListView):
    def __init__(self, parent=None):
        QListView.__init__(self, parent)
        self.setAlternatingRowColors(True)
        self.setContextMenuPolicy(Qt.ActionsContextMenu)

        delete = QAction('Delete', self)
        delete.triggered.connect(self.on_delete)
        self.addAction(delete)

        edit = QAction('Edit', self)
        edit.triggered.connect(self.on_edit)
        self.addAction(edit)

        self.setWordWrap(True)
        self.setTextElideMode(Qt.ElideNone)

    @pyqtSlot()
    def on_delete(self):
        self.model().removeRows(self.currentIndex().row(), 1)

    @pyqtSlot()
    def on_edit(self):
        item = self.model().data(self.currentIndex(), Qt.EditRole)
        editor = LayerDialog()
        editor.setValue(item)
        res = editor.exec_()
        if res == QDialog.Accepted:
            item = editor.value()
            self.model().setData(self.currentIndex(), item, Qt.EditRole)

    @pyqtSlot()
    def on_add(self):
        editor = LayerDialog()
        res = editor.exec_()
        if res == QDialog.Accepted:
            self.model().addLayer(editor.value())


class newAE(QDialog):
    """Диалог для инициализации автокодировщика"""

    params = pyqtSignal(list)

    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('New Autoencoder')
        self.initUI()

    def reset(self):

        # Сброс размера изображения
        self.img_size.clear()

        # Сброс слоев в списке
        m = LayerListModel([])
        self.layers.setModel(m)

    def initUI(self):

        self.finished[int].connect(self.on_finished)

        # Размер входного изображения
        # Ограничения:
        # - размер входного изображения должен совпадать
        #   с размером изображений обучающей выборки
        # - входное изображение квадратное, т.е. здесь
        #   задаются его высота и ширина
        img_size = QLineEdit()
        img_size.setValidator(QIntValidator(1, 500))
        self.img_size = img_size

        # Список слоев
        m = LayerListModel([])
        layers = LayerListView()
        layers.setModel(m)
        self.layers = layers

        # Кнопка для добавления нового слоя
        add = QPushButton('add new layer')
        add.clicked.connect(self.layers.on_add)

        # Кнопки для подтверждения инициализации
        # Может, добавить reset для сброса?
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.on_accepted)
        buttons.rejected.connect(self.reject)

        # Layout, в который будут добавляться слои
        self.layer_lo = QVBoxLayout()

        lo = QVBoxLayout(self)
        lo.addWidget(QLabel('Image size:'))
        lo.addWidget(self.img_size)
        lo.addWidget(QLabel('Layers:'))
        lo.addWidget(self.layers)
        lo.addWidget(add)
        lo.addWidget(buttons)
        lo.addStretch()

        self.setLayout(lo)

    def value(self):
        """
        Возвращает параметры для инициализации
        """

        img_size = self.img_size.text()
        if not img_size:
            img_size = self.img_size.validator().bottom()
        else:
            img_size = int(img_size)

        m = self.layers.model()
        layer_params = m.layers

        return [img_size, layer_params]

    def on_finished(self, res):
        if (res == QDialog.Accepted):
            value = self.value()
            self.params.emit(value)

    def on_accepted(self):
        """
        Диалог завершился успешно, если добавлен хотя бы один слой,
        иначе выводится сообщение об ошибке
        """
        if self.layers.model().rowCount() > 0:
            if not self.img_size.text():
                QMessageBox(
                    QMessageBox.Warning,
                    'Bad input',
                    'Image size must be set.',
                    QMessageBox.Ok
                ).exec_()
            else:
                self.accept()
        else:
            res = QMessageBox(
                QMessageBox.Warning,
                'Bad input',
                'Nnet must have at least one conv layer.\n'
                + 'Nothing will be created.',
                QMessageBox.Cancel | QMessageBox.Ok
            ).exec_()
            if res == QMessageBox.Ok:
                self.reject()
