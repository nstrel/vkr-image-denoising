from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PIL import Image, ImageQt
from util import image_to_array

class Apply(QWidget):
    fname_changed = pyqtSignal(str)
    applying = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.title = 'Apply'
        self.min = 250
        self.max = 500

        self.initUI()

    def initUI(self):
        """
        Здесь инициализируется Canvas для оригинального и
        обработанного изображений.
        """

        self.setWindowTitle(self.title)

        self.original = QLabel()
        self.processed = QLabel()

        # Изображения располагаются по горизонтали
        img_lo = QHBoxLayout()
        img_lo.addWidget(self.original)
        img_lo.addWidget(self.processed)

        # Кнопка для выбора изображения для обработки
        self.open = QPushButton('open')
        self.open.pressed.connect(self.on_open)

        # Кнопка для обработки загруженного изображения
        # Возможно, стоит объединить выбор и обработку?
        self.apply = QPushButton('apply')

        btn_lo = QHBoxLayout()
        btn_lo.addStretch()
        btn_lo.addWidget(self.open)
        btn_lo.addWidget(self.apply)

        lo = QVBoxLayout()
        lo.addLayout(img_lo)
        lo.addLayout(btn_lo)
        self.setLayout(lo)

        self.show()

    def pil_to_pix(self, pil: Image.Image):
        if not pil.mode == 'RGBA':
            pil = pil.convert('RGBA')
        qimage = ImageQt.ImageQt(pil)
        if qimage.size().width() < self.min:
            qimage = qimage.scaled(
                self.min, self.min, Qt.KeepAspectRatio
            )
        if qimage.size().width() > self.max:
            qimage = qimage.scaled(
                self.max, self.max, Qt.KeepAspectRatio
            )
        pix = QPixmap.fromImage(qimage)
        return pix

    def on_open(self):
        fname = QFileDialog.getOpenFileName()
        if fname and fname[0]:
            with Image.open(open(fname[0], 'rb')) as image:
                image = self.pil_to_pix(image)
                self.original.setPixmap(image)
                self.fname_changed.emit(fname[0])

    def set_processed(self, image: Image.Image):
        image = self.pil_to_pix(image)
        self.processed.setPixmap(image)

    def on_select(self):
        """
        При нажатии на select появляется диалог для выбора имени файла
        Если файл выбран, и это изображение, он отображается в self.original
        и сохраняется для дальнейшей обработки автокодировщиком в self.image_original
        """
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.selectNameFilter("Images (*.png *.jpg)")
        fname = dialog.getOpenFileName()

        if fname:
            with Image.open(open(fname[0], 'rb')) as img:
                if not img.mode == 'RGB':
                    img = img.convert('RGB')
                img_arr = image_to_array(img)
                self.image_original = img_arr
                self.original.plot(img_arr)

    def on_apply(self):
        """
        Здесь к self.image_original должен быть применен автокодер и
        результат должен быть загружен в self.processed
        """
        self.applying.emit()
