import sys

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

# установка локали для корректного преобразования float-ов
QLocale.setDefault(QLocale.c())

from six.moves import cPickle
from PIL import ImageFilter

from util import *
from net import *

from training_set import create_training_set, load_training_set
from train_thread import TrainThread
from gui.main_window import Window


class AEInterface(QObject):
    info = pyqtSignal(str)
    changed = pyqtSignal(list)
    iteration = pyqtSignal(float)
    interrupt = pyqtSignal()
    train_finished = pyqtSignal()

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.reset()

    def reset(self):
        self.ae_file = None
        self.ae_shape = None
        self.ae_params = None
        self.ae = None
        self.ae_err = 0.0
        self.changed.emit(self.status_())

    @pyqtSlot(list)
    def new(self, params):
        self.info.emit(
            'Initializing AE'
            # 'Initializing AE with params: %s' % (str(params))
        )

        sz, ps = params
        self.ae_shape = (1, 1, 3, sz, sz)
        self.ae_params = ps
        self.ae = Autoencoder(self.ae_shape, self.ae_params)
        self.ae_file = None
        self.ae_err = None

        self.info.emit(
            'Initialized AE'
            # 'Initialized AE with params: %s' % (str(params))
        )
        self.changed.emit(self.status_())

    def status_(self):
        if self.ae:
            status = True
            fields = {
                'File': self.ae_file,
                'Input shape': self.ae_shape,
                'Layer info': self.ae_params,
                'Error': self.ae_err
            }
        else:
            status = False
            fields = {}
        return [status, fields]

    @pyqtSlot(str)
    def load(self, fname):
        with open(fname, 'rb') as file:
            self.info.emit('loading AE from %s' % (fname))
            try:
                objects = cPickle.load(file)
            except Exception as e:
                # очень интересно, почему не работает
                self.info.emit(
                    'failed to load AE from %s (%s)'
                    % (fname, str(e))
                )
                return

            if len(objects) != 4:
                raise NotImplementedError

            shp, ps, ae, err = objects
            self.ae_file = fname
            self.ae_shape = shp
            self.ae_params = ps
            self.ae = ae
            self.ae_err = err

            self.info.emit('loaded ae from %s' % (fname))
            self.changed.emit(self.status_())

    @pyqtSlot(str)
    def save(self, fname):
        if not fname:
            dialog = QFileDialog()
            res = dialog.getSaveFileName()
            if res and res[0]:
                fname = res[0]
                self.ae_file = fname
                self.changed.emit(self.status_())
            else:
                return

        with open(fname, 'wb') as file:
            self.info.emit('saving AE to %s' % (fname))
            cPickle.dump([
                self.ae_shape,
                self.ae_params,
                self.ae,
                self.ae_err
            ],
                file, protocol=cPickle.HIGHEST_PROTOCOL
            )
            self.info.emit('Saved AE to %s' % (fname))

    @pyqtSlot(dict)
    def train(self, set, optimizer='None', constants=None,
              epochs=5, batch_size=16, stop=None):

        if not self.ae:
            self.info.emit('Failed to train: AE is not initialized')
        else:
            self.info.emit('Starting to train AE')

            if optimizer == 'None':
                optimizer = sgd
            elif optimizer == 'Momentum':
                optimizer = momentum
            elif optimizer == 'Adadelta':
                optimizer = adadelta
            else:
                optimizer = sgd

            train = self.ae.make_train_function(
                set, batch_size, optimizer, constants
            )

            # количество mini-batch-ей
            k = get_shape(set[0]) // batch_size

            train_thread = TrainThread()

            train_thread.set_params(
                train=train,
                size=k,
                epochs=epochs,
                err=stop
            )

            # Проброс сигналов - хранить тред в поле не стоит, он каждый раз создается новый
            train_thread.iteration[float].connect(
                self.iteration
            )
            self.interrupt.connect(
                train_thread.interrupt
            )
            train_thread.train_finished[float, bool].connect(
                self.on_train_finished
            )

            train_thread.start()

    def on_train_finished(self, err, ok):

        if ok:
            self.info.emit('Trained AE successfully')
        else:
            self.info.emit('Stopped training AE')

        self.ae_err = err
        self.train_finished.emit()
        self.changed.emit(self.status_())

    @pyqtSlot()
    def close(self):
        self.info.emit('closing AE')
        self.reset()
        self.info.emit('closed AE')


class Worker(QObject):
    fname_changed = pyqtSignal()
    image_changed = pyqtSignal(Image.Image)
    has_no_work = pyqtSignal(bool)

    set_changed = pyqtSignal(list)

    info = pyqtSignal(str)

    def __init__(self, parent=None):
        QObject.__init__(self, parent)

        self.fname = ''
        self.image = None

        self.filter = ImageFilter.MedianFilter()
        self.ae = AEInterface()
        self.reset_set()

    def reset_set(self):
        self.training_set = None

    @pyqtSlot()
    def close_set(self):
        self.reset_set()
        self.set_changed.emit([False, {}])

    @pyqtSlot(str)
    def on_load_set(self, path):
        if path:
            set = load_training_set(path)
            if set:
                self.training_set = make_shared(set)
                self.set_changed.emit(
                    [True,
                     {
                         'Directory': path,
                         'Count': set[0].shape[0],
                         'Image size': set[0].shape[-1],
                     }
                     ]
                )
                self.info[str].emit(
                    'loaded training set'
                )

    @pyqtSlot(dict)
    def on_new_set(self, params):
        set = create_training_set(**params)
        self.training_set = make_shared(set)
        self.set_changed.emit(
            [True,
             {
                 'Directory': params['destination'],
                 'Count': set[0].shape[0],
                 'Image size': set[0].shape[-1],
             }
             ]
        )
        self.info[str].emit(
            'new training set'
        )

    @pyqtSlot(dict)
    def on_train(self, params):
        self.ae.train(self.training_set, **params)

    @pyqtSlot(str)
    def set_fname(self, fname):
        self.fname = fname
        self.fname_changed.emit()

    @pyqtSlot(Image.Image)
    def set_image(self, image):
        self.image = image
        self.image_changed.emit(image)

    @pyqtSlot()
    def on_apply_ae(self):
        self.has_no_work.emit(False)
        with Image.open(open(self.fname, 'rb')) as pimage:
            if not pimage.mode == 'RGB':
                pimage = pimage.convert('RGB')
            images = self.crop(pimage)
            processed = self.apply_ae(images)
            pimage = self.glue(processed)
            self.set_image(pimage)
            self.has_no_work.emit(True)

    @pyqtSlot()
    def on_apply_filter(self):
        self.has_no_work.emit(False)
        with Image.open(open(self.fname, 'rb')) as pimage:
            if not pimage.mode == 'RGB':
                pimage = pimage.convert('RGB')
            pimage = pimage.filter(self.filter)
            self.set_image(pimage)
            self.has_no_work.emit(True)

    def crop(self, image: Image.Image):
        """
        Разбивает image на куски size x size.
        Полученные куски предназначены для обработки автокодировщиком.
        :param image: изображение для разбиения
        :return: двумерный массив кусков типа PIL.Image
        """
        size = self.ae.ae_shape[-1]
        w, h = image.size
        w_pieces, w_border = divmod(w, size)
        h_pieces, h_border = divmod(h, size)

        cropped = image  # .crop((w_border // 2, h_border // 2,
        #      w - w_border // 2, h - h_border // 2))

        res = []
        for i in range(h_pieces):
            line = []
            for j in range(w_pieces):
                img = cropped.crop((size * j, size * i,
                                    size * (j + 1), size * (i + 1)))
                line.append(img)
            res.append(line)

        return res

    def glue(self, images: list):
        """
        Склеивает куски, полученные в результате работы crop в изображение
        :param images: двумерный список кусков типа PIL.Image
        :return: склеенное изображение
        """

        # Список обязательно должен быть двумерным и непустым
        assert (len(images) > 0 and len(images[0]) > 0)

        # print('glue', len(images), len(images[0]))

        size = images[0][0].size[0]
        w = (len(images[0])) * size
        h = (len(images)) * size

        res = Image.new('RGB', (w, h))

        for i, img_row in enumerate(images):
            for j, img in enumerate(img_row):
                res.paste(img, (j * size, i * size))
        return res

    def apply_filter(self, image: Image.Image):
        """
        Применяет указанный фильтр к изображению типа PIL.Image
        :param image: изображение
        :return: фильтрованное изображение
        """
        return image.filter(self.filter)

    def apply_ae(self, images: list):
        """
        Применяет автокодировщик к каждому изображению из списка
        :param images: список изображений размера,
                       принимаемого автокодировщиком
        :return: список фильтрованных изображений
        """
        res = []
        for row in images:
            t = []
            for image in row:
                img = image_to_array(image)
                img = array_to_tensor(img, 5)
                t.append(img)
            t = np.concatenate((t), axis=0)
            t = self.ae.ae.func(t)

            imgs = batch_tensor_to_array(t)
            r = []
            for img, image in zip(imgs, row):
                img = array_to_image(img)
                w, h = image.size
                img = img.crop((1, 1, w - 1, h - 1))
                image.paste(img, (1, 1))
                r.append(image)
            res.append(r)
        return res


class Main(QObject):
    info = pyqtSignal(str)

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.gui = Window()
        self.worker = Worker()

        self.worker.info[str].connect(self.gui.on_status)
        self.worker.ae.info[str].connect(self.gui.on_status)
        self.info[str].connect(self.gui.on_status)

        self.connect_ae_with_menu()
        self.connect_worker_with_menu()
        self.connect_worker_with_apply()
        self.connect_worker_with_info()

        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.start()

    def connect_ae_with_menu(self):
        self.gui.new_ae[list].connect(
            lambda x: self.worker.ae.new(x)
        )
        self.gui.load_ae[str].connect(
            lambda x: self.worker.ae.load(x)
        )
        self.gui.save_ae[str].connect(
            lambda x: self.worker.ae.save(x)
        )
        self.gui.close_ae.connect(
            self.worker.ae.close
        )
        self.gui.train.train[dict].connect(
            lambda x: self.worker.on_train(x)
        )
        self.worker.ae.iteration[float].connect(
            lambda x: self.gui.train.progress.update_figure(x)
        )

    def connect_worker_with_menu(self):
        self.gui.new_set[dict].connect(
            self.worker.on_new_set
        )
        self.gui.load_set[str].connect(
            self.worker.on_load_set
        )
        self.gui.close_set.connect(
            self.worker.close_set
        )

    def connect_worker_with_apply(self):
        self.gui.apply.fname_changed[str].connect(
            lambda x: self.worker.set_fname(x)
        )
        self.gui.apply.apply.clicked.connect(
            self.worker.on_apply_ae
        )
        self.worker.image_changed.connect(
            self.gui.apply.set_processed
        )
        self.worker.has_no_work[bool].connect(
            self.gui.apply.apply.setEnabled
        )

    def connect_worker_with_info(self):
        self.worker.ae.changed[list].connect(
            lambda x: self.gui.info.net_status.update_(*x)
        )
        self.worker.set_changed[list].connect(
            lambda x: self.gui.info.set_status.update_(*x)
        )


if __name__ == '__main__':
    app = QApplication(sys.argv)
    exe = Main(app)
    sys.exit(app.exec_())
