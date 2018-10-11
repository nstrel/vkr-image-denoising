import numpy as np
import theano as th
import theano.tensor as T
import pylab as pl
from six.moves import cPickle

import training_set
from autoencoder.optimizators import *
from autoencoder.layers import *

from util import norm_colour

# --------------------------------------------------------------------
# Автокодировщик

class Autoencoder(object):
    """ Класс автокодировщика
    Состоит из многослойных кодера и декодера,
    параметров, функций
    """

    def __init__(self, image_shp, layer_params):
        """
        Для инициализации требуется указать размер входных изображений
        и параметры для каждого сверточного слоя.

        Для каждого слоя свертки автоматически генерируется
        слой транспонированной свертки как обратный.

        Пример:
        Пусть cl_1, cl_2 - сверточные слои. Тогда будет построена
        сеть X->cl_1->cl_2->dcl_2->dcl_1->Y, где dcl_i = (cl_i)^{-1}.
        В сети используются skip-connections: если X проходит через cl_i,
        то на выходе получается X'. Тогда на вход dcl_i подается результат,
        полученный на предыдущем слое + X'.
        Такая архитектура помогает решить проблему исчезающего градиента.

        :param image_shp: размер входных изображений (None, 1, 3, h, w)
                          Если 0-ой элемент != None, то он все равно заменится
        :param layer_params: список параметров для conv слоев
            - число и размер фильтра (count, size)
            - border 'valid' | 'full' | (d,h,w)
            - subsample (d,h,w)
            - dilation пока не используется
        """
        # размер входных изображений хранится отдельно:
        # это нужно для валидации загруженного объекта Nnet из .save файла
        self.input_shape = image_shp

        # заполним значениями по умолчанию None-ы
        # layer_params = self.complete_layer_params(layer_params)

        # параметры, которые уже известны
        # self.filter_shps, self.border_shps, \
        # self.subsamples, self.dilations = list(zip(*layer_params))

        # эти параметры будут вычисляться
        self.image_shps = []
        self.c_layers = []
        self.dc_layers = []

        # символьные переменные
        input = T.tensor5('input', dtype=th.config.floatX)
        expected = T.tensor5('expected', dtype=th.config.floatX)
        self.x = input
        self.y = expected

        # инициализация слоев свертки
        x = input
        i_shp = (None, *image_shp[1:])
        # for (count, size), border, subsample, dilation in layer_params:
        for param in layer_params:
            print(str(param))
            self.image_shps.append(i_shp)
            count, size = param['filter_shape']
            f_shp = (count, i_shp[1], i_shp[2], size, size)
            param['filter_shape'] = f_shp
            param['image_shape'] = i_shp
            # cl = ConvLayer3D(np.random, x, f_shp, i_shp, border, subsample)
            cl = ConvLayer3D(np.random, x, **param)
            self.c_layers.append(cl)
            x = cl.output
            i_shp = cl.output_shp()

        # инициализация слоев транспонированной свертки
        # for i, (((count, size), border, subsample, dilation), i_shp, c_l) \
        #        in enumerate(zip(reversed(layer_params),
        #                         reversed(self.image_shps),
        #                         reversed(self.c_layers))):
        for i, (param, c_l) in enumerate(zip(reversed(layer_params),
                                             reversed(self.c_layers))):
            # f_shp = (count, i_shp[1], i_shp[2], size, size)
            # dcl = DeconvLayer3D(np.random, x, f_shp, i_shp, border, subsample)
            dcl = DeconvLayer3D(np.random, x, **param)
            self.dc_layers.append(dcl)
            x = dcl.output + c_l.input \
                if i < len(layer_params) - 1 else dcl.output

        # результат применения сети к случайному изображению
        result = T.nnet.relu(x)
        self.func = th.function(
            [input], result
        )

        cost = T.square(result - expected).mean() / 2
        self.cost = cost

        # результат применения сети к изображению,
        # для которого известен желаемый результат, и оценка
        self.check = th.function(
            [input, expected],
            [result, cost],
        )

        params = [l.W for l in self.c_layers + self.dc_layers] \
                 + [l.b for l in self.c_layers + self.dc_layers]
        self.params = params

        grads = T.grad(cost, params)
        self.grads = grads

        self.train_stop = False

        print('inited')

    @staticmethod
    def complete_layer_params(params):
        """
        Дополняет параметры слоев до значений по-умолчанию
        :param params: список (filter_shape, border_mode, subsample, dilation)
                       как минимум, должен быть указан filter_shape;
                       остальные параметры могут быть опущены или
                       не иметь значения
        :return: список (filter_shape, border_mode, subsample, dilation)
        """
        params = [t + (None,) * (4 - len(t)) for t in params]
        return [
            (f_shp,
             border if border else 'valid',
             subsample if subsample else (1, 1, 1),
             dilation if dilation else (1, 1, 1))
            for f_shp, border, subsample, dilation in params
        ]

    def make_train_function(self, set, batch, optimizer, constants=None):

        i = T.iscalar('index')
        updates = optimizer(
            self.grads,
            self.params,
            **constants if constants else {}
        )
        train = th.function(
            [i],
            self.cost,
            updates=updates,
            givens={
                self.x: set[0][i * batch:(i + 1) * batch],
                self.y: set[1][i * batch:(i + 1) * batch]
            }
        )
        return train


# --------------------------------------------------------------------
# Вспомогательные функции


def get_shape(arr):
    return T.shape(arr)[0].eval()


def train_for_N_epochs(net, set, optimizer, constants=None,
                       epochs=5, batch_size=16, stop=None):
    """
    Обучение сети
    :param set: набор входных и выходных сигналов
    :param epochs: число эпох
    :param batch_size: размер mimi-batch
    :param stop: останов по достижению заданой точности
    :return: None
    """

    if optimizer == 'None':
        optimizer = sgd
    elif optimizer == 'Momentum':
        optimizer = momentum
    elif optimizer == 'Adadelta':
        optimizer = adadelta

    train = net.make_train_function(set, batch_size, optimizer, constants)

    # количество mini-batch-ей
    k = T.shape(set[0])[0].eval() // batch_size
    indices = list(range(k))

    print('\tepoch;\terr;')

    for i in range(epochs):
        avg_err = 0

        # перемешивание пар x : y
        np.random.shuffle(indices)

        for index in indices:
            err = train(index)
            avg_err += abs(err)

        print('\t%d;\t%f;' % (i, avg_err / k))

        if net.train_stop or (stop and avg_err / k < stop):
            break


def compare(arrs, scale=None):
    plots = [np.squeeze(x).transpose(1, 2, 0) for x in arrs]

    if scale:
        plots = [norm_colour(x, scale) for x in arrs]

    k = len(plots)
    for i, x in enumerate(plots):
        pl.subplot(1, k, i + 1)
        pl.axis('off')
        pl.imshow(x)
    pl.show()


def load_ae(file, input_shape, params):
    try:
        print('loading %s' % (file))

        # файл лучше закрыть до того, как начнутся косяки
        fil = open(file, 'rb')
        objects = cPickle.load(fil)
        fil.close()

        # проверка, что загружена сеть с указаными параметрами
        shp, ps, nn = objects
        assert (shp == input_shape)
        for p, param in zip(ps, params):
            assert (p == param)

        return nn
    except:
        print('failed to load %s, compiling' % (file))
        return Autoencoder(input_shape, params)


def save_ae(file, shp, ps, nn):
    fil = open(file, 'wb')
    cPickle.dump([shp, ps, nn], fil, protocol=cPickle.HIGHEST_PROTOCOL)
    fil.close()


def make_shared(data, borrow=True):
    x, y = data

    shared_x = th.shared(x, borrow=borrow)
    shared_y = th.shared(y, borrow=borrow)

    return (shared_x, shared_y)


if __name__ == "__main__":
    image_shp = (None, 1, 3, 50, 50)
    params = [{
        'filter_shape': (10, 5)
    }]
    ae = Autoencoder(image_shp, params)

    ts = training_set.load_training_set('./dst')
    ts = make_shared(ts)

    train_for_N_epochs(ae, ts, 'Momentum')
