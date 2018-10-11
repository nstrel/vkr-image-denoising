from sys import maxsize
from os.path import abspath
from shutil import rmtree
from PIL import Image, ImageFilter
import skimage as ski
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage.filters.rank import median
from util import *


def create_training_set(source, destination,
                        start, stop, num,
                        scale_sz=None, crop_sz=None):
    """Функция для генерации нового тренировочного сета"""

    source = abspath(source)
    destination = abspath_with_sep(destination)
    dir_xs = destination + 'x'
    dir_ys = destination + 'y'

    # сурс должен существовать
    if not os.path.exists(source):
        raise FileExistsError

    # если целевой папки нет, создать ее
    if not os.path.exists(destination):
        os.makedirs(destination)
    else:
        # иначе надо удалить предыдущие выборки
        if os.path.exists(dir_xs):
            rmtree(dir_xs)
        if os.path.exists(dir_ys):
            rmtree(dir_ys)

    os.makedirs(dir_xs)
    os.makedirs(dir_ys)

    # составить список имен изображений в сурс директории
    fnames = get_image_names(source)

    if scale_sz and crop_sz:
        mod_function = scale_and_split
        params = [scale_sz, crop_sz]
    elif scale_sz:
        mod_function = scale
        params = [scale_sz]
    elif crop_sz:
        mod_function = split
        params = [crop_sz]
    else:
        mod_function = identity
        params = []

    if start > stop:
        start, stop = stop, start

    t_xs = []
    t_ys = []
    i = 0
    # открыв каждое изображение из этого списка
    for fname in fnames:
        with Image.open(open(fname, 'rb')) as img:
            if not img.mode == 'RGB':
                img = img.convert('RGB')

            # применить преобразование к изображению
            mod_images = mod_function(img, *params)

            # для каждого изображения полученного в результате преобразования
            for y in mod_images:
                # добавить шум к изображению
                arr_y = image_to_array(y)
                for level in np.linspace(start, stop, num):
                    arr_x = add_noise(arr_y, level)

                    # сохранить
                    x = array_to_image(arr_x)
                    y.save("%s/%d_%s.jpg" % (dir_ys, i, 'y'))
                    x.save("%s/%d_%s.jpg" % (dir_xs, i, 'x'))

                    t_x = array_to_tensor(arr_x, 5)
                    t_y = array_to_tensor(arr_y, 5)
                    t_xs.append(t_x)
                    t_ys.append(t_y)

                    i += 1
    set_x = np.concatenate((t_xs), axis=0)
    set_y = np.concatenate((t_ys), axis=0)

    return (set_x, set_y)


def load_training_set(path):
    """Функция для загрузки тренировочного сета"""

    names_x, names_y = get_training_set_names(path)

    t_xs = []
    t_ys = []

    for name_x, name_y in zip(names_x, names_y):
        with Image.open(open(name_x, 'rb')) as x, \
                Image.open(open(name_y, 'rb')) as y:
            t_x = array_to_tensor(image_to_array(x), 5)
            t_y = array_to_tensor(image_to_array(y), 5)
            t_xs.append(t_x)
            t_ys.append(t_y)

    # Если папки пустые, массивы будут пустые и будет брошена ошибка
    # ValueError: need at least one array to concatenate
    set_x = np.concatenate((t_xs), axis=0)
    set_y = np.concatenate((t_ys), axis=0)

    return (set_x, set_y)


def compare_images(expected, got):
    if isinstance(expected, PIL.Image.Image):
        expected = image_to_array(expected)
    if isinstance(got, PIL.Image.Image):
        got = image_to_array(got)

    res = np.mean((expected - got) ** 2) / 2
    return res


def check_training_set(path, filter: PIL.ImageFilter):
    names_x, names_y = get_training_set_names(path)

    mean_err = 0
    min_err = maxsize
    max_err = 0

    for name_x, name_y in zip(names_x, names_y):
        with Image.open(open(name_x, 'rb')) as x, \
                Image.open(open(name_y, 'rb')) as y:

            x = x.filter(filter)
            x = image_to_array(x)
            y = image_to_array(y)

            mse = compare_images(y, x)

            if mse < min_err:
                min_err = mse
            if mse > max_err:
                max_err = mse

            mean_err += mse

    if len(names_x):
        mean_err /= len(names_x)

    return (mean_err, min_err, max_err)



def check_with_ski(path):
    names_x, names_y = get_training_set_names(path)

    mean_err = 0
    min_err = maxsize
    max_err = 0

    count = len(names_x)

    for i, (name_x, name_y) in \
        enumerate(zip(names_x, names_y)):
        with Image.open(open(name_x, 'rb')) as x, \
                Image.open(open(name_y, 'rb')) as y:


            noisy = image_to_array(x)
            clean = image_to_array(y)

            processed = denoise_wavelet(
                noisy,
                multichannel=True,
                convert2ycbcr=True
            )

            mse = compare_images(clean, processed)
            print('%d/%d %s' % (i, count, str(mse)))

            if mse < min_err:
                min_err = mse
            if mse > max_err:
                max_err = mse

            mean_err += mse

    if len(names_x):
        mean_err /= len(names_x)

    return (mean_err, min_err, max_err)


if __name__ == '__main__':
    # median_filter = PIL.ImageFilter.MedianFilter(5)
    # res = check_training_set('./dst100', median_filter)
    # print(res)

    #dst100 (0.0079289057485199162, 2.0744955112957128e-08, 0.054502577096688494)
    '''res = check_with_ski(
        './dst100',
        denoise_wavelet,
        multichannel=True,
        convert2ycbcr=True
    )'''

    print(res)
