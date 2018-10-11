import os
import numpy as np
import PIL
from PIL import Image


def scale_array(arr, interval=(0, 1), domain=(0, 255)):
    """
    Приводит значения массива к указанному интервалу
    """

    (m, M), (a, b) = domain, interval
    return a + (b - a) * (arr - m) / (M - m)


def norm_colour(arr, interval):
    domain = np.min(arr), np.max(arr)
    return scale_array(arr, interval, domain)


def image_to_array(img: PIL.Image, normalization='unit', dtype='float32'):
    """
    Из изображения построить массив (h, w, d).
    Значения пикселей нормализуются, если указано.
    """

    res = np.asarray(img, dtype=dtype)
    if normalization == 'unit':  # для сокращения
        res /= 255
    elif isinstance(normalization, tuple):  # интервал (a, b)
        res = scale_array(res, normalization)

    return res


def array_to_image(arr: np.ndarray, normalization='unit'):
    """
    Из массива получить изображение.
    Параметр normalization должен быть тем же,
    что и при image_to_array
    """

    if normalization == 'unit':
        res = arr * 256
    elif isinstance(normalization, tuple):
        res = scale_array(arr, (0, 255), normalization)
    else:
        res = arr
    return Image.fromarray(np.uint8(res))


def array_to_tensor(arr: np.ndarray, dim):
    """
    Из массива (h, w, d) получить массив (1, 1, d, h, w)
    или (1, d, h, w). Последние массивы обрабатываются
    автокодировщиком, поэтому такое название.
    """
    arr = np.transpose(arr, (2, 0, 1))
    return arr.reshape((1,) * (dim - 3) + arr.shape)


def tensor_to_array(arr: np.ndarray):
    """
    Из тензора получить массив. Если тензор (N, d, h, w)
    или (N, M, d, h, w) имеет N > 1 или M > 1,
    будет работать некорректно.
    """
    return np.squeeze(arr).transpose(1, 2, 0)


def batch_tensor_to_array(arr: np.ndarray):
    """
    Функция для случая (N, d, h, w) или (N, 1, d, h, w) с N > 1
    """
    if arr.shape[0] > 1:
        arrs = np.split(arr, arr.shape[0])
        return [tensor_to_array(arr) for arr in arrs]
    return [tensor_to_array(arr)]


def current_directory():
    return os.path.dirname(os.path.abspath(__file__))


def abspath_with_sep(path):
    return os.path.abspath(path) + os.sep


def get_image_names(path):
    '''Возвращает имена всех файлов изображений в директории
    :param path: string
    :return: list(string)
    '''
    ext = ('.jpg', '.jpeg', '.png', '.ppm', '.eps', '.bmp', '.tif')
    path = abspath_with_sep(path)
    fnames = []
    for (d, dn, fn) in os.walk(path):
        fnames = [path + f for f in fn if f.lower().endswith(ext)]
        break
    return fnames


def load_image_as_array(fname):
    with Image.open(open(fname, 'rb')) as img:
        if not img.mode == 'RGB':
            img = img.convert('RGB')
        res = image_to_array(img)
    return res


def get_training_set_names(path):
    # из пути получить путь до х и у
    path = abspath_with_sep(path)
    path_x = path + 'x'
    path_y = path + 'y'

    if not os.path.isdir(path_x) \
            or not os.path.isdir(path_y):
        print('Error loading: no x, y directories')
        return None

    # достать списки файлов из каждой директории
    names_x = get_image_names(path_x)
    names_y = get_image_names(path_y)

    return (names_x, names_y)


def add_noise(arr: np.ndarray, level=0.1):
    return np.clip(arr + np.random.normal(0.0, level, arr.shape), 0.0, 1.0)


def add_impulse_noise(arr: np.ndarray, level=0.1):
    shp = (arr.shape[0], arr.shape[1], 1)
    s_and_p = np.random.uniform(0, 1, shp)
    res = arr
    res[np.argwhere(s_and_p < level)] = 0.0
    res[np.argwhere(s_and_p > 1 - level)] = 1.0
    return res


def scale(img: PIL.Image, size=200):
    """ Приводит img к размеру (size,size) """

    center = (img.size[0] // 2, img.size[1] // 2)
    d = min(center)

    res = img.crop((center[0] - d, center[1] - d,
                    center[0] + d, center[1] + d))

    return [res.resize((size, size))]


def split(img: PIL.Image, size=200):
    """ Режет img на квадраты размера (size,size) """

    return [
        img.crop((size * i, size * j,
                  size * (i + 1), size * (j + 1)))
        for i in range(img.size[0] // size)
        for j in range(img.size[1] // size)
    ]


def scale_and_split(img: PIL.Image, scale_sz, split_sz):
    """ Меняет размер изображения и режет на квадраты """

    scaled = scale(img, scale_sz)  # массив с единственным изображением
    return split(scaled[0], split_sz)


def identity(img: PIL.Image):
    return [img]
