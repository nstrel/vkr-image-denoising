import numpy as np
import theano as th
import theano.tensor as T
from collections import OrderedDict


def sgd(grads, params, learning_rate=0.0025):
    """
    Обновление параметров сети по алгоритму градиентного спуска
    x = x-lr*grad(x)

    :param cost: функция цены ошибки
    :param params: параметры сети
    :return:
    """
    updates = [(param, param - learning_rate * grad)
               for param, grad in zip(params, grads)]
    return updates


def momentum(grads, params, learning_rate=0.0025, momentum=0.9):
    """
    Оптимизация: момент
    1. v_t = m*v_{t-1} + lr*grad(x)
    2. x_t = x_{t-1} - v_t

    :param grads:
    :param params:
    :param learning_rate:
    :param momentum:
    :return:
    """

    # x = x - lr*grad(x)
    updates = sgd(grads, params, learning_rate)

    updates = OrderedDict(updates)

    for param in params:
        value = param.get_value(borrow=True)

        velocity = th.shared(
            np.zeros(value.shape, dtype=value.dtype),
            broadcastable=param.broadcastable
        )

        # v_t = m*v_{t-1} + lr*grad(x) = m*v_{t-1} + (x - lr*grad(x)) - x
        # x_t = x_{t-1} - v_t = x_{t-1} - m*v_{t-1} - lr*grad(x)
        x = momentum * velocity + updates[param]
        updates[velocity] = x - param
        updates[param] = x

    return updates


def adadelta(grads, params, rho=0.85, epsilon=1e-6):
    """
    Оптимизация отсюда:
    http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
    Алгоритм:
        1. Вычислить градиент g_t
        2. Аккумулировать градиент E[g^2]_t=rho*E[g^2]_{t-1}+(1-rho)*g_t^2
        3. Вычислить dx_t=-sqrt(E[dx^2]_{t-1})/sqrt(E[g^2]_t)*gt
        4. Аккумулировать dx E[dx^2]_t=rho*E[dx^2]_{t-1}+(1-rho)*dx_t^2
        5. Обновить x_{t+1}=x_t+dx_t
    :param cost: функция цены ошибки
    :param params: параметры сети

    в adadelta learning rate - это
    sqrt(E[dx^2]_{t-1})/sqrt(E[g^2]_t),
    поэтому параметра learning_rate нет

    :param rho: learning decay - иными словами момент инерции
    :param epsilon: константа, близкая к 0; нужна, чтобы не делить на ноль
    :return: updates для функции
    """

    updates = OrderedDict()

    one = T.constant(1)

    for param, grad in zip(params, grads):
        # value: np.ndarray, представляющий W или b
        value = param.get_value(borrow=True)
        # accu: аккумулятор градиента
        accu = th.shared(
            np.zeros(value.shape, dtype=value.dtype),
            broadcastable=param.broadcastable
        )
        # delta_accu: аккумулятор dx
        delta_accu = th.shared(
            np.zeros(value.shape, dtype=value.dtype),
            broadcastable=param.broadcastable
        )

        # Для вычисления dx_t используется E[dx^2]_{t-1}, E[g^2]_t
        # Архитектура theano.function такова, что updates использует
        # для обновления значения с предыдущей итерации, например:
        # x_new = x_old + t
        # y_new = x_old + y_old
        # Поэтому нужна дополнительная переменная
        accu_new = rho * accu + (one - rho) * grad ** 2
        updates[accu] = accu_new

        # update - это learning_rate*grad
        update = (grad * T.sqrt(delta_accu + epsilon) /
                  T.sqrt(accu_new + epsilon))
        updates[param] = param - update

        # Дополнительная переменная, которая
        # вводится по тем же причинам, что и accu_new
        delta_accu_new = rho * delta_accu + (one - rho) * update ** 2
        updates[delta_accu] = delta_accu_new

    return updates
