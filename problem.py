# -*- coding: utf-8 -*-

"""A valley"""
from torch import sin, cos, abs, stack, sqrt


def madsen(tensor):
    """Madsen function (1981)."""
    x1, x2 = tensor
    r = x1 ** 2 + x2 ** 2 + x1 * x2 + sin(x1) + cos(x2)  # + abs(noise(x1,x2))
    return r


def dmadsen(tensor):
    x1, x2 = tensor
    dx1 = 2. * x1 + x2 + cos(x1)
    dx2 = 2. * x2 + x1 - sin(x2)
    return stack([dx1, dx2], 1)[0]


def schaffern4(tensor):
    """https://www.sfu.ca/~ssurjano/schaffer4.html"""
    x1, x2 = tensor

    r = 0.5 + (cos(sin(abs(x1 ** 2 - x2 ** 2))) - 0.5) / (1 + 0.001 * (x1 ** 2 + x2 ** 2)) ** 2
    return r * 100


def dschaffern4(tensor):
    x1, x2 = tensor
    dx1 = -2 * (x1 ** 2 - x2 ** 2) * x1 * cos(abs(-x1 ** 2 + x2 ** 2)) * sin(
        sin(abs(-x1 ** 2 + x2 ** 2))) / (
                  (0.001 * x1 ** 2 + 0.001 * x2 ** 2 + 1.0) ** 2 * abs(-x1 ** 2 + x2 ** 2)) - 0.004 * x1 * (
                  cos(sin(abs(-x1 ** 2 + x2 ** 2))) - 0.5) / (
                  0.001 * x1 ** 2 + 0.001 * x2 ** 2 + 1.0) ** 3
    dx2 = 2 * (x1 ** 2 - x2 ** 2) * x2 * cos(abs(-x1 ** 2 + x2 ** 2)) * sin(
        sin(abs(-x1 ** 2 + x2 ** 2))) / (
                  (0.001 * x1 ** 2 + 0.001 * x2 ** 2 + 1.0) ** 2 * abs(-x1 ** 2 + x2 ** 2)) - 0.004 * x2 * (
                  cos(sin(abs(-x1 ** 2 + x2 ** 2))) - 0.5) / (
                  0.001 * x1 ** 2 + 0.001 * x2 ** 2 + 1.0) ** 3
    return stack([dx1, dx2], 1)[0]


def rosenbrock(tensor):
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def drosenbrock(tensor):
    x, y = tensor
    dx = 400 * (x ** 2 - y) + 2 * x - 2
    dy = -200 * x ** 2 + 200 * y
    return stack([dx, dy], 1)[0]


def eggholder(tensor):
    """https://www.sfu.ca/~ssurjano/camel6.html"""
    x1, x2 = tensor

    r = - (x2 + 47) * sin(sqrt(abs(x2 + x1 / 2 + 47))) - x1 * sin(sqrt(abs(x1 - (x2 + 47))))
    return r


def deggholder(tensor):
    x1, x2 = tensor
    dx1 = -1 / 8 * (x1 + 2 * x2 + 94) * (x2 + 47) * cos(sqrt(abs(1 / 2 * x1 + x2 + 47))) / abs(
        1 / 2 * x1 + x2 + 47) ** (3 / 2) - 1 / 2 * (x1 - x2 - 47) * x1 * cos(
        sqrt(abs(-x1 + x2 + 47))) / abs(-x1 + x2 + 47) ** (3 / 2) - sin(sqrt(abs(-x1 + x2 + 47)))
    dx2 = -1 / 4 * (x1 + 2 * x2 + 94) * (x2 + 47) * cos(sqrt(abs(1 / 2 * x1 + x2 + 47))) / abs(
        1 / 2 * x1 + x2 + 47) ** (3 / 2) + 1 / 2 * (x1 - x2 - 47) * x1 * cos(
        sqrt(abs(-x1 + x2 + 47))) / abs(-x1 + x2 + 47) ** (3 / 2) - sin(
        sqrt(abs(1 / 2 * x1 + x2 + 47)))
    return stack([dx1, dx2], 1)[0]


def six_humped_camel_back(tensor):
    """https://www.sfu.ca/~ssurjano/camel6.html"""
    x1, x2 = tensor

    r = (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2 + \
        x1 * x2 + \
        (-4 + 4 * x2 ** 2) * x2 ** 2
    return r


def dsix_humped_camel_back(tensor):
    x1, x2 = tensor
    dx1 = -0.333333333333333 * (-4.00000000000000 * x1 ** 3 + 12.6000000000000 * x1) * x1 ** 2 - 0.666666666666667 * (
            -1.00000000000000 * x1 ** 4 + 6.30000000000000 *
            x1 ** 2 - 12.0000000000000) * x1 + x2
    dx2 = 8 * x2 ** 3 + 8 * (x2 ** 2 - 1) * x2 + x1
    return stack([dx1, dx2], 1)[0]


def beales(tensor):
    """Beales function, like a valley"""
    x, y = tensor
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2
