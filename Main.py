import pandas as pd

from tabulate import tabulate
import os.path
from sympy import diff, symbols, cos, sin, sqrt, lambdify, sympify, tan, exp, pi, solve, ln, integrals, integrate
from check import typeofread, read, is_number
import matplotlib.pyplot as plt
import numpy as np

x, y, xv, yv = symbols('x y xv yv')


def vidyravnenia(num):
    if num == 1:
        return 1 + 3 * y + 2 * x ** 2
    if num == 2:
        return y * tan(x) - x ** 2
    if num == 3:
        return 2 * y + 3 * sin(2 * x) - cos(x)


def vidtochresh(num, x0, y0):
    if num == 1:
        C = (y0 + 2 / 3 * x0 ** 2 + 4 / 9 * x0 + 13 / 27) / exp(3 * x0)
        return C * exp(3 * x) - 2 / 3 * x ** 2 - 4 / 9 * x - 13 / 27
    if num == 2:
        C = (y0 + 2 * x0) * np.cos(x0) + (x0 ** 2 - 2) * sin(x0)
        return (C - (x ** 2 - 2) * sin(x)) / cos(x) - 2 * x
    if num == 3:
        C = (y0 + 3 / 4 * (np.sin(2 * x0) + cos(2 * x0)) + 1 / 5 * (sin(x0) - 2 * cos(x0))) / exp(2 * x0)
        return - 3 / 4 * (sin(2 * x) + cos(2 * x)) - 1 / 5 * (sin(x) - 2 * cos(x)) + C * exp(2 * x)


typeoffunction = int(typeofread("Выберете функцию (введите номер):\n" +
                                "1." + str(vidyravnenia(1)) + "\n2." + str(vidyravnenia(2)) + "\n3."
                                + str(vidyravnenia(3)) + "\n", ["1", "2", "3"]))
fun = vidyravnenia(typeoffunction)
func = lambdify([x, y], fun)
x0 = read("Введите начальное значение x_0:")
y0 = read("Введите начальное значение y_0:")
xn = read("Введите конечное значение x_n:")
h = read("Введите шаг для дифференцирования:")
e = read("Введите точность измерений:")
tofun = vidtochresh(typeoffunction, x0, y0)
typeofr = int(typeofread("Выберете метод решения (введите номер):\n" +
                         "1. Метод Эйлера\n2. Усовершенствованный метод Эйлера\n"
                         "3. Милна\n4. Все методы\n", ["1", "2", "3", "4"]))


def el(h):
    X = [x0]
    Y = [y0]
    for i in range(1, int((xn - x0) / h + 1)):
        x = X[i - 1] + h
        y = Y[i - 1]
        f = func(x - h, y)
        X.append(x)
        Y.append(y + h * f)
    return X, Y


H = h
if typeofr in {1, 4}:
    t = True
    while t:
        X1, Y1 = el(h)
        X2, Y2 = el(h / 2)
        if abs(Y1[-1] - Y2[-1]) <= e:
            head = ["x_i", "y_i"]
            EL1 = np.column_stack([X1, Y1])
            r=[EL1[0]]
            for i in range(int(H//h),len(EL1),int(H//h)):
                r.append(EL1[i])
            print(tabulate(r, head, tablefmt="github", floatfmt=".5f"))
            print("шаг =", h)
            print()
            plt.plot(X1, Y1, color="green", label="Метод Эйлера")
            t = False
        h /= 2


def elm(h):
    X = [x0]
    Y = [y0]
    for i in range(1, int((xn - x0) / h + 1)):
        x = X[i - 1] + h
        y = Y[i - 1]
        f = func(x - h, y)
        X.append(x)
        Y.append(y + h / 2 * (f + func(x - h, y + h * func(x - 2 * h, y))))
    return X, Y


h = H
if typeofr in {2, 4}:
    t = True
    while t:
        X1, Y1 = elm(h)
        X2, Y2 = elm(h / 2)
        print(Y1[-1],h)
        if abs(Y1[-1] - Y2[-1]) / 3 <= e:
            head = ["x_i", "y_i"]
            ELm1 = np.column_stack([X1, Y1])
            r = [ELm1[0]]
            for i in range(int(H // h), len(ELm1), int(H // h)):
                r.append(ELm1[i])
            print(tabulate(r, head, tablefmt="github", floatfmt=".5f"))
            print("шаг =", h)
            print()
            plt.plot(X1, Y1, color="blue", label="Усовершенствованный метод Эйлера")
            t = False
        h /= 2
h = H

if typeofr in {3, 4}:
    tofunf = lambdify(x, tofun)
    y1 = [0] * int((xn - x0) / h + 1)
    y1[0] = y0
    x = x0
    for i in range(1, 4):
        k1 = h * func(x, y1[i - 1])
        k2 = h * func(x + h / 2, y1[i - 1] + k1 / 2)
        k3 = h * func(x + h / 2, y1[i - 1] + k2 / 2)
        k4 = h * func(x + h, y1[i - 1] + k3)
        y1[i] = y1[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x += h

    yp = [0] * int((xn - x0) / h + 1)
    yk = [0] * int((xn - x0) / h + 1)
    x = [0] * int((xn - x0) / h + 1)
    x[0] = x0
    for i in range(int((xn - x0) / h + 1)):
        if i < 4:
            yp[i] = y1[i]
            yk[i] = y1[i]
        else:
            yp[i] = yp[i - 4] + 4 * h / 3 * (
                    2 * func(x[i - 3], yp[i - 3]) - func(x[i - 2], yp[i - 2]) + 2 * func(x[i - 1],
                                                                                         yp[i - 1]))
            yk[i] = yk[i - 2] + h / 3 * (
                    func(x[i - 2], yk[i - 2]) + 4 * func(x[i - 1], yk[i - 1]) + func(x[i - 1] + h, yp[i]))
            while abs(yk[i] - yp[i]) > e:
                yp[i] = yk[i]
                yk[i] = yk[i - 2] + h / 3 * (
                            func(x[i - 2], yk[i - 2]) + 4 * func(x[i - 1], yk[i - 1]) + func(x[i - 1] + h, yp[i]))
        if i > 0:
            x[i] = x[i - 1] + h
    x1 = x.copy()
    x= symbols("x")
    to = [0]*len(x1)
    de = [0]*len(x1)
    for i in range(len(x1)):
        to[i] = tofunf(x1[i])
        de[i]=abs(yk[i]-tofunf(x1[i]))
    r = [x1, yk, to, de]
    head = ["x_i", "y","Точное y"]
    print(tabulate(np.transpose(r), head, tablefmt="github", floatfmt=".5f"))
    x, y = symbols("x y")
    plt.plot(x1, yk, color="red", label="Милна")
x2132 = np.arange(x0,xn, 0.0001)
tofunf = lambdify(x, tofun)
plt.plot(x2132, tofunf(x2132), label="Точное решение")
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()
