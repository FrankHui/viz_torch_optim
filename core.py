# !/Applications/anaconda/envs/4PyCharm/bin/python3.4
# -*- coding: utf-8 -*-
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from torch import optim

from problem import *

FIG = plt.figure()


class TrajectoryAnimation(animation.FuncAnimation):
    """轨迹动画"""

    def __init__(self, trace_dict, fig=None, ax=None, frames=None,
                 interval=60, repeat_delay=5, blit=True, **kwargs):

        self.trace_dict = trace_dict
        self.labels = list(trace_dict.keys())

        # 获取当前fig和ax
        if fig is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()
        else:
            if ax is None:
                ax = fig.gca()

        if frames is None:
            frames = max(trace.shape[0] for trace in trace_dict.values())

        self.lines = [ax.plot([], [], label=label, lw=1)[0] for label in trace_dict]
        self.points = [ax.plot([], [], 'o', color=line.get_color())[0] for line in self.lines]

        super(TrajectoryAnimation, self).__init__(fig, self.animate, init_func=self.init_anim,
                                                  frames=frames, interval=interval, blit=blit,
                                                  repeat_delay=repeat_delay, **kwargs)

    def init_anim(self):
        for line, point in zip(self.lines, self.points):
            line.set_data([], [])
            point.set_data([], [])
        return self.lines + self.points

    def animate(self, i):
        """动画函数，应该是一帧调用一次这个函数"""
        for line, point, trace in zip(self.lines, self.points, self.trace_dict.values()):
            line.set_data(*trace.T[:2, :i])
            point.set_data(*trace.T[:2, i - 1:i])
        return self.lines + self.points


class TrajectoryAnimation3D(animation.FuncAnimation):

    def __init__(self, trace_dict, fig=None, ax=None, frames=None,
                 interval=60, repeat_delay=5, blit=True, **kwargs):

        self.trace_dict = trace_dict
        self.labels = list(trace_dict.keys())

        if fig is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()
        else:
            if ax is None:
                ax = fig.gca()

        if frames is None:
            frames = max(trace.shape[0] for trace in trace_dict.values())

        self.lines = [ax.plot([], [], [], label=label, lw=1)[0] for label in trace_dict]

        super(TrajectoryAnimation3D, self).__init__(fig, self.animate, init_func=self.init_anim,
                                                    frames=frames, interval=interval, blit=blit,
                                                    repeat_delay=repeat_delay, **kwargs)

    def init_anim(self):
        for line in self.lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return self.lines

    def animate(self, i):
        for line, trace in zip(self.lines, self.trace_dict.values()):
            line.set_data(*trace.T[:2, :i])
            # line.set_3d_properties(trace[:, 2][:i])
            line.set_3d_properties(trace.T[2, :i])
        return self.lines


class Problem:
    def __init__(self, f, start, minima, bound, z_limit, grid_step, train_step, fig, optimizer_fn_dict):
        """
        将二元函数的梯度下降可视化方法集成在这个类中
        :param f: 函数
        :param start: 起点
        :param minima: 最小点
        :param bound: 边界，形如[[x_min, x_max], [y_min, y_max]]
        :param z_limit: Z轴的界限
        :param grid_step: 画等高线时，点的密度
        :param train_step: 梯度下降的步数
        :param fig: plt.fig实例
        :param optimizer_fn_dict: torch.optim子类的列表
        """
        self.f = f
        self.start = torch.tensor(start, dtype=torch.float)
        self.minima = torch.tensor(minima)
        [[self.x_min, self.x_max], [self.y_min, self.y_max]] = bound
        self.bound = bound
        self.z_limit = z_limit
        self.grid_step = grid_step
        self.train_step = train_step
        self.optimizer_fn_dict = optimizer_fn_dict
        self.trace_dict = {}  # 梯度下降轨迹, 每条trace的shape = 3 × step
        self.x, self.y, self.z = self.calc_surface()
        self.fig = fig

    def train(self):
        """梯度下降过程，将轨迹记录下来"""
        for name, optimizer_fn in self.optimizer_fn_dict.items():
            param = torch.nn.Parameter(self.start.clone(), requires_grad=True)
            optimizer = optimizer_fn([param])
            trace = []
            for _ in range(self.train_step):
                # 清空梯度
                optimizer.zero_grad()

                # 记录坐标
                loss = self.f(param)
                x, y = param.data.numpy().tolist()
                z = loss.data.numpy()
                trace.append([x, y, z])

                # 梯度下降
                loss.backward()
                optimizer.step()
            trace = np.asarray(trace)
            self.trace_dict[name] = trace

    def calc_surface(self):
        """计算函数的二维面"""
        x, y = np.meshgrid(np.arange(self.x_min, self.x_max, step=self.grid_step),
                           np.arange(self.y_min, self.y_max, step=self.grid_step))
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        z = self.f([x, y])
        x, y, z = x.numpy(), y.numpy(), z.numpy()
        return x, y, z

    def show_trace_2d(self, show_trace=True, save=True):
        """展示二维的等高线 & 梯度下降的轨迹"""
        ax = plt.gca()

        # 画出平面等高线
        # ax.contour(self.x, self.y, self.z, colors='#1f77b4')
        ax.contour(self.x, self.y, self.z, levels=np.logspace(0, np.log(self.z.max()) // 2, 55), norm=LogNorm(),
                   cmap='rainbow', alpha=0.5, linewidths=.3)
        # plt.colorbar(cm)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim((self.x_min, self.x_max))
        ax.set_ylim((self.y_min, self.y_max))

        # 起点和最低点
        ax.plot([self.start[0]], [self.start[1]], marker='+')
        ax.plot([self.minima[0]], [self.minima[1]], marker='*')

        # 画出二维梯度下降的轨迹
        if show_trace:
            for name, trace in self.trace_dict.items():
                coors = [i[:2] for i in trace]
                ax.plot(*zip(*coors), label=name)

        if save:
            plt.savefig(f"{self.f.__name__}_二维静态图.jpg")

    def show_trace_3d(self, show_trace=True, save=True):
        """展示三维面 & 梯度下降的轨迹"""
        ax = plt.gca()
        if not isinstance(ax, Axes3D):
            ax = plt.axes(projection='3d')

        # 画出surface
        if self.z_limit:
            drop_index = np.bitwise_or(self.z < self.z_limit[0], self.z > self.z_limit[1])
            self.x[drop_index] = np.nan
            self.y[drop_index] = np.nan
            self.z[drop_index] = np.nan
        ax.plot_surface(
            self.x,
            self.y,
            self.z,
            rstride=1,
            cstride=1,
            # edgecolor='none',
            alpha=.3,
            cmap='rainbow'
        )
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim((self.x_min, self.x_max))
        ax.set_ylim((self.y_min, self.y_max))

        # 起点和最小点
        ax.plot([self.start[0]], [self.start[1]], [self.f(self.start)], marker='+')
        ax.plot([self.minima[0]], [self.minima[1]], [self.f(self.minima)], marker='*')

        # 画出梯度下降的路径
        if show_trace:
            for name, trace in self.trace_dict.items():
                x, y, z = trace[:, 0], trace[:, 1], trace[:, 2]
                line = ax.plot(x, y, z, label=name)[0]
                line.set_3d_properties(z)
        if save:
            plt.savefig(f"{self.f.__name__}_三维静态图.jpg")

    def animation_2d(self, show_static_trace=False):
        self.show_trace_2d(show_static_trace)  # 默认不展示固定的轨迹

        ax = plt.gca()
        fps = 30
        save_file = f"{self.f.__name__}_二维动态图.gif"
        anim = TrajectoryAnimation(self.trace_dict, ax=ax, interval=1000 // fps)
        anim.save(save_file, fps=fps, writer='PillowWriter')

    def animation_3d(self, show_static_trace=False):
        self.show_trace_3d(show_static_trace)  # 默认不展示固定的轨迹

        ax = plt.gca()
        fps = 30
        save_file = f"{self.f.__name__}_三维动态图.gif"
        anim = TrajectoryAnimation3D(self.trace_dict, ax=ax, interval=1000 // fps)
        anim.save(save_file, fps=fps, writer='PillowWriter')


def fn(x_y):
    x, y = x_y
    return x ** 2 + 2 * y ** 2


def main():
    # lr = .1
    optimizer_fn_dict = dict(
        # Good high dimensional optimizers sometimes do poorly in low D spaces,
        # so we will lower the LR on simple optimisers
        SGD=partial(optim.SGD, lr=3e-3),
        # momentum=partial(optim.SGD, lr=lr / 80, momentum=0.9, nesterov=False, dampening=0),
        # momentum_dampen=partial(optim.SGD, lr=lr / 80, momentum=0.9, nesterov=False, dampening=0.3),
        # nesterov=partial(optim.SGD, lr=lr / 80, momentum=0.9, nesterov=True, dampening=0),
        # nesterov_decay=partial(optim.SGD, lr=lr / 80, momentum=0.9, nesterov=True, weight_decay=1e-4, dampening=0),
        # need larger lr's sometimes
        # Adadelta=optim.Adadelta,
        # Adagrad=partial(optim.Adagrad, lr=2),
        # Adamax=partial(optim.Adamax, lr=2),
        # RMSprop=partial(optim.RMSprop, lr=.1),
        # RMSprop1=partial(optim.RMSprop, lr=.03),
        # RMSprop2=partial(optim.RMSprop, lr=.05),
        Adam=partial(optim.Adam, lr=1),
    )
    # problem = Problem(fn, [-4, -2], [[-5, 5], [-4, 4]], 0.1, 100, FIG, optimizer_fn_dict)
    problem = Problem(
        f=beales,
        start=[2, 1.7],
        minima=[3., 0.5],
        bound=[[-4.5, 4.5], [-4.5, 5.5]],
        z_limit=[0, 300],  # 如果bound和grid_step选择不恰当，使得Z轴过长，而梯度下降轨迹在Z轴太短，会以为梯度下降在一个平面内，故取Z轴截断
        grid_step=0.1,
        train_step=1000,
        fig=FIG,
        optimizer_fn_dict=optimizer_fn_dict)
    problem.train()
    problem.show_trace_2d()
    # problem.show_trace_3d()
    # problem.animation_2d()
    # problem.animation_3d()
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()
