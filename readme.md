# 可视化pytorch的优化器
这个工程学习了 [原工程](https://github.com/3springs/viz_torch_optim) 的可视化的方法，主要目的是对比学习
1. 同一优化器在不同参数下的表现
2. 同一函数下在不同优化器下的梯度下降表现

深度学习是对loss进行梯度下降，从而得到神经网络的参数。本工程以二元函数为例子，提供了二维、三维的静态、动态轨迹图的方法。
使用者仅需要在core.py中main函数里定义好Problem实例，选择好想要对比的优化器，再选择输出什么轨迹图。如下：
```python
def main():
    problem = Problem(
        f=beales,  # 二元函数
        start=[2, 1.7],  # 优化起点
        minima=[3., 0.5],  # 函数的极小值坐标
        bound=[[-3, 3], [-3, 3]],  # 函数自变量的绘图范围
        z_limit=[0, 300],  # 函数值的绘图范围
        grid_step=0.1,  # 网格的密度
        train_step=10,  # 训练步数
        fig=FIG,  # plt.figure实例
        optimizer_fn_dict=optimizer_fn_dict  # 优化器字典
      ) 
    optimizer_fn_dict = dict(
        SGD=partial(optim.SGD, lr=.4),  # SGD
        Adadelta=optim.Adadelta,  # Adadelta
        Adam=partial(optim.Adam, lr=1  # Adam
    )

    # 选择轨迹图
    problem.train()
    problem.show_trace_2d()  # 二维静态轨迹
    # problem.show_trace_3d()  # 三维静态轨迹
    # problem.animation_2d()  # 二维动态轨迹
    # problem.animation_3d()  # 三维动态轨迹
    plt.legend(loc='upper right')
    plt.show()
```
最后运行
 ```
python core.py
```
即可。
Problem.py中存放着一些有趣的二元函数，你也可以使用最简单的二元二次函数。

目前图还做得不太美观，欢迎提建议。