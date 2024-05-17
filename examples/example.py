

import numpy as np
from modules.simulation.hazard import EvolutionBase,HazardBase
import time
from modules.simulation.mesh import LocalMeshScene
import matplotlib
# matplotlib.use('TkAgg')

# <editor-fold desc="Test functions">

def EvolutionsTestCase_01():
    print("----- Single point test: without evolution functions -----")
    EvolutionBaseObj = EvolutionBase(id="01",
                                     name="EvolutionTest01",
                                     class_name="Hazardbase",
                                     init_value=0,
                                     init_grad=10,
                                     init_dgrad=-3,
                                     min_value=0,
                                     max_value=100,
                                     total_sum=2000,
                                     )
    # Define a custom evolution function
    EvolutionBaseObj.time_evolution_function.params = [1, 1]

    def update_callback(Obj: EvolutionBase):
        """A test for update callback """
        # Obj.time_evolution_function.params = [Obj.get_value()] # PASS
        Obj.time_evolution_function.params = [(Obj.total_sum - Obj.current_sum), Obj.grad]
        Obj.current_sum = Obj.current_sum + Obj.get_value()
        pass

    EvolutionBaseObj.update_callback = update_callback

    def Ev_func1(args):
        # return args[-1] if args[0] > 0 else 0
        return 2.5

    EvolutionBaseObj.time_evolution_function.add_functions(Ev_func1)


    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig1, ax1 = plt.subplots(1, 1)
    ax1.set_xlabel('时间(分钟)')
    ax1.set_ylabel('燃烧功率(兆瓦)')

    t = np.array(list(range(0, 100)))
    x, y = [], []

    def init():
        x, y = [], []
        im = plt.plot(x, y, "r-")
        EvolutionBaseObj.disable_time_devolution()
        return im

    def update_point(step):
        x.append(step)
        y.append(EvolutionBaseObj.update_in_temperal())
        if step ==50:
            EvolutionBaseObj.enable_time_devolution()
        im = plt.plot(x, y, "r-")
        return im

    ani = FuncAnimation(fig1, update_point, frames=t,
                        init_func=init, interval=300, repeat=False)

    # ani.save(r"D:\Project\EmergencyDeductionEngine\docs\figs\evolution_base_functions_0.gif")
    plt.show()


def EvolutionsTestCase_02():
    print("----- Single point test: with evolution functions -----")
    EvolutionBaseObj = EvolutionBase(id="01",
                                     name="EvolutionTest01",
                                     class_name="Hazardbase",
                                     init_value=0,
                                     init_grad=0.5,
                                     init_dgrad=0.1,
                                     init_spread=-0.01,
                                     init_dspread=-0.01,
                                     min_value=0,
                                     max_value=100,
                                     total_sum=1000,
                                     )
    # Define a custom evolution function
    EvolutionBaseObj.time_evolution_function.params = [1]
    # def Ev_func1(args):
    # """
    # Test: PASS
    # """
    #     # assuming that the args[0] is the grad
    #     return args[0]
    def update_callback(Obj: EvolutionBase):
        """A test for update callback """
        # Obj.time_evolution_function.params = [Obj.get_value()] # PASS
        Obj.time_evolution_function.params = [(Obj.total_sum - Obj.current_sum+Obj.get_value()/100)]
        Obj.current_sum = Obj.current_sum + Obj.get_value()
        pass

    EvolutionBaseObj.update_callback = update_callback

    def Ev_func1(args):
        return args[0]/100

    EvolutionBaseObj.time_evolution_function.add_functions(Ev_func1)


    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig1, ax1 = plt.subplots(1, 1)
    ax1.set_xlabel('时间(分钟)')
    ax1.set_ylabel('燃烧功率(兆瓦)')

    t = np.array(list(range(0, 35)))
    x, y = [], []

    def init():
        x, y = [], []
        im = plt.plot(x, y, "r-")
        return im

    def update_point(step):
        x.append(step)
        y.append(EvolutionBaseObj.update())
        im = plt.plot(x, y, "r-")
        return im

    ani = FuncAnimation(fig1, update_point, frames=t,
                        init_func=init, interval=200, repeat=False)

    # ani.save(r"D:\Project\EmergencyDeductionEngine\docs\figs\evolution_base_function.gif")
    plt.show()


def EvolutionsTestCase_03():
    print("----- Single point test: with devolution functions -----")
    EvolutionBaseObj = EvolutionBase(id="01",
                                     name="EvolutionTest01",
                                     class_name="Hazardbase",
                                     init_value=0,
                                     init_grad=0.5,
                                     init_dgrad=-0.01,
                                     init_spread=-0.01,
                                     init_dspread=-0.01,
                                     min_value=0,
                                     max_value=100,
                                     total_sum=2000,
                                     )
    # Define a custom evolution function
    EvolutionBaseObj.time_evolution_function.params = [1]

    def update_callback(Obj: EvolutionBase):
        """A test for update callback """
        # Obj.time_evolution_function.params = [Obj.get_value()] # PASS
        Obj.time_evolution_function.params = [(Obj.total_sum - Obj.current_sum)/10]
        Obj.current_sum = Obj.current_sum + Obj.get_value()
        pass

    EvolutionBaseObj.update_callback = update_callback

    def Ev_func1(args):
        return args[0]/100

    def Ev_func2(args):
        return args[0]/50

    EvolutionBaseObj.time_evolution_function.add_functions(Ev_func1)


    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig1, ax1 = plt.subplots(1, 1)
    ax1.set_xlabel('时间(分钟)')
    ax1.set_ylabel('燃烧功率(兆瓦)')

    t = np.array(list(range(0, 100)))
    x, y = [], []

    def init():
        x, y = [], []
        im = plt.plot(x, y, "r-")
        return im

    def update_point(step):
        x.append(step)
        y.append(EvolutionBaseObj.update())
        print("step:", step)
        if step == 10:
            EvolutionBaseObj.time_evolution_function.add_functions(Ev_func2)
            pass
        im = plt.plot(x, y, "r-")
        return im

    ani = FuncAnimation(fig1, update_point, frames=t,
                        init_func=init, interval=300, repeat=False)

    # ani.save(r"D:\Project\EmergencyDeductionEngine\docs\figs\evolution_base_function_multi.gif")
    plt.show()


def EvolutionsTestCase_04():
    print("----- Mesh points test: with time evolution functions -----")
    # init_value = np.zeros([10, 10])
    import random
    x, y = np.mgrid[-5:5:10j, -5:5:10j]
    sigma = 2
    z = np.round(np.array(1 / (2 * np.pi * (sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)))*1000, 3)
    init_value = z
    print(init_value)
    # init_value = np.array([[random.randint(0, 50) for j in range(0, 10)] for i in range(0, 10)])
    init_grad = np.ones([10, 10])*0.05
    init_dgrad = np.ones([10, 10])*-0.01
    init_spread = np.ones([10, 10])*-0.01
    init_dspread = np.ones([10, 10])*-0.01
    total_sum = np.ones([10, 10])*2000

    EvolutionBaseObj = EvolutionBase(id="01",
                                     name="EvolutionTest01",
                                     class_name="Hazardbase",
                                     init_value=init_value,
                                     init_grad=init_grad,
                                     init_dgrad=init_dgrad,
                                     init_spread=init_spread,
                                     init_dspread=init_dspread,
                                     min_value=0,
                                     max_value=100,
                                     total_sum=total_sum,
                                     area=[10, 10, 10]
                                     )
    # Define a custom evolution function
    EvolutionBaseObj.time_evolution_function.params = [np.zeros([10, 10]), np.zeros([10, 10])]        # init
    EvolutionBaseObj.set_mode(mode="mesh")

    def update_callback(Obj: EvolutionBase):
        """A test for update callback """
        # Obj.time_evolution_function.params = [Obj.get_value()] # PASS
        Obj.time_evolution_function.params = [(Obj.total_sum - Obj.current_sum)/10, Obj.grad]
        Obj.current_sum = Obj.current_sum + Obj.get_value()
        pass

    EvolutionBaseObj.update_callback = update_callback

    def Ev_func1(args):
        return args[0]/100

    def Ev_func2(args):
        return args[0]/50

    EvolutionBaseObj.time_evolution_function.add_functions(Ev_func1)


    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # fig1, ax1 = plt.subplots(1, 1)
    fig2 = plt.figure(num=2, figsize=(128, 108))
    # ax1.set_xlabel('时间(分钟)')
    # ax1.set_ylabel('燃烧功率(兆瓦)')
    x, y = [], []

    def Evolution_plot(retval: np.ndarray):
        plt.subplot(1, 2, 1)
        meshval = retval.reshape([10, 10])
        im = plt.imshow(meshval, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=110)
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar()
        plt.xticks(np.arange(0, 10, 10))  # fixed
        plt.yticks(np.arange(0, 10, 10))  # fixed
        cb.set_label('热功率 单位(MW)')
        plt.title('热功率空间分布图')

        ax1 = plt.subplot(1, 2, 2)
        im = plt.plot(x, y1, "r-")
        im = plt.plot(x, y2, "g-")
        im = plt.plot(x, y3, "b-")
        ax1.set_xlabel('时间(分钟)')
        ax1.set_ylabel('燃烧功率(兆瓦)')
        return im

    t = np.array(list(range(0, 60)))

    # ax1.set_xlim(0, np.max(t))
    # ax1.set_ylim(0, EvolutionBaseObj.max_value+10)

    x, y1, y2, y3 = [], [], [], []

    def init():
        x, y1, y2, y3 = [], [], [], []
        # im = plt.plot(x, y, "r-")
        retval = EvolutionBaseObj.update()
        # im = plt.imshow(retval)
        print(retval)
        return Evolution_plot(retval)

    def update_point(step):
        retval = EvolutionBaseObj.update()
        x.append(step)
        y1.append(retval[0][0])
        y2.append(retval[3][3])
        y3.append(retval[5][5])
        if step == 10:
            EvolutionBaseObj.time_evolution_function.add_functions(Ev_func2)
        print(retval)
        # im = plt.imshow(retval)

        fig2.savefig(r"D:\Project\EmergencyDeductionEngine\docs\figs\imgs\img_{:0>2d}.png".format(step))
        return Evolution_plot(retval)

    ani = FuncAnimation(fig2, update_point, frames=t,
                        init_func=init, interval=300, repeat=False)

    # ani.save(r"D:\Project\EmergencyDeductionEngine\docs\figs\evolution_base_functions_with_space.gif")
    # ani.save(r"D:\Project\EmergencyDeductionEngine\docs\figs\evolution_base_functions_with_space.mp4",  fps=30, extra_args=['-vcodec', 'libx264'])
    # ani.save(r"D:\Project\EmergencyDeductionEngine\docs\figs\evolution_base_functions_with_space.gif", writer='imagemagick')
    # ani.save(r"D:\Project\EmergencyDeductionEngine\docs\figs\evolution_base_functions_with_space.gif").to_jshtml()
    plt.show()


def EvolutionsTestCase_05():
    print("----- Mesh points test: with space evolution functions -----")
    # init_value = np.zeros([10, 10])
    # =============== init data ===============
    # import random
    # zero_bound = 2
    # x, y = np.mgrid[-5:5:11j, -5:5:11j]
    # sigma = 2
    # z = np.round(np.array(1 / (2 * np.pi * (sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)))*100, 3)
    # print(np.where(z < zero_bound))
    # z[np.where(z < zero_bound)] = 0
    # init_value = z
    # print(init_value, init_value.size)
    # ============== init data ===============
    minv, maxv, stride = -7, 3, 1
    x, y = np.meshgrid(range(minv, maxv, stride), range(minv, maxv, stride))
    xx, yy = x, y
    print(x, y)
    init_value = np.zeros([10, 10])
    init_value[7:8, 7:8] = 1
    # print(xx, yy)

    # Get the index of the center in matrix
    cx = np.unique(np.where(x == 0)[1])[0]
    cy = np.unique(np.where(y == 0)[0])[0]

    h_offset, v_offset, hv_offset = init_value.copy(), init_value.copy(), init_value.copy()

    h_offset[:, 0:cx] = h_offset[:, 1:cx + 1]
    h_offset[:, cx + 1:-1] = h_offset[:, cx:-2]
    # print(h_offset)

    v_offset[0:cy, :] = v_offset[1:cy + 1, :]
    v_offset[cy + 1:-1, :] = v_offset[cy:-2, :]
    # print(v_offset)

    hv_offset[:, 0:cx] = hv_offset[:, 1:cx + 1]
    hv_offset[:, cx + 1:-1] = hv_offset[:, cx:-2]
    hv_offset[0:cy, :] = hv_offset[1:cy + 1, :]
    hv_offset[cy + 1:-1, :] = hv_offset[cy:-2, :]
    # print(hv_offset)

    evolution_value = 0.25 * h_offset + 0.25 * v_offset + 0.5 * hv_offset
    print(evolution_value)


def EvolutionsTestCase_06():
    print("----- Mesh points test: space evolution functions -----")
    # =============== init data ===============
    minv, maxv, stride = -50, 50, 1
    x, y = np.meshgrid(range(minv, maxv, stride), range(minv, maxv, stride))
    xx, yy = x, y
    # print(x, y)
    init_value = np.zeros([100, 100])
    init_value[49:51, 49:51] = 50
    # print(init_value)
    init_grad = np.ones([100, 100]) * 0.05
    init_dgrad = np.ones([100, 100]) * -0.01
    init_spread = np.ones([100, 100]) * -0.01
    init_dspread = np.ones([100, 100]) * -0.01
    total_sum = np.ones([100, 100]) * 2000

    # Get the index of the center in matrix
    cx = np.unique(np.where(x == 0)[1])[0]
    cy = np.unique(np.where(y == 0)[0])[0]

    def space_evolution(value):
        h_offset, v_offset, hv_offset = value.copy(), value.copy(), value.copy()

        h_offset[:, 0:cx] = h_offset[:, 1:cx + 1]
        h_offset[:, cx + 1:-1] = h_offset[:, cx:-2]
        # print(h_offset)

        v_offset[0:cy, :] = v_offset[1:cy + 1, :]
        v_offset[cy + 1:-1, :] = v_offset[cy:-2, :]
        # print(v_offset)

        hv_offset[:, 0:cx] = hv_offset[:, 1:cx + 1]
        hv_offset[:, cx + 1:-1] = hv_offset[:, cx:-2]
        hv_offset[0:cy, :] = hv_offset[1:cy + 1, :]
        hv_offset[cy + 1:-1, :] = hv_offset[cy:-2, :]
        # print(hv_offset)

        evolution_value = 0.25 * h_offset + 0.25 * v_offset + 0.5 * hv_offset
        print(evolution_value)
        return evolution_value

    EvolutionBaseObj = EvolutionBase(id="01",
                                     name="EvolutionTest01",
                                     class_name="Hazardbase",
                                     init_value=init_value,
                                     init_grad=init_grad,
                                     init_dgrad=init_dgrad,
                                     init_spread=init_spread,
                                     init_dspread=init_dspread,
                                     min_value=0,
                                     max_value=100,
                                     total_sum=total_sum,
                                     area=[10, 10, 10]
                                     )
    # Define a custom evolution function
    EvolutionBaseObj.time_evolution_function.params = [np.zeros([10, 10]), np.zeros([10, 10])]  # init
    EvolutionBaseObj.set_mode(mode="mesh")

    def update_callback(Obj: EvolutionBase):
        """A test for update callback """
        # Obj.time_evolution_function.params = [Obj.get_value()] # PASS
        Obj.time_evolution_function.params = [(Obj.total_sum - Obj.current_sum) / 10, Obj.grad]
        Obj.current_sum = Obj.current_sum + Obj.get_value()
        pass

    EvolutionBaseObj.update_callback = update_callback

    def Ev_func1(args):
        return args[0] / 100

    def Ev_func2(args):
        return args[0] / 50

    EvolutionBaseObj.time_evolution_function.add_functions(Ev_func1)

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig2 = plt.figure(num=2, figsize=(128, 108))
    x, y = [], []

    def Evolution_plot(retval: np.ndarray):
        plt.subplot(1, 1, 1)
        meshval = retval.reshape([100, 100])
        im = plt.imshow(meshval, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=110)
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar()
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('热功率 单位(MW)')
        plt.title('热功率空间分布图')
        return im

    t = np.array(list(range(0, 60)))

    def init():
        pass

    def update_point(step):
        # retval = EvolutionBaseObj.update()
        retval = space_evolution(EvolutionBaseObj.get_value())
        EvolutionBaseObj.set_value(value=retval)
        # fig2.savefig(r"D:\Project\EmergencyDeductionEngine\docs\figs\imgs\img_{:0>2d}.png".format(step))
        return Evolution_plot(retval)

    ani = FuncAnimation(fig2, update_point, frames=t,
                        init_func=init, interval=300, repeat=False)

    # ani.save(r"D:\Project\EmergencyDeductionEngine\docs\figs\space_evolution.gif")

    plt.show()


def EvolutionsTestCase_07():
    print("----- Mesh points test: space evolution functions -----")
    # =============== init data ===============
    init_value = np.zeros([100, 100])
    init_value[49:51, 49:51] = 50
    # print(init_value)
    init_grad = np.ones([100, 100]) * 0.05
    init_dgrad = np.ones([100, 100]) * -0.01
    init_spread = np.ones([100, 100]) * -0.01  # How to use the param
    init_dspread = np.ones([100, 100]) * -0.01  # How to use the param
    total_sum = np.ones([100, 100]) * 2000

    EvolutionBaseObj = EvolutionBase(id="01",
                                     name="EvolutionTest01",
                                     class_name="Hazardbase",
                                     init_value=init_value,
                                     init_grad=init_grad,
                                     init_dgrad=init_dgrad,
                                     init_spread=init_spread,
                                     init_dspread=init_dspread,
                                     min_value=0,
                                     max_value=100,
                                     total_sum=total_sum,
                                     area=[100, 100, 100],
                                     stride=2
                                     )

    EvolutionBaseObj_5 = EvolutionBase(id="02",
                                       name="EvolutionTest01",
                                       class_name="Hazardbase",
                                       init_value=init_value,
                                       init_grad=init_grad,
                                       init_dgrad=init_dgrad,
                                       init_spread=init_spread,
                                       init_dspread=init_dspread,
                                       min_value=0,
                                       max_value=100,
                                       total_sum=total_sum,
                                       area=[100, 100, 100],
                                       stride=3
                                       )
    EvolutionBaseObj_10 = EvolutionBase(id="02",
                                        name="EvolutionTest01",
                                        class_name="Hazardbase",
                                        init_value=init_value,
                                        init_grad=init_grad,
                                        init_dgrad=init_dgrad,
                                        init_spread=init_spread,
                                        init_dspread=init_dspread,
                                        min_value=0,
                                        max_value=100,
                                        total_sum=total_sum,
                                        area=[100, 100, 100],
                                        stride=5
                                        )
    # Define a custom evolution function
    EvolutionBaseObj.time_evolution_function.params = [np.zeros([100, 100]), np.zeros([100, 100])]  # init
    EvolutionBaseObj.set_mode(mode="mesh")
    EvolutionBaseObj_5.time_evolution_function.params = [np.zeros([100, 100]), np.zeros([100, 100])]  # init
    EvolutionBaseObj_5.set_mode(mode="mesh")
    EvolutionBaseObj_10.time_evolution_function.params = [np.zeros([100, 100]), np.zeros([100, 100])]  # init
    EvolutionBaseObj_10.set_mode(mode="mesh")

    def update_callback(Obj: EvolutionBase):
        """A test for update callback """
        # Obj.time_evolution_function.params = [Obj.get_value()] # PASS
        Obj.time_evolution_function.params = [(Obj.total_sum - Obj.current_sum) / 10, Obj.grad]
        Obj.current_sum = Obj.current_sum + Obj.get_value()
        pass

    EvolutionBaseObj.update_callback = update_callback
    EvolutionBaseObj_5.update_callback = update_callback
    EvolutionBaseObj_10.update_callback = update_callback

    def Ev_func1(args):
        return args[0] / 100

    def Ev_func2(args):
        return args[0] / 50

    EvolutionBaseObj.time_evolution_function.add_functions(Ev_func1)
    EvolutionBaseObj_5.time_evolution_function.add_functions(Ev_func1)
    EvolutionBaseObj_10.time_evolution_function.add_functions(Ev_func1)

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig2 = plt.figure(num=2, figsize=(128, 108))
    x, y = [], []

    def Evolution_plot(retval: np.ndarray):
        plt.subplot(1, 1, 1)
        meshval = retval.reshape([100, 100])
        im = plt.imshow(meshval, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=110)
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar()
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('热功率 单位(MW)')
        plt.title('热功率空间分布图')
        return im

    def Evolution_plot_v2(retval: np.ndarray, retval_5: np.ndarray, retval_10: np.ndarray, step):
        plt.subplot(1, 3, 1)
        plt.text(0, -20, "step={}".format(step))
        meshval = retval.reshape([100, 100])
        im = plt.imshow(meshval, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=110)
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('热功率 单位(MW)')
        plt.title('热功率空间分布图:stride=2')
        plt.subplot(1, 3, 2)
        meshval_5 = retval_5.reshape([100, 100])
        im = plt.imshow(meshval_5, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=110)
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('热功率 单位(MW)')
        plt.title('热功率空间分布图:stride=3')
        plt.subplot(1, 3, 3)
        meshval_10 = retval_10.reshape([100, 100])
        im = plt.imshow(meshval_10, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=110)
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('热功率 单位(MW)')
        plt.title('热功率空间分布图:stride=5')
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        return im

    t = np.array(list(range(0, 100)))

    def init():
        pass

    def update_point(step):
        retval = EvolutionBaseObj.update()
        retval_5 = EvolutionBaseObj_5.update()
        retval_10 = EvolutionBaseObj_10.update()
        # retval = space_evolution(EvolutionBaseObj.get_value())
        # EvolutionBaseObj.set_value(value=retval)
        # fig2.savefig(r"D:\Project\EmergencyDeductionEngine\docs\figs\imgs\img_{:0>2d}.png".format(step))
        return Evolution_plot_v2(retval, retval_5, retval_10, step)

    ani = FuncAnimation(fig2, update_point, frames=t,
                        init_func=init, interval=300, repeat=False)

    # ani.save(r"D:\Project\EmergencyDeductionEngine\docs\figs\space_evolution_with_different_stride.gif")
    plt.show()


def EvolutionsTestCase_08():
    print("----- Time and space evolution functions -----")
    # =============== init data ===============
    init_value = np.zeros([100, 100])
    init_value[49:51, 49:51] = 50
    # print(init_value)
    init_grad = np.ones([100, 100]) * 0.05
    init_dgrad = np.ones([100, 100]) * -0.01
    init_spread = np.ones([100, 100]) * -0.01  # How to use the param
    init_dspread = np.ones([100, 100]) * -0.01  # How to use the param
    total_sum = np.ones([100, 100]) * 2000

    EvolutionBaseObj = EvolutionBase(id="01",
                                     name="EvolutionTest01",
                                     class_name="Hazardbase",
                                     init_value=init_value,
                                     init_grad=init_grad,
                                     init_dgrad=init_dgrad,
                                     init_spread=init_spread,
                                     init_dspread=init_dspread,
                                     min_value=0,
                                     max_value=100,
                                     total_sum=total_sum,
                                     area=[100, 100, 100],
                                     stride=2
                                     )

    # Define a custom evolution function
    EvolutionBaseObj.time_evolution_function.params = [np.zeros([100, 100]), np.zeros([100, 100])]  # init
    EvolutionBaseObj.set_mode(mode="mesh")
    EvolutionBaseObj.localmesh.mask = np.zeros([100, 100])

    def update_callback(Obj: EvolutionBase):
        """A test for update callback """
        # Obj.time_evolution_function.params = [Obj.get_value()] # PASS
        Obj.time_evolution_function.params = [(Obj.total_sum - Obj.current_sum) / 10, Obj.grad]
        Obj.current_sum = Obj.current_sum + Obj.get_value()
        Obj.localmesh.mask = (Obj.get_value() > 0) * 1
        pass

    EvolutionBaseObj.update_callback = update_callback

    def Ev_func1(args):
        return args[0] / 100

    def Ev_func2(args):
        return args[0] / 50

    EvolutionBaseObj.time_evolution_function.add_functions(Ev_func1)

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig2 = plt.figure(num=2, figsize=(128, 108))
    x, y = [], []

    def Evolution_plot(retval: np.ndarray):
        plt.subplot(1, 2, 1)
        meshval = retval.reshape([100, 100])
        im = plt.imshow(meshval, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=110)
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar()
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('热功率 单位(MW)')
        plt.title('热功率空间分布图')

        ax1 = plt.subplot(1, 2, 2)
        im = plt.plot(x, y1, "r-")
        im = plt.plot(x, y2, "g-")
        im = plt.plot(x, y3, "b-")
        ax1.set_xlabel('时间(分钟)')
        ax1.set_ylabel('燃烧功率(兆瓦)')
        return im

    t = np.array(list(range(0, 60)))
    x, y1, y2, y3 = [], [], [], []

    def init():
        pass

    def update_point(step):
        retval = EvolutionBaseObj.update()
        x.append(step)
        y1.append(retval[0][0])
        y2.append(retval[25][25])
        y3.append(retval[50][50])
        if step == 10:
            EvolutionBaseObj.time_evolution_function.add_functions(Ev_func2)
        # retval = space_evolution(EvolutionBaseObj.get_value())
        # EvolutionBaseObj.set_value(value=retval)
        # fig2.savefig(r"D:\Project\EmergencyDeductionEngine\docs\figs\imgs\img_{:0>2d}.png".format(step))
        return Evolution_plot(retval)

    ani = FuncAnimation(fig2, update_point, frames=t,
                        init_func=init, interval=300, repeat=False)

    # ani.save(r"D:\Project\EmergencyDeductionEngine\docs\figs\space_evolution_with_different_stride.gif")
    plt.show()


def EvolutionsTestCase_09():
    print("----- Time and space evolution functions (with same init value) -----")
    # =============== init data ===============
    init_value = np.zeros([100, 100])
    init_value[49:51, 49:51] = 1
    # print(init_value)
    init_grad = np.ones([100, 100]) * 0
    init_dgrad = np.ones([100, 100]) * -0.01
    # init_spread = np.ones([100, 100]) * -0.01  # How to use the param
    # init_dspread = np.ones([100, 100]) * -0.01  # How to use the param
    init_spread = [2, 2, 1]
    init_dspread = [1, 1, 1]
    total_sum = np.ones([100, 100]) * 2000

    EvolutionBaseObj = EvolutionBase(id="01",
                                     name="EvolutionTest01",
                                     class_name="Hazardbase",
                                     init_value=init_value,
                                     init_grad=init_grad,
                                     init_dgrad=init_dgrad,
                                     init_spread=init_spread,
                                     init_dspread=init_dspread,
                                     min_value=0,
                                     max_value=100,
                                     total_sum=total_sum,
                                     area=[100, 100, 100],
                                     stride=[2, 2, 1],
                                     )

    # Define a custom evolution function
    EvolutionBaseObj.time_evolution_function.params = [np.zeros([100, 100]), np.zeros([100, 100])]  # init
    EvolutionBaseObj.set_mode(mode="mesh")
    EvolutionBaseObj.evolution_localmesh.mask = np.zeros([100, 100])
    EvolutionBaseObj.devolution_localmesh.mask = np.zeros([100, 100])

    def update_callback(Obj: EvolutionBase):
        """A test for update callback """
        # Obj.time_evolution_function.params = [Obj.get_value()] # PASS
        Obj.time_evolution_function.params = [(Obj.total_sum - Obj.current_sum) / 10, Obj.grad]
        Obj.current_sum = Obj.current_sum + Obj.get_value()
        # Obj.localmesh.mask = (Obj.get_value() > 0) * 1
        # Obj.localmesh.mask = (Obj.init_value > 0) * 1
        pass

    EvolutionBaseObj.update_callback = update_callback

    def Ev_func1(args):
        return args[0] / 50
        # return 1

    def Ev_func2(args):
        return args[0] / 50

    EvolutionBaseObj.time_evolution_function.add_functions(Ev_func1)

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig2 = plt.figure(num=2, figsize=(128, 108))
    x, y = [], []

    def Evolution_plot(retval: np.ndarray):
        plt.subplot(1, 2, 1)
        meshval = retval.reshape([100, 100])
        im = plt.imshow(meshval, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=110)
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar()
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('热功率 单位(MW)')
        plt.title('热功率空间分布图')

        # plt.subplot(2, 2, 2)
        # im = plt.imshow(delta_v, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=200)
        # plt.xlabel('经度方向坐标x')
        # plt.ylabel('纬度方向坐标y')
        # cb = plt.colorbar()
        # plt.xticks(np.arange(0, 100, 10))  # fixed
        # plt.yticks(np.arange(0, 100, 10))  # fixed
        # cb.set_label('残差热功率 单位(MW)')
        # plt.title('残差空间分布图')

        # plt.subplot(2, 2, 3)
        # im = plt.imshow(grad, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=1)
        # plt.xlabel('经度方向坐标x')
        # plt.ylabel('纬度方向坐标y')
        # cb = plt.colorbar()
        # plt.xticks(np.arange(0, 100, 10))  # fixed
        # plt.yticks(np.arange(0, 100, 10))  # fixed
        # cb.set_label('梯度')
        # plt.title('梯度空间分布图')

        ax1 = plt.subplot(1, 2, 2)
        im = plt.plot(x, y1, "r-")
        im = plt.plot(x, y2, "g-")
        im = plt.plot(x, y3, "b-")
        ax1.set_xlabel('时间(分钟)')
        ax1.set_ylabel('燃烧功率(兆瓦)')

        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        return im

    t = np.array(list(range(0, 90)))
    x, y1, y2, y3 = [], [], [], []

    def init():
        # EvolutionBaseObj.set_mask(mask=(EvolutionBaseObj.get_value() > 0)*1)
        EvolutionBaseObj.evolution_localmesh.mask = (EvolutionBaseObj.get_value() > 0) * 1
        EvolutionBaseObj.devolution_localmesh.mask = np.zeros_like(EvolutionBaseObj.evolution_localmesh.mask)
        pass

    def update_point(step):
        retval = EvolutionBaseObj.update()
        x.append(step)
        y1.append(retval[0][0])
        y2.append(retval[25][25])
        y3.append(retval[50][50])
        # if step == 10:
        #     EvolutionBaseObj.time_evolution_function.add_functions(Ev_func2)
        # retval = space_evolution(EvolutionBaseObj.get_value())
        # EvolutionBaseObj.set_value(value=retval)
        # fig2.savefig(r"D:\Project\EmergencyDeductionEngine\docs\figs\imgs\img_{:0>2d}.png".format(step))
        return Evolution_plot(retval)

    ani = FuncAnimation(fig2, update_point, frames=t,
                        init_func=init, interval=300, repeat=False)

    # ani.save(r"D:\Project\EmergencyDeductionEngine\docs\figs\space_evolution_with_different_stride.gif")
    plt.show()


def EvolutionsTestCase_10():
    print("----- Time and space evolution and devolution functions -----")
    # =============== init data ===============
    init_value = np.zeros([100, 100])
    # init_value[49:51, 49:51] = 1
    init_value[49:51, 49:51] = 1
    # print(init_value)
    init_grad = np.ones([100, 100]) * 0.1
    init_dgrad = np.ones([100, 100]) * -0.1
    # init_spread = np.ones([100, 100]) * -0.01  # How to use the param
    # init_dspread = np.ones([100, 100]) * -0.01  # How to use the param
    init_spread = [2, 2, 1]
    init_dspread = [3, 3, 1]
    total_sum = np.ones([100, 100]) * 2000

    EvolutionBaseObj = EvolutionBase(id="01",
                                     name="EvolutionTest01",
                                     class_name="Hazardbase",
                                     init_value=init_value,
                                     init_grad=init_grad,
                                     init_dgrad=init_dgrad,
                                     init_spread=init_spread,
                                     init_dspread=init_dspread,
                                     min_value=0,
                                     max_value=100,
                                     total_sum=total_sum,
                                     area=[100, 100, 100],
                                     stride=[2, 2, 1],
                                     )

    # Define a custom evolution function
    EvolutionBaseObj.time_evolution_function.params = [np.zeros([100, 100]), np.zeros([100, 100])]  # init
    EvolutionBaseObj.set_mode(mode="mesh")
    EvolutionBaseObj.evolution_localmesh.mask = np.zeros([100, 100])
    EvolutionBaseObj.devolution_localmesh.mask = np.zeros([100, 100])

    def update_callback(Obj: EvolutionBase):
        """A test for update callback """
        # Obj.time_evolution_function.params = [Obj.get_value()] # PASS
        Obj.time_evolution_function.params = [(Obj.total_sum - Obj.current_sum) / 10, Obj.grad]
        Obj.current_sum = Obj.current_sum + Obj.get_value()
        # Obj.localmesh.mask = (Obj.get_value() > 0) * 1
        # Obj.localmesh.mask = (Obj.init_value > 0) * 1
        pass

    EvolutionBaseObj.update_callback = update_callback

    def Ev_func1(args):
        return args[0] / 50
        # return 1

    def Ev_func2(args):
        return -1

    EvolutionBaseObj.time_evolution_function.add_functions(Ev_func1)


    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig2 = plt.figure(num=2, figsize=(128, 108))
    x, y = [], []

    def Evolution_plot(retval: np.ndarray, evolution_mask:np.ndarray, devolution_mask:np.ndarray, mask:np.ndarray):
        plt.subplot(2, 3, 1)
        meshval = retval.reshape([100, 100])
        im = plt.imshow(meshval, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=110)
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar()
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('热功率 单位(MW)')
        plt.title('热功率空间分布图')

        plt.subplot(2, 3, 2)
        im = plt.imshow(evolution_mask, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=1)
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar()
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('影响程度')
        plt.title('EvolutionMask')

        plt.subplot(2, 3, 3)
        im = plt.imshow(devolution_mask, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=1)
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar()
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('影响程度')
        plt.title('DevolutionMask')

        plt.subplot(2, 3, 4)
        im = plt.imshow(mask, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=1)
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar()
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('影响程度')
        plt.title('Mask')

        ax1 = plt.subplot(2, 3, 5)
        im = plt.plot(x, y1, "r-")
        im = plt.plot(x, y2, "g-")
        im = plt.plot(x, y3, "b-")
        ax1.set_xlabel('时间(分钟)')
        ax1.set_ylabel('燃烧功率(兆瓦)')

        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        return im

    t = np.array(list(range(0, 120)))
    x, y1, y2, y3 = [], [], [], []

    def init():
        # EvolutionBaseObj.set_mask(mask=(EvolutionBaseObj.get_value() > 0)*1)
        EvolutionBaseObj.evolution_localmesh.mask = (EvolutionBaseObj.get_value() > 0) * 1
        # EvolutionBaseObj.devolution_localmesh.mask = np.zeros_like(EvolutionBaseObj.evolution_localmesh.mask)
        # EvolutionBaseObj.devolution_localmesh.reset_origin(mode="2D", l_start=-65, w_start=-65)
        # EvolutionBaseObj.devolution_localmesh.get_meshgrid(mode="2D")
        EvolutionBaseObj.devolution_localmesh.mask = np.zeros_like(EvolutionBaseObj.evolution_localmesh.mask)
        EvolutionBaseObj.devolution_localmesh.reset_origin(mode="2D", l_start=-65, w_start=-65)
        EvolutionBaseObj.devolution_localmesh.get_mesh(mode="2D")
        EvolutionBaseObj.devolution_localmesh.mask[60:70, 60:70]=1
        pass

    def update_point(step):
        retval = EvolutionBaseObj.update()
        x.append(step)
        y1.append(retval[0][0])
        y2.append(retval[25][25])
        y3.append(retval[50][50])

        if step == 10:
            # tmp = EvolutionBaseObj.get_mask()
            # tmp[60: 70, 60: 70] = 1
            # EvolutionBaseObj.set_mask(tmp)
            EvolutionBaseObj.time_devolution_function.add_functions(Ev_func2)
            EvolutionBaseObj.disable_space_devolution()
            # EvolutionBaseObj.devolution_localmesh.mask = np.ones_like(EvolutionBaseObj.get_value())
        # retval = space_evolution(EvolutionBaseObj.get_value())
        # EvolutionBaseObj.set_value(value=retval)
        # fig2.savefig(r"D:\Project\EmergencyDeductionEngine\docs\figs\imgs\img_{:0>2d}.png".format(step))
        return Evolution_plot(retval,
                              EvolutionBaseObj.evolution_localmesh.mask,
                              EvolutionBaseObj.devolution_localmesh.mask,
                              EvolutionBaseObj.get_mask())

    ani = FuncAnimation(fig2, update_point, frames=t,
                        init_func=init, interval=300, repeat=False)

    # ani.save(r"D:\Project\EmergencyDeductionEngine\docs\figs\space_evolution_with_different_stride.gif")
    plt.show()


def EvolutionsTestCase_11():
    print("----- Time and space evolution and devolution functions with six points -----")
    # =============== init data ===============
    init_value = np.zeros([100, 100])
    # init_value[49:51, 49:51] = 1
    init_value[49:51, 49:51] = 1
    # print(init_value)
    init_grad = np.ones([100, 100]) * 0.1
    init_dgrad = np.ones([100, 100]) * -0.1
    # init_spread = np.ones([100, 100]) * -0.01  # How to use the param
    # init_dspread = np.ones([100, 100]) * -0.01  # How to use the param
    init_spread = [2, 2, 1]
    init_dspread = [1, 1, 1]
    total_sum = np.ones([100, 100]) * 4000

    EvolutionBaseObj = EvolutionBase(id="01",
                                     name="EvolutionTest01",
                                     class_name="Hazardbase",
                                     init_value=init_value,
                                     init_grad=init_grad,
                                     init_dgrad=init_dgrad,
                                     init_spread=init_spread,
                                     init_dspread=init_dspread,
                                     min_value=0,
                                     max_value=100,
                                     total_sum=total_sum,
                                     area=[100, 100, 100],
                                     stride=[2, 2, 1],
                                     )

    # Define a custom evolution function
    EvolutionBaseObj.time_evolution_function.params = [np.zeros([100, 100]), np.zeros([100, 100])]  # init
    EvolutionBaseObj.set_mode(mode="mesh")
    EvolutionBaseObj.evolution_localmesh.mask = np.zeros([100, 100])
    EvolutionBaseObj.devolution_localmesh.mask = np.zeros([100, 100])

    def update_callback(Obj: EvolutionBase):
        """A test for update callback """
        Obj.time_evolution_function.params = [(Obj.total_sum - Obj.current_sum) / 10, Obj.grad, Obj.passive_params]
        Obj.current_sum = Obj.current_sum + Obj.get_value()
        pass

    EvolutionBaseObj.update_callback = update_callback

    def Ev_func1(args):
        return args[0] / 200
        # return 1

    def Ev_func2(args):
        return -10

    def Ev_func3(args):
        # print("args[-1]/20:", args[-1]/20)
        return args[-1]/3

    EvolutionBaseObj.time_evolution_function.add_functions(Ev_func1)

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    import matplotlib
    matplotlib.rcParams['animation.embed_limit'] = 2 ** 128
    fig2 = plt.figure(num=2, figsize=(15, 8))
    x, y = [], []

    pt_view = [[99, 0], [25, 35], [50, 50], [60, 40], [25, 75], [75, 75]]

    def Evolution_plot(retval: np.ndarray, evolution_mask: np.ndarray, devolution_mask: np.ndarray, mask: np.ndarray):
        plt.subplot(2, 2, 1)
        meshval = retval.reshape([100, 100])
        im = plt.imshow(meshval, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=110)
        im = plt.plot(pt_view[0][0], pt_view[0][1], "o", color="r")
        im = plt.plot(pt_view[1][0], pt_view[1][1], "o", color="g")
        im = plt.plot(pt_view[2][0], pt_view[2][1], "o", color="b")
        im = plt.plot(pt_view[3][0], pt_view[3][1], "o", color="c")
        im = plt.plot(pt_view[4][0], pt_view[4][1], "o", color="m")
        im = plt.plot(pt_view[5][0], pt_view[5][1], "o", color="y")
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar()
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('热功率 单位(MW)')
        plt.title('热功率空间分布图')

        plt.subplot(2, 2, 3)
        im = plt.imshow(evolution_mask, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=1)
        im = plt.plot(pt_view[0][0], pt_view[0][1], "o", color="r")
        im = plt.plot(pt_view[1][0], pt_view[1][1], "o", color="g")
        im = plt.plot(pt_view[2][0], pt_view[2][1], "o", color="b")
        im = plt.plot(pt_view[3][0], pt_view[3][1], "o", color="c")
        im = plt.plot(pt_view[4][0], pt_view[4][1], "o", color="m")
        im = plt.plot(pt_view[5][0], pt_view[5][1], "o", color="y")
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar()
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('影响程度')
        plt.title('EvolutionMask')

        plt.subplot(2, 2, 4)
        im = plt.imshow(devolution_mask, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=1)
        im = plt.plot(pt_view[0][0], pt_view[0][1], "o", color="r")
        im = plt.plot(pt_view[1][0], pt_view[1][1], "o", color="g")
        im = plt.plot(pt_view[2][0], pt_view[2][1], "o", color="b")
        im = plt.plot(pt_view[3][0], pt_view[3][1], "o", color="c")
        im = plt.plot(pt_view[4][0], pt_view[4][1], "o", color="m")
        im = plt.plot(pt_view[5][0], pt_view[5][1], "o", color="y")
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar()
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('影响程度')
        plt.title('DevolutionMask')

        # plt.subplot(2, 3, 4)
        # im = plt.imshow(mask, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=1)
        # im = plt.plot(pt_view[0][0], pt_view[0][1], "o", color="r")
        # im = plt.plot(pt_view[1][0], pt_view[1][1], "o", color="g")
        # im = plt.plot(pt_view[2][0], pt_view[2][1], "o", color="b")
        # im = plt.plot(pt_view[3][0], pt_view[3][1], "o", color="c")
        # im = plt.plot(pt_view[4][0], pt_view[4][1], "o", color="m")
        # im = plt.plot(pt_view[5][0], pt_view[5][1], "o", color="y")
        # plt.xlabel('经度方向坐标x')
        # plt.ylabel('纬度方向坐标y')
        # cb = plt.colorbar()
        # plt.xticks(np.arange(0, 100, 10))  # fixed
        # plt.yticks(np.arange(0, 100, 10))  # fixed
        # cb.set_label('影响程度')
        # plt.title('Mask')

        ax1 = plt.subplot(2, 2, 2)
        im = plt.plot(x, y1, "r-")
        im = plt.plot(x, y2, "g-")
        im = plt.plot(x, y3, "b-")
        im = plt.plot(x, y4, "c-")
        im = plt.plot(x, y5, "m-")
        im = plt.plot(x, y6, "y-")
        ax1.set_xlabel('时间(分钟)')
        ax1.set_ylabel('燃烧功率(兆瓦)')

        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        return im

    t = np.array(list(range(0, 200)))
    x, y1, y2, y3, y4, y5, y6 = [], [], [], [], [], [], []

    def init():
        # EvolutionBaseObj.set_mask(mask=(EvolutionBaseObj.get_value() > 0)*1)
        EvolutionBaseObj.evolution_localmesh.mask = ((EvolutionBaseObj.get_value() > 0) * 1).astype("float64")
        # EvolutionBaseObj.devolution_localmesh.mask = np.zeros_like(EvolutionBaseObj.evolution_localmesh.mask)
        # EvolutionBaseObj.devolution_localmesh.reset_origin(mode="2D", l_start=-65, w_start=-65)
        # EvolutionBaseObj.devolution_localmesh.get_meshgrid(mode="2D")
        EvolutionBaseObj.devolution_localmesh.mask = np.zeros_like(EvolutionBaseObj.evolution_localmesh.mask).astype(
            "float64")
        EvolutionBaseObj.devolution_localmesh.reset_origin(mode="2D", l_start=-25, w_start=-25)
        EvolutionBaseObj.devolution_localmesh.get_mesh(mode="2D")

        pass

    def update_point(step):
        retval = EvolutionBaseObj.update()
        x.append(step)
        y1.append(retval[pt_view[0][1]][pt_view[0][0]])
        y2.append(retval[pt_view[1][1]][pt_view[1][0]])
        y3.append(retval[pt_view[2][1]][pt_view[2][0]])
        y4.append(retval[pt_view[3][1]][pt_view[3][0]])
        y5.append(retval[pt_view[4][1]][pt_view[4][0]])
        y6.append(retval[pt_view[5][1]][pt_view[5][0]])

        EvolutionBaseObj.passive_params = step
        if step == 10:
            EvolutionBaseObj.evolution_localmesh.mask[14:16, 14:16] = 1.0
            EvolutionBaseObj.time_evolution_function.add_functions(Ev_func3)
        if step == 20:
            # tmp = EvolutionBaseObj.get_mask()
            # tmp[60: 70, 60: 70] = 1
            # EvolutionBaseObj.set_mask(tmp)
            EvolutionBaseObj.devolution_localmesh.mask[60:70, 60:70] = 1.0
            EvolutionBaseObj.time_devolution_function.add_functions(Ev_func2)
            EvolutionBaseObj.enable_time_devolution()
        # if step == 25:
        #     EvolutionBaseObj.time_evolution_function.add_functions(Ev_func3)
        if step == 35:
            EvolutionBaseObj.time_evolution_function.delete_functions(Ev_func3)
        if step == 50:
            EvolutionBaseObj.disable_space_devolution()
        if step == 60:
            # EvolutionBaseObj.disable_space_devolution()
            EvolutionBaseObj.devolution_localmesh.mask = np.zeros([100, 100]).astype("float64")
        if step == 70:
            EvolutionBaseObj.enable_space_devolution()
            EvolutionBaseObj.devolution_localmesh.mask[60:70, 20:30] = 1.0
            # EvolutionBaseObj.devolution_localmesh.mask = np.ones_like(EvolutionBaseObj.get_value())
        # retval = space_evolution(EvolutionBaseObj.get_value())
        # EvolutionBaseObj.set_value(value=retval)
        # fig2.savefig(r"D:\Project\EmergencyDeductionEngine\docs\figs\imgs\img_{:0>3d}.png".format(step))
        return Evolution_plot(retval,
                              EvolutionBaseObj.evolution_localmesh.mask,
                              EvolutionBaseObj.devolution_localmesh.mask,
                              EvolutionBaseObj.get_mask())

    ani = FuncAnimation(fig2, update_point, frames=t,
                        init_func=init, interval=300, repeat=False)

    # ani.save(r"D:\Project\EmergencyDeductionEngine\docs\figs\6pts_in_hazardbase_sim.gif")
    # with open(r"D:\Project\EmergencyDeductionEngine\docs\figs\0421.html", "w") as f:
    #     print(ani.to_jshtml(), file=f)

    plt.show()



# </editor-fold>



def Multi_hazard_multi_peopleTest():
    print("Multi hazard and multi people evolution test")
    init_value_1 = np.zeros([100, 100])
    init_value_1[49:51, 49:51] = 1
    init_grad_1 = np.ones([100, 100]) * 2
    init_dgrad_1 = np.ones([100, 100]) * -0.1
    init_spread_1 = [2, 2, 1]
    init_dspread_1 = [1, 1, 1]
    total_sum_1 = np.ones([100, 100]) * 4000
    # -----------Master side_1-----------------#
    MasterObj_1 = EvolutionBase(
        id="01",
        name="MasterEvolutionObj_1",
        class_name="EvolutionBase",
        init_value=init_value_1,
        init_grad=init_grad_1,
        init_dgrad=init_dgrad_1,
        init_spread=init_spread_1,
        init_dspread=init_dspread_1,
        min_value=0,
        max_value=100,
        total_sum=total_sum_1,
        area=[100, 100, 100],
        stride=[2, 2, 1])
    MasterObj_1.time_evolution_function.params = [MasterObj_1.get_value(),  # value
                                                  MasterObj_1.grad,  # grad
                                                  MasterObj_1.total_sum,  # total sum
                                                  MasterObj_1.current_sum,  # current sum
                                                  [0]  # input params
                                                  ]
    MasterObj_1.time_devolution_function.params = [MasterObj_1.get_value(),  # value
                                                   MasterObj_1.grad,  # grad
                                                   MasterObj_1.total_sum,  # total sum
                                                   MasterObj_1.current_sum,  # current sum
                                                   [0]  # input params
                                                   ]
    MasterObj_1.space_evolution_function.params = [MasterObj_1.get_value(),  # value
                                                   MasterObj_1.grad,  # grad
                                                   MasterObj_1.total_sum,  # total sum
                                                   MasterObj_1.current_sum,  # current sum
                                                   [0]  # input params
                                                   ]
    MasterObj_1.space_devolution_function.params = [MasterObj_1.get_value(),  # value
                                                    MasterObj_1.grad,  # grad
                                                    MasterObj_1.total_sum,  # total sum
                                                    MasterObj_1.current_sum,  # current sum
                                                    [0]  # input params
                                                    ]
    MasterObj_1.set_mode(mode="mesh")
    MasterObj_1.evolution_localmesh.mask = (init_value_1 > 0) * 1.0
    MasterObj_1.devolution_localmesh.mask = np.zeros([100, 100])
    MasterObj_1.enable_time_evolution()
    MasterObj_1.enable_space_evolution()
    MasterObj_1.set_default_update_func(MasterObj_1.update)

    # -----------Master side_2-----------------#
    init_value_2 = np.ones([100, 100]) * 400
    init_value_2[49:51, 49:51] = 800
    init_grad_2 = np.ones([100, 100]) * 200
    init_dgrad_2 = np.ones([100, 100]) * -0.1
    init_spread_2 = [2, 2, 1]
    init_dspread_2 = [1, 1, 1]
    total_sum_2 = total_sum_1 * 2.3
    MasterObj_2 = EvolutionBase(
        id="02",
        name="MasterEvolutionObj_2",
        class_name="EvolutionBase",
        init_value=init_value_2,
        init_grad=init_grad_2,
        init_dgrad=init_dgrad_2,
        init_spread=init_spread_2,
        init_dspread=init_dspread_2,
        min_value=0,
        max_value=10000,
        total_sum=total_sum_2,
        area=[100, 100, 100],
        stride=[2, 2, 1])
    MasterObj_2.time_evolution_function.params = [MasterObj_2.get_value(),  # value
                                                  MasterObj_2.grad,  # grad
                                                  MasterObj_2.total_sum,  # total sum
                                                  MasterObj_2.current_sum,  # current sum
                                                  [0]  # input params
                                                  ]
    MasterObj_2.time_devolution_function.params = [MasterObj_2.get_value(),  # value
                                                   MasterObj_2.grad,  # grad
                                                   MasterObj_2.total_sum,  # total sum
                                                   MasterObj_2.current_sum,  # current sum
                                                   [0]  # input params
                                                   ]
    MasterObj_2.space_evolution_function.params = [MasterObj_2.get_value(),  # value
                                                   MasterObj_2.grad,  # grad
                                                   MasterObj_2.total_sum,  # total sum
                                                   MasterObj_2.current_sum,  # current sum
                                                   [0]  # input params
                                                   ]
    MasterObj_2.space_devolution_function.params = [MasterObj_2.get_value(),  # value
                                                    MasterObj_2.grad,  # grad
                                                    MasterObj_2.total_sum,  # total sum
                                                    MasterObj_2.current_sum,  # current sum
                                                    [0]  # input params
                                                    ]
    MasterObj_2.set_mode(mode="mesh")
    MasterObj_2.evolution_localmesh.mask = (init_value_2 > 400) * 1.0
    MasterObj_2.devolution_localmesh.mask = np.zeros([100, 100])
    MasterObj_2.enable_time_evolution()
    MasterObj_2.enable_space_evolution()
    MasterObj_2.set_default_update_func(MasterObj_2.update)

    # ===== slaveObj_1 =====
    slaveObj_1 = EvolutionBase(
        id='01',
        name='SlaveEvolutionObj_1',
        class_name='EvolutionBase',
        init_value=100,
        init_grad=-1,
        init_dgrad=1,
        min_value=0,
        max_value=100,
        total_sum=100,
    )
    # Define a custom evolution function
    slaveObj_1.time_evolution_function.params = [slaveObj_1.get_value(), slaveObj_1.grad, slaveObj_1.total_sum, slaveObj_1.current_sum, []]  # value/grad/total sum/current sum/input params
    slaveObj_1.time_devolution_function.params = [slaveObj_1.get_value(), slaveObj_1.grad, slaveObj_1.total_sum, slaveObj_1.current_sum, []]  # value/grad/total sum/current sum/input params
    slaveObj_1.space_evolution_function.params = [slaveObj_1.get_value(), slaveObj_1.grad, slaveObj_1.total_sum, slaveObj_1.current_sum, []]  # value/grad/total sum/current sum/input params
    slaveObj_1.space_devolution_function.params = [slaveObj_1.get_value(), slaveObj_1.grad, slaveObj_1.total_sum, slaveObj_1.current_sum, []]  # value/grad/total sum/current sum/input params

    slaveObj_1.set_mode(mode="point")
    slaveObj_1.set_pt_pos(pt_pos=[50, 50])
    slaveObj_1.enable_time_evolution()
    slaveObj_1.enable_space_evolution()
    slaveObj_1.set_default_update_func(func=slaveObj_1.update_in_temperal)

    slaveObj_2 = EvolutionBase(
        id='02',
        name='SlaveEvolutionObj_2',
        class_name='EvolutionBase',
        init_value=100,
        init_grad=-1,
        init_dgrad=1,
        min_value=0,
        max_value=100,
        total_sum=100,
    )
    # Define a custom evolution function
    slaveObj_2.time_evolution_function.params = [slaveObj_2.get_value(), slaveObj_2.grad, slaveObj_2.total_sum, slaveObj_2.current_sum, []]  # value/grad/total sum/current sum/input params
    slaveObj_2.time_devolution_function.params = [slaveObj_2.get_value(), slaveObj_2.grad, slaveObj_2.total_sum, slaveObj_2.current_sum, []]  # value/grad/total sum/current sum/input params
    slaveObj_2.space_evolution_function.params = [slaveObj_2.get_value(), slaveObj_2.grad, slaveObj_2.total_sum, slaveObj_2.current_sum, []]  # value/grad/total sum/current sum/input params
    slaveObj_2.space_devolution_function.params = [slaveObj_2.get_value(), slaveObj_2.grad, slaveObj_2.total_sum, slaveObj_2.current_sum, []]  # value/grad/total sum/current sum/input params

    slaveObj_2.set_mode(mode="point")
    slaveObj_2.set_pt_pos(pt_pos=[75, 75])
    slaveObj_2.enable_time_evolution()
    slaveObj_2.enable_space_evolution()
    slaveObj_2.set_default_update_func(func=slaveObj_2.update_in_temperal)

    def slave_side_callback_func_v2(Obj: EvolutionBase = None):
        Obj.time_evolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              Obj.input_params]
        print(Obj.input_params)
        Obj.current_sum = Obj.current_sum + Obj.get_value()

    def slave_side_evolution_func_v3(args):
        #print(args[-1][1])
        return -(0.5 * (0.8 * float(args[-1][0]) / 10 + 0.2 * float(args[-1][1]) / 1000))
    # def slave_side_evolution_func_v4(args):
    #     #print(args[-1][1])
    #     return -(0.5 * (0.8 * float(args[-1][2]) / 10 + 0.2 * float(args[-1][3]) / 1000))

    slaveObj_1.update_callback = slave_side_callback_func_v2
    slaveObj_2.update_callback = slave_side_callback_func_v2
    # slaveObj_1.time_evolution_function.add_functions(slave_side_evolution_func_v1)
    # slaveObj_1.time_evolution_function.add_functions(slave_side_evolution_func_v2)
    slaveObj_1.time_evolution_function.add_functions(slave_side_evolution_func_v3)
    slaveObj_2.time_evolution_function.add_functions(slave_side_evolution_func_v3)
    slaveObj_1.time_evolution_function.params = [slaveObj_1.get_value(), slaveObj_1.grad, slaveObj_1.total_sum, slaveObj_1.current_sum, [MasterObj_1.get_value(slaveObj_1.get_pt_pos())]]
    slaveObj_2.time_evolution_function.params = [slaveObj_2.get_value(), slaveObj_2.grad, slaveObj_2.total_sum,
                                                 slaveObj_2.current_sum,
                                                 [MasterObj_2.get_value(slaveObj_1.get_pt_pos())]]   #TODO:false add annother master
    # slaveObj_1.time_evolution_function.params = [0, 0, 0, 0, [0, 0, 0, 0]]
    # slaveObj_2.time_evolution_function.params = [0, 0, 0, 0, [0, 0, 0, 0]]

    HazardBaseObj = HazardBase()

    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=MasterObj_1,
        slave_hazard_object=slaveObj_1
    )

    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=MasterObj_2,
        slave_hazard_object=slaveObj_1
    )
    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=MasterObj_1,
        slave_hazard_object=slaveObj_2
    )

    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=MasterObj_2,
        slave_hazard_object=slaveObj_2
    )

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig2 = plt.figure(num=2, figsize=(15, 8))

    def Evolution_plot(retval_1: np.ndarray, retval_2: np.ndarray):
        # -----------------plot the Master_1 time and space evolution-------------------#
        plt.subplot(2, 3, 1)
        meshval_1 = retval_1.reshape([100, 100])
        im = plt.imshow(meshval_1, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=110)
        im = plt.plot(50, 50, color='red', marker="o")
        im = plt.plot(75, 75, color='green', marker="o")
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar()
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('热功率 单位(MW)')
        plt.title('热功率空间分布图')
        # -----------------plot the Master_2 time and space evolution-------------------#
        plt.subplot(2, 3, 2)
        meshval_2 = retval_2.reshape([100, 100])
        # im = plt.imshow(meshval_2, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=10000)
        # im = plt.imshow((meshval_2>420)*1, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=1)
        im = plt.imshow(meshval_2, interpolation=None, cmap=plt.cm.BuGn, vmin=400, vmax=10000)
        im = plt.plot(50, 50, color='red', marker="o")
        im = plt.plot(75, 75, color='green', marker="o")
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar()
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('CO2浓度 单位(PPM)')
        plt.title('CO2浓度空间分布图')

        # ------------------ plot the Master_1 time evolution -------------------#
        ax1 = plt.subplot(2, 3, 3)
        im = plt.plot(x, ya, "r-")
        im = plt.plot(x, yb, "g-")
        ax1.set_xlabel('时间(分钟)')
        ax1.set_ylabel('热功率（MW）')

        # ------------------ plot the Master_2 time evolution -------------------#
        ax1 = plt.subplot(2, 3, 4)
        im = plt.plot(x, yc, "r-")
        im = plt.plot(x, yd, "g-")
        ax1.set_xlabel('时间(分钟)')
        ax1.set_ylabel('CO2浓度（PPM）')
        # ------------------ plot the Master_1 and Master_2 impact for slave_1  -------------------#
        ax1 = plt.subplot(2, 3, 5)
        im = plt.plot(x, y1, "r-")
        ax1.set_xlabel('时间(分钟)')
        ax1.set_ylabel('生命值')
        # ------------------ plot the Master_1 and Master_2 impact for slave_2  -------------------#
        ax1 = plt.subplot(2, 3, 6)
        im = plt.plot(x, y2, "r-")
        ax1.set_xlabel('时间(分钟)')
        ax1.set_ylabel('生命值')

        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        return im

    t = np.array(list(range(0, 120)))
    x, ya, yb, yc, yd, y1, y2, y3, y4 = [], [], [], [], [], [], [], [], []

    def init():
        pass

    def update_point(step):
        # MasterEvolutionObj/SlaveEvolutionObj_1/SlaveEvolutionObj_2
        HazardBaseObj.update()
        x.append(step)
        ya.append(HazardBaseObj.value_list[MasterObj_1.get_name()][slaveObj_1.pos[0]][slaveObj_1.pos[1]])
        yb.append(HazardBaseObj.value_list[MasterObj_1.get_name()][slaveObj_2.pos[0]][slaveObj_2.pos[1]])
        yc.append(HazardBaseObj.value_list[MasterObj_2.get_name()][slaveObj_1.pos[0]][slaveObj_1.pos[1]])
        yd.append(HazardBaseObj.value_list[MasterObj_2.get_name()][slaveObj_2.pos[0]][slaveObj_2.pos[1]])
        y1.append(HazardBaseObj.value_list[slaveObj_1.get_name()])
        y2.append(HazardBaseObj.value_list[slaveObj_2.get_name()])

        # fig2.savefig(r"D:\Project\EmergencyDeductionEngine\docs\figs\imgs\img_{:0>2d}.png".format(step))
        return Evolution_plot(HazardBaseObj.value_list[MasterObj_1.get_name()],
                              HazardBaseObj.value_list[MasterObj_2.get_name()])

    ani = FuncAnimation(fig2, update_point, frames=t,
                        init_func=init, interval=300, repeat=False)

    # ani.save(r"D:\Project\EmergencyDeductionEngine\docs\figs\space_evolution_with_different_stride.gif")
    plt.show()
    pass

def hazard_scene():
    print("===== hazard_scene_test =====")
    # HazardBaseTest_v1()
    # HazardBaseTest_v2()
    #print("----- HazardBase test -----")

    init_value = np.zeros([100, 100])
    # init_value[49:51, 49:51] = 0
    init_grad = np.ones([100, 100]) * 2
    init_dgrad = np.ones([100, 100]) * -0.1
    init_spread = [2, 2, 1]
    init_dspread = [1, 1, 1]
    total_sum = np.ones([100, 100]) * 4000
    # ===== Disaster =====
    DisasterObj = EvolutionBase(
        id="01",
        name="DisasterEvolutionObj",
        class_name="EvolutionBase",
        init_value=init_value,
        init_grad=init_grad,
        init_dgrad=init_dgrad,
        init_spread=init_spread,
        init_dspread=init_dspread,
        min_value=0,
        max_value=100,
        total_sum=total_sum,
        area=[100, 100, 100],
        stride=[2, 2, 1])
    DisasterObj.time_evolution_function.params = [DisasterObj.get_value(),  # value
                                                DisasterObj.grad,  # grad
                                                DisasterObj.total_sum,  # total sum
                                                DisasterObj.current_sum,  # current sum
                                                [0]  # input params
                                                ]
    DisasterObj.time_devolution_function.params = [DisasterObj.get_value(),  # value
                                                DisasterObj.grad,  # grad
                                                DisasterObj.total_sum,  # total sum
                                                DisasterObj.current_sum,  # current sum
                                                 [0]  # input params
                                                 ]
    DisasterObj.space_evolution_function.params = [DisasterObj.get_value(),  # value
                                                DisasterObj.grad,  # grad
                                                DisasterObj.total_sum,  # total sum
                                                DisasterObj.current_sum,  # current sum
                                                 [0]  # input params
                                                 ]
    DisasterObj.space_devolution_function.params = [DisasterObj.get_value(),  # value
                                                DisasterObj.grad,  # grad
                                                DisasterObj.total_sum,  # total sum
                                                DisasterObj.current_sum,  # current sum
                                                  [0]  # input params
                                                  ]
    DisasterObj.set_mode(mode="mesh")
    DisasterObj.evolution_localmesh.mask = (init_value > 0) * 1.0
    DisasterObj.devolution_localmesh.mask = np.zeros([100, 100])
    DisasterObj.enable_time_evolution()
    DisasterObj.enable_space_evolution()
    DisasterObj.set_default_update_func(DisasterObj.update)


    # ===== Anti_disaster =====
    Anti_disasterObj = EvolutionBase(
        id="04",
        name="AntiDisasterObj",
        class_name="EvolutionBase",
        init_value=0,
        init_grad=0,
        init_dgrad=-1,
        init_spread=[1, 1, 1],
        init_dspread=[1, 1, 1],
        min_value=0,
        max_value=5,
        total_sum=200,
        area=[100, 100, 100],
        stride=[2, 2, 1],
        pos=[90, 90]
    )
    Anti_disasterObj.time_evolution_function.params = [Anti_disasterObj.get_value(), Anti_disasterObj.grad, Anti_disasterObj.total_sum, Anti_disasterObj.current_sum, []]
    Anti_disasterObj.time_devolution_function.params = [Anti_disasterObj.get_value(), Anti_disasterObj.grad, Anti_disasterObj.total_sum, Anti_disasterObj.current_sum, []]
    Anti_disasterObj.space_evolution_function.params = [Anti_disasterObj.get_value(), Anti_disasterObj.grad, Anti_disasterObj.total_sum, Anti_disasterObj.current_sum, []]
    Anti_disasterObj.space_devolution_function.params = [Anti_disasterObj.get_value(), Anti_disasterObj.grad, Anti_disasterObj.total_sum, Anti_disasterObj.current_sum, []]
    Anti_disasterObj.set_mode(mode="point")
    # Anti_disasterObj.set_pos(pos=[90, 90])
    Anti_disasterObj.set_pt_pos(pt_pos=None)
    # Anti_disasterObj.set_value(value=2, mode="no_default")
    Anti_disasterObj.set_default_update_func(Anti_disasterObj.update_in_temperal)

    # ===== slaveObj_1 =====
    personObj_1 = EvolutionBase(
        id='02',
        name='SlaveEvolutionObj_1',
        class_name='EvolutionBase',
        init_value=100,
        init_grad=-1,
        init_dgrad=1,
        min_value=0,
        max_value=100,
        total_sum=100,
    )
    # Define a custom evolution function
    personObj_1.time_evolution_function.params = [personObj_1.get_value(), personObj_1.grad, personObj_1.total_sum, personObj_1.current_sum, []]  # value/grad/total sum/current sum/input params
    personObj_1.time_devolution_function.params = [personObj_1.get_value(), personObj_1.grad, personObj_1.total_sum, personObj_1.current_sum, []]  # value/grad/total sum/current sum/input params
    personObj_1.space_evolution_function.params = [personObj_1.get_value(), personObj_1.grad, personObj_1.total_sum, personObj_1.current_sum, []]  # value/grad/total sum/current sum/input params
    personObj_1.space_devolution_function.params = [personObj_1.get_value(), personObj_1.grad, personObj_1.total_sum, personObj_1.current_sum, []]  # value/grad/total sum/current sum/input params

    personObj_1.set_mode(mode="point")
    # personObj_1.set_pos(pos=[50, 50])
    personObj_1.set_pt_pos(pt_pos=[50, 50])
    personObj_1.enable_time_evolution()
    personObj_1.enable_space_evolution()
    personObj_1.set_default_update_func(func=personObj_1.update_in_temperal)

    # ===== slaveObj_2 =====
    personObj_2 = EvolutionBase(
        id='03',
        name='SlaveEvolutionObj_2',
        class_name='EvolutionBase',
        init_value=100,
        init_grad=-1,
        init_dgrad=1,
        min_value=0,
        max_value=100,
        total_sum=100,
    )
    # Define a custom evolution function
    personObj_2.time_evolution_function.params = [personObj_2.get_value(), personObj_2.grad, personObj_2.total_sum, personObj_2.current_sum, []]  # value/grad/total sum/current sum/input params
    personObj_2.time_devolution_function.params = [personObj_2.get_value(), personObj_2.grad, personObj_2.total_sum, personObj_2.current_sum, []]  # value/grad/total sum/current sum/input params
    personObj_2.space_evolution_function.params = [personObj_2.get_value(), personObj_2.grad, personObj_2.total_sum, personObj_2.current_sum, []]  # value/grad/total sum/current sum/input params
    personObj_2.space_devolution_function.params = [personObj_2.get_value(), personObj_2.grad, personObj_2.total_sum, personObj_2.current_sum, []]  # value/grad/total sum/current sum/input params

    personObj_2.set_mode(mode="point")
    # personObj_2.set_pos(pos=[75, 75])
    personObj_2.set_pt_pos(pt_pos=[75, 75])
    personObj_2.enable_time_evolution()
    personObj_2.enable_space_evolution()
    personObj_2.set_default_update_func(func=personObj_2.update_in_temperal)

    # ===== Medical unit =====
    medicalObj = EvolutionBase(
        id='05',
        name='medicalObj',
        class_name='EvolutionBase',
        init_value=0,
        init_grad=0,
        init_dgrad=0,
        min_value=0,
        max_value=100,
        total_sum=100,
    )

    medicalObj.time_evolution_function.params = [medicalObj.get_value(), medicalObj.grad, medicalObj.total_sum, medicalObj.current_sum, []]  # value/grad/total sum/current sum/input params
    medicalObj.time_devolution_function.params = [medicalObj.get_value(), medicalObj.grad, medicalObj.total_sum, medicalObj.current_sum, []]  # value/grad/total sum/current sum/input params
    medicalObj.space_evolution_function.params = [medicalObj.get_value(), medicalObj.grad, medicalObj.total_sum, medicalObj.current_sum, []]  # value/grad/total sum/current sum/input params
    medicalObj.space_devolution_function.params = [medicalObj.get_value(), medicalObj.grad, medicalObj.total_sum, medicalObj.current_sum, []]  # value/grad/total sum/current sum/input params
    medicalObj.set_mode(mode="point")
    medicalObj.set_pt_pos(pt_pos=[75, 75])
    medicalObj.enable_time_evolution()
    medicalObj.enable_space_evolution()
    medicalObj.set_default_update_func(func=medicalObj.update_in_temperal)

    def disaster_callback_func_v1(Obj: EvolutionBase = None):
        Obj.time_evolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              Obj.input_params]
        Obj.time_devolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              Obj.input_params]
        Obj.current_sum = Obj.current_sum + Obj.get_value()

    def anti_disaster_callback_func_v1(Obj: EvolutionBase = None):
        Obj.time_evolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              Obj.input_params]
        Obj.current_sum = Obj.current_sum + Obj.get_value()


    def person_callback_func_v1(Obj: EvolutionBase = None):
        Obj.time_evolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              Obj.input_params]
        Obj.time_devolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              Obj.input_params]
        Obj.current_sum = Obj.current_sum + Obj.get_value()

    def medical_callback_func_v1(Obj: EvolutionBase = None):
        Obj.time_evolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              Obj.input_params]
        Obj.current_sum = Obj.current_sum + Obj.get_value()

    def disaster_evolution_func_v1(args):
        return (args[2]-args[3])/500

    def disaster_evolution_func_v2(args):
        # print("disaster_evolution_func_v2:", args)
        return 2

    def disaster_devolution_func_v1(args):
        # print("args:", args)
        # return args[-1][0]*-1
        return -10

    def anti_disaster_evoultion_func_v1(args):     #fireman status
        # 0-value, 1-grad, 2 total_sum, 3 current_sumtmp_value
        # print("anit args:", args)
        if (args[2]-args[3]) > 0.1*args[2]:
            return 0
        else:
            # return (args[2]-args[3])/2000
            return -1

    def medical_evolution_func_v1(args):            #medicalman status
        if (args[2] - args[3] > 0.1*args[2]):
            return 0
        else:
            return -1

    def personlife_evolution_func_v1(args):
        # print("personlife args:", args)
        tmp = args[-1][0]
        if tmp > 90:
            # print("tmp>90")
            return -5
        elif tmp > 70 and tmp <= 90:
            # print("tmp>70")
            return -3
        elif tmp > 50 and tmp <= 70:
            # print("tmp>50")
            return -2
        elif tmp > 20 and tmp <= 50:
            # print("tmp>20")
            return -1
        elif tmp > 0 and tmp <= 20:
            # print("tmp>0")
            return -0.5
        else:
            return 0

    def personlife_devolution_func_v1(args):
        print("personlife_devolution_func_v1:", args)
        return args[-1][1]*0.4

    # ====== CALLBACK FUNCTION SETTINGS ======
    DisasterObj.update_callback = disaster_callback_func_v1

    personObj_1.update_callback = person_callback_func_v1
    personObj_2.update_callback = person_callback_func_v1

    Anti_disasterObj.update_callback = anti_disaster_callback_func_v1
    medicalObj.update_callback = medical_callback_func_v1
    # slaveObj_1.set_default_update_func(func=slaveObj_1.update_in_temperal)

    # ====== TIME EVOLUTION FUNCTIONS SETTINGS ======
    DisasterObj.time_evolution_function.add_functions(disaster_evolution_func_v1)
    DisasterObj.time_evolution_function.add_functions(disaster_evolution_func_v2)
    DisasterObj.time_devolution_function.add_functions(disaster_devolution_func_v1)

    personObj_1.time_evolution_function.add_functions(personlife_evolution_func_v1)
    personObj_2.time_evolution_function.add_functions(personlife_evolution_func_v1)

    Anti_disasterObj.time_evolution_function.add_functions(anti_disaster_evoultion_func_v1)
    medicalObj.time_evolution_function.add_functions(medical_evolution_func_v1)
    personObj_2.time_devolution_function.add_functions(personlife_devolution_func_v1)

    # ====== TIME EVOLUTION PARAMETERS INIT ======
    DisasterObj.time_evolution_function.params = [DisasterObj.get_value(), DisasterObj.grad, DisasterObj.total_sum, DisasterObj.current_sum, [Anti_disasterObj.get_value()]]
    personObj_1.time_evolution_function.params = [personObj_1.get_value(), personObj_1.grad, personObj_1.total_sum, personObj_1.current_sum, [DisasterObj.get_value(personObj_1.get_pt_pos())]]
    personObj_2.time_evolution_function.params = [personObj_2.get_value(), personObj_2.grad, personObj_2.total_sum, personObj_2.current_sum, [DisasterObj.get_value(personObj_1.get_pt_pos())]]

    Anti_disasterObj.time_evolution_function.params = [Anti_disasterObj.get_value(), Anti_disasterObj.grad, Anti_disasterObj.total_sum, Anti_disasterObj.current_sum, [0]]
    # slaveObj_2.set_default_update_func(func=slaveObj_2.update_in_temperal)
    medicalObj.time_evolution_function.params = [medicalObj.get_value(), medicalObj.grad, medicalObj.total_sum, medicalObj.current_sum, [medicalObj.get_value()]]
    personObj_2.time_devolution_function.params = [personObj_2.get_value(), personObj_2.grad, personObj_2.total_sum, personObj_2.current_sum, [DisasterObj.get_value(personObj_1.get_pt_pos()), medicalObj.get_value()]]
    # ====== CREATING OBJECT OF HAZARDBASE  =======
    HazardBaseObj = HazardBase()

    # ------ MAPPING THE SPECIFIC HAZARD ------


    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=DisasterObj,
        slave_hazard_object=personObj_1
    )

    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=DisasterObj,
        slave_hazard_object=personObj_2
    )

    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=Anti_disasterObj,
        slave_hazard_object=DisasterObj
    )

    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=medicalObj,
        slave_hazard_object=personObj_2
    )

    # # HazardBaseObj.hazards_mappings_list[HazardMapping.get_default_mapping_name(master_name=MasterObj.get_name(), slave_name=slaveObj_1.get_name())].set_slave_callback_function()

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig2 = plt.figure(num=2, figsize=(15, 8))

    max_step = 200

    def Evolution_plot(retval: np.ndarray, ):
        plt.subplot(2, 4, 1)
        meshval = retval.reshape([100, 100])
        im = plt.imshow(meshval, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=110)
        im = plt.plot(50, 50, color='red', marker="o")
        im = plt.plot(75, 75, color='green', marker="o")
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar()
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('热功率 单位(MW)')
        plt.title('热功率空间分布图')

        ax1 = plt.subplot(2, 4, 2)
        im = plt.plot(x, ya, "r-")
        im = plt.plot(x, yb, "g-")
        ax1.set_xlabel('时间(分钟)')
        ax1.set_ylabel('热功率（MW）')
        plt.xlim(0, max_step)
        plt.ylim(DisasterObj.min_value, DisasterObj.max_value+10)
        plt.title('选择点的热功率曲线')

        ax1 = plt.subplot(2, 4, 3)
        im = plt.plot(x, y1, "r-")
        ax1.set_xlabel('时间(分钟)')
        ax1.set_ylabel('生命值')
        plt.xlim(0, max_step)
        plt.ylim(personObj_1.min_value, personObj_1.max_value+10)
        plt.title('A单元生命演化')

        ax1 = plt.subplot(2, 4, 4)
        im = plt.plot(x, y2, "g-")
        ax1.set_xlabel('时间(分钟)')
        ax1.set_ylabel('生命值')
        plt.xlim(0, max_step)
        plt.ylim(personObj_2.min_value, personObj_2.max_value+10)
        plt.title('B单元生命演化')

        ax1 = plt.subplot(2, 4, 5)
        im = plt.plot(x, y_anti_d, "b-")
        ax1.set_xlabel('时间(分钟)')
        ax1.set_ylabel('灭火剂量')
        plt.xlim(0, max_step)
        plt.ylim(Anti_disasterObj.min_value, Anti_disasterObj.max_value+1)
        plt.title('消防灭火设施状态')

        ax1 = plt.subplot(2, 4, 6)
        im = plt.plot(x, y_medical, color="darkcyan", linestyle="-")
        ax1.set_xlabel('时间(分钟)')
        ax1.set_ylabel('治愈能力')
        plt.xlim(0, max_step)
        plt.ylim(Anti_disasterObj.min_value, Anti_disasterObj.max_value+1)
        plt.title('医疗单元救治状态')


        plt.subplots_adjust(wspace=0.6, hspace=0.6)
        return im

    t = np.array(list(range(0, max_step)))
    x, ya, yb, y1, y2, y_anti_d, y_medical = [], [], [], [], [], [], []

    def init():
        DisasterObj.disable_time_evolution()
        DisasterObj.disable_space_evolution()
        # Anti_disasterObj.enable_time_evolution()
        # Anti_disasterObj.disable_time_devolution()
        # Anti_disasterObj.set_value(value=2, mode="no_default")
        pass

    def update_point(step):
        # DisasterEvolutionObj/SlaveEvolutionObj_1/SlaveEvolutionObj_2
        HazardBaseObj.update()

        x.append(step)
        ya.append(HazardBaseObj.value_list[DisasterObj.get_name()][personObj_1.get_pt_pos()[0]][personObj_1.get_pt_pos()[1]])
        yb.append(HazardBaseObj.value_list[DisasterObj.get_name()][personObj_2.get_pt_pos()[0]][personObj_2.get_pt_pos()[1]])
        y1.append(HazardBaseObj.value_list[personObj_1.get_name()])
        y2.append(HazardBaseObj.value_list[personObj_2.get_name()])
        y_anti_d.append(HazardBaseObj.value_list[Anti_disasterObj.get_name()])
        y_medical.append(HazardBaseObj.value_list[medicalObj.get_name()])


        if step == 10:
            init_value[49:51, 49:51] = 20
            DisasterObj.set_value(init_value)
            # MasterObj.evolution_localmesh.mask = (init_value > 0) * 1.0
            DisasterObj.enable_time_evolution()
            DisasterObj.enable_space_evolution()

        if step == 50:
            Anti_disasterObj.set_value(2)
            Anti_disasterObj.enable_time_evolution()
            DisasterObj.enable_time_devolution()
            DisasterObj.devolution_localmesh.mask[70:75, 70:75] = 1
            DisasterObj.enable_space_devolution()

        if step == 90:
            medicalObj.set_value(value=2, mode="no_mask")
            medicalObj.enable_time_evolution()
            personObj_2.enable_time_devolution()


        # fig2.savefig(r"D:\Project\EmergencyDeductionEngine\docs\figs\imgs\img_{:0>2d}.png".format(step))
        return Evolution_plot(HazardBaseObj.value_list[DisasterObj.get_name()])

    ani = FuncAnimation(fig2, update_point, frames=t,
                        init_func=init, interval=300, repeat=False)

    # ani.save(r"D:\Project\EmergencyDeductionEngine\docs\figs\multi_units_evolution_0504.gif")
    # with open (r"D:\Project\EmergencyDeductionEngine\docs\figs\multi_units_evolution_0504.html", "w") as f:
    #     print(ani.to_jshtml(), file = f)      #保存为html文件，可随时间回溯
    plt.show()
    pass

def hazard_scene_testv1():
    print("----- HazardBase test -----")

    init_value = np.zeros([100, 100])
    # init_value[49:51, 49:51] = 0
    init_grad = np.ones([100, 100]) * 2
    init_dgrad = np.ones([100, 100]) * -0.1
    init_spread = [2, 2, 1]
    init_dspread = [1, 1, 1]
    total_sum = np.ones([100, 100]) * 4000
    # ===== Disaster =====
    DisasterObj = EvolutionBase(
        id="01",
        name="DisasterEvolutionObj",
        class_name="EvolutionBase",
        init_value=init_value,
        init_grad=init_grad,
        init_dgrad=init_dgrad,
        init_spread=init_spread,
        init_dspread=init_dspread,
        min_value=0,
        max_value=100,
        total_sum=total_sum,
        area=[100, 100, 100],
        stride=[2, 2, 1])
    DisasterObj.time_evolution_function.params = [np.array([100, 100]),  # value
                                                  np.array([100, 100]),  # grad
                                                  np.array([100, 100]),  # total sum
                                                  np.array([100, 100]),  # current sum
                                                  []  # input params
                                                  ]
    DisasterObj.time_devolution_function.params = [np.array([100, 100]),  # value
                                                   np.array([100, 100]),  # dgrad
                                                   np.array([100, 100]),  # total sum
                                                   np.array([100, 100]),  # current sum
                                                   []  # input params
                                                   ]
    DisasterObj.space_evolution_function.params = [np.array([100, 100]),  # value
                                                   np.array([100, 100]),  # spread
                                                   np.array([100, 100]),  # total sum
                                                   np.array([100, 100]),  # current sum
                                                   []  # input params
                                                   ]
    DisasterObj.space_devolution_function.params = [np.array([100, 100]),  # value
                                                    np.array([100, 100]),  # dspread
                                                    np.array([100, 100]),  # total sum
                                                    np.array([100, 100]),  # current sum
                                                    []  # input params
                                                    ]
    DisasterObj.set_mode(mode="mesh")
    DisasterObj.evolution_localmesh.mask = (init_value > 0) * 1.0
    DisasterObj.devolution_localmesh.mask = np.zeros([100, 100])
    DisasterObj.enable_time_evolution()
    DisasterObj.enable_space_evolution()
    DisasterObj.set_default_update_func(DisasterObj.update)

    # ===== Anti_disaster =====
    Anti_disasterObj = EvolutionBase(
        id="04",
        name="AntiDisasterObj",
        class_name="EvolutionBase",
        init_value=100,
        init_grad=1,
        init_dgrad=-1,
        init_spread=[1, 1, 1],
        init_dspread=[1, 1, 1],
        min_value=0,
        max_value=100,
        total_sum=total_sum,
        area=[100, 100, 100],
        stride=[2, 2, 1],
        pos=[90, 90]
    )
    Anti_disasterObj.time_evolution_function.params = [0, 0, 0, 0, []]
    Anti_disasterObj.time_devolution_function.params = [0, 0, 0, 0, []]
    Anti_disasterObj.space_evolution_function.params = [0, 0, 0, 0, []]
    Anti_disasterObj.space_devolution_function.params = [0, 0, 0, 0, []]
    Anti_disasterObj.set_mode(mode="point")
    # Anti_disasterObj.set_pos(pos=[90, 90])
    Anti_disasterObj.set_pt_pos(pt_pos=None)
    Anti_disasterObj.disable_time_evolution()
    Anti_disasterObj.set_default_update_func(Anti_disasterObj.update_in_temperal)

    # ===== slaveObj_1 =====
    personObj_1 = EvolutionBase(
        id='02',
        name='SlaveEvolutionObj_1',
        class_name='EvolutionBase',
        init_value=100,
        init_grad=-1,
        init_dgrad=1,
        min_value=0,
        max_value=100,
        total_sum=100,
    )
    # Define a custom evolution function
    personObj_1.time_evolution_function.params = [0, 0, 0, 0, []]  # value/grad/total sum/current sum/input params
    personObj_1.time_devolution_function.params = [0, 0, 0, 0, []]  # value/grad/total sum/current sum/input params
    personObj_1.space_evolution_function.params = [0, 0, 0, 0, []]  # value/grad/total sum/current sum/input params
    personObj_1.space_devolution_function.params = [0, 0, 0, 0, []]  # value/grad/total sum/current sum/input params

    personObj_1.set_mode(mode="point")
    # personObj_1.set_pos(pos=[50, 50])
    personObj_1.set_pt_pos(pt_pos=[50, 50])
    personObj_1.enable_time_evolution()
    personObj_1.enable_space_evolution()
    personObj_1.set_default_update_func(func=personObj_1.update_in_temperal)

    # ===== slaveObj_2 =====
    personObj_2 = EvolutionBase(
        id='03',
        name='SlaveEvolutionObj_2',
        class_name='EvolutionBase',
        init_value=100,
        init_grad=-1,
        init_dgrad=1,
        min_value=0,
        max_value=100,
        total_sum=100,
    )
    # Define a custom evolution function
    personObj_2.time_evolution_function.params = [0, 0, 0, 0, []]  # value/grad/total sum/current sum/input params
    personObj_2.time_devolution_function.params = [0, 0, 0, 0, []]  # value/grad/total sum/current sum/input params
    personObj_2.space_evolution_function.params = [0, 0, 0, 0, []]  # value/grad/total sum/current sum/input params
    personObj_2.space_devolution_function.params = [0, 0, 0, 0, []]  # value/grad/total sum/current sum/input params

    personObj_2.set_mode(mode="point")
    # personObj_2.set_pos(pos=[75, 75])
    personObj_2.set_pt_pos(pt_pos=[75, 75])
    personObj_2.enable_time_evolution()
    personObj_2.enable_space_evolution()
    personObj_2.set_default_update_func(func=personObj_2.update_in_temperal)

    def disaster_callback_func_v1(Obj: EvolutionBase = None):
        Obj.time_evolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              [Obj.input_params]]
        Obj.current_sum = Obj.current_sum + Obj.get_value()

    def person_callback_func_v1(Obj: EvolutionBase = None):
        Obj.time_evolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              [Obj.input_params]]
        Obj.current_sum = Obj.current_sum + Obj.get_value()

    def disaster_evolution_func_v1(args):
        return (args[2] - args[3]) / 500

    def anti_disaster_evoultion_func_v1(args):
        return (args[2] - args[3]) / 1000

    def personlife_evolution_func_v1(args):
        # print("args:", args)
        tmp = args[-1][0]
        # print("tmp:", tmp)
        if tmp > 90:
            # print("tmp>90")
            return -5
        elif tmp > 70 and tmp <= 90:
            # print("tmp>70")
            return -3
        elif tmp > 50 and tmp <= 70:
            # print("tmp>50")
            return -2
        elif tmp > 20 and tmp <= 50:
            # print("tmp>20")
            return -1
        elif tmp > 0 and tmp <= 20:
            # print("tmp>0")
            return -0.5
        else:
            return 0

    # ====== CALLBACK FUNCTION SETTINGS ======
    DisasterObj.update_callback = disaster_callback_func_v1

    personObj_1.update_callback = person_callback_func_v1
    personObj_2.update_callback = person_callback_func_v1

    Anti_disasterObj.update_callback = disaster_callback_func_v1
    # slaveObj_1.set_default_update_func(func=slaveObj_1.update_in_temperal)

    # ====== TIME EVOLUTION FUNCTIONS SETTINGS ======
    DisasterObj.time_evolution_function.add_functions(disaster_evolution_func_v1)

    personObj_1.time_evolution_function.add_functions(personlife_evolution_func_v1)
    personObj_2.time_evolution_function.add_functions(personlife_evolution_func_v1)

    Anti_disasterObj.time_evolution_function.add_functions(anti_disaster_evoultion_func_v1)

    # ====== TIME EVOLUTION PARAMETERS INIT ======
    DisasterObj.time_evolution_function.params = [0, 0, 0, 0, [0]]
    personObj_1.time_evolution_function.params = [0, 0, 0, 0, [0]]
    personObj_2.time_evolution_function.params = [0, 0, 0, 0, [0]]
    # slaveObj_2.set_default_update_func(func=slaveObj_2.update_in_temperal)

    # ====== CREATING OBJECT OF HAZARDBASE  =======
    HazardBaseObj = HazardBase()

    # ------ MAPPING THE SPECIFIC HAZARD ------
    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=DisasterObj,
        slave_hazard_object=personObj_1
    )

    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=DisasterObj,
        slave_hazard_object=personObj_2
    )

    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=Anti_disasterObj,
        slave_hazard_object=DisasterObj
    )

    # HazardBaseObj.hazards_mappings_list[HazardMapping.get_default_mapping_name(master_name=MasterObj.get_name(), slave_name=slaveObj_1.get_name())].set_slave_callback_function()

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig2 = plt.figure(num=2, figsize=(15, 8))

    max_step = 140

    def Evolution_plot(retval: np.ndarray, ):
        plt.subplot(2, 2, 1)
        meshval = retval.reshape([100, 100])
        im = plt.imshow(meshval, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=110)
        im = plt.plot(50, 50, color='red', marker="o")
        im = plt.plot(75, 75, color='green', marker="o")
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar()
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('热功率 单位(MW)')
        plt.title('热功率空间分布图')

        ax1 = plt.subplot(2, 2, 2)
        im = plt.plot(x, ya, "r-")
        im = plt.plot(x, yb, "g-")
        ax1.set_xlabel('时间(分钟)')
        ax1.set_ylabel('热功率（MW）')
        plt.xlim(0, max_step)
        plt.ylim(DisasterObj.min_value, DisasterObj.max_value + 10)
        plt.title('选择点的热功率曲线')

        ax1 = plt.subplot(2, 2, 3)
        im = plt.plot(x, y1, "r-")
        ax1.set_xlabel('时间(分钟)')
        ax1.set_ylabel('生命值')
        plt.xlim(0, max_step)
        plt.ylim(personObj_1.min_value, personObj_1.max_value + 10)
        plt.title('A单元生命演化')

        ax1 = plt.subplot(2, 2, 4)
        im = plt.plot(x, y2, "g-")
        ax1.set_xlabel('时间(分钟)')
        ax1.set_ylabel('生命值')
        plt.xlim(0, max_step)
        plt.ylim(personObj_2.min_value, personObj_2.max_value + 10)
        plt.title('B单元生命演化')

        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        return im

    t = np.array(list(range(0, max_step)))
    x, ya, yb, y1, y2 = [], [], [], [], []

    def init():
        DisasterObj.disable_time_evolution()
        DisasterObj.disable_space_evolution()
        pass

    def update_point(step):
        # DisasterEvolutionObj/SlaveEvolutionObj_1/SlaveEvolutionObj_2
        HazardBaseObj.update()
        x.append(step)
        ya.append(
            HazardBaseObj.value_list[DisasterObj.get_name()][personObj_1.get_pt_pos()[0]][personObj_1.get_pt_pos()[1]])
        yb.append(
            HazardBaseObj.value_list[DisasterObj.get_name()][personObj_2.get_pt_pos()[0]][personObj_2.get_pt_pos()[1]])
        y1.append(HazardBaseObj.value_list[personObj_1.get_name()])
        y2.append(HazardBaseObj.value_list[personObj_2.get_name()])

        if step == 10:
            init_value[49:51, 49:51] = 20
            DisasterObj.set_value(init_value)
            # MasterObj.evolution_localmesh.mask = (init_value > 0) * 1.0
            DisasterObj.enable_time_evolution()
            DisasterObj.enable_space_evolution()

        # fig2.savefig(r"D:\Project\EmergencyDeductionEngine\docs\figs\imgs\img_{:0>2d}.png".format(step))
        return Evolution_plot(HazardBaseObj.value_list[DisasterObj.get_name()])

    ani = FuncAnimation(fig2, update_point, frames=t,
                        init_func=init, interval=300, repeat=False)

    # ani.save(r"D:\Project\EmergencyDeductionEngine\docs\figs\multi_units_evolution.gif")
    # with open(r"D:\Project\EmergencyDeductionEngine\docs\figs\multi_units_evolution.html", "w") as f:
    #     print(ani.to_jshtml(), file=f)  # 保存为html文件，可随时间回溯
    plt.show()
    pass

def hazard_scene_testv2():
    print("----- HazardBase test -----")

    init_value = np.zeros([100, 100])
    init_value[49:51, 49:51] = 20
    init_grad = np.ones([100, 100]) * 2
    init_dgrad = np.ones([100, 100]) * -0.1
    init_spread = [2, 2, 1]
    init_dspread = [1, 1, 1]
    total_sum = np.ones([100, 100]) * 4000
    MasterObj = EvolutionBase(
        id="01",
        name="DisasterEvolutionObj",
        class_name="EvolutionBase",
        init_value=init_value,
        init_grad=init_grad,
        init_dgrad=init_dgrad,
        init_spread=init_spread,
        init_dspread=init_dspread,
        min_value=0,
        max_value=100,
        total_sum=total_sum,
        area=[100, 100, 100],
        stride=[2, 2, 1])
    MasterObj.time_evolution_function.params = [np.array([100, 100]),  # value
                                                np.array([100, 100]),  # grad
                                                np.array([100, 100]),  # total sum
                                                np.array([100, 100]),  # current sum
                                                []  # input params
                                                ]
    MasterObj.time_devolution_function.params = [np.array([100, 100]),  # value
                                                 np.array([100, 100]),  # dgrad
                                                 np.array([100, 100]),  # total sum
                                                 np.array([100, 100]),  # current sum
                                                 []  # input params
                                                 ]
    MasterObj.space_evolution_function.params = [np.array([100, 100]),  # value
                                                 np.array([100, 100]),  # spread
                                                 np.array([100, 100]),  # total sum
                                                 np.array([100, 100]),  # current sum
                                                 []  # input params
                                                 ]
    MasterObj.space_devolution_function.params = [np.array([100, 100]),  # value
                                                  np.array([100, 100]),  # dspread
                                                  np.array([100, 100]),  # total sum
                                                  np.array([100, 100]),  # current sum
                                                  []  # input params
                                                  ]
    MasterObj.set_mode(mode="mesh")
    MasterObj.evolution_localmesh.mask = (init_value > 0) * 1.0
    MasterObj.devolution_localmesh.mask = np.zeros([100, 100])
    MasterObj.enable_time_evolution()
    MasterObj.enable_space_evolution()
    MasterObj.set_default_update_func(MasterObj.update)

    # ===== slaveObj_1 =====
    slaveObj_1 = EvolutionBase(
        id='02',
        name='SlaveEvolutionObj_1',
        class_name='EvolutionBase',
        init_value=100,
        init_grad=-1,
        init_dgrad=1,
        min_value=0,
        max_value=1000,
        total_sum=100,
    )
    # Define a custom evolution function
    slaveObj_1.time_evolution_function.params = [0, 0, 0, 0, []]  # value/grad/total sum/current sum/input params
    slaveObj_1.time_devolution_function.params = [0, 0, 0, 0, []]  # value/grad/total sum/current sum/input params
    slaveObj_1.space_evolution_function.params = [0, 0, 0, 0, []]  # value/grad/total sum/current sum/input params
    slaveObj_1.space_devolution_function.params = [0, 0, 0, 0, []]  # value/grad/total sum/current sum/input params

    slaveObj_1.set_mode(mode="point")
    slaveObj_1.set_pos(pos=[50, 50])
    slaveObj_1.enable_time_evolution()
    slaveObj_1.enable_space_evolution()
    slaveObj_1.set_default_update_func(func=slaveObj_1.update_in_temperal)

    # ===== slaveObj_2 =====
    slaveObj_2 = EvolutionBase(
        id='03',
        name='SlaveEvolutionObj_2',
        class_name='EvolutionBase',
        init_value=100,
        init_grad=-1,
        init_dgrad=1,
        min_value=0,
        max_value=1000,
        total_sum=100,
    )
    # Define a custom evolution function
    slaveObj_2.time_evolution_function.params = [0, 0, 0, 0, []]  # value/grad/total sum/current sum/input params
    slaveObj_2.time_devolution_function.params = [0, 0, 0, 0, []]  # value/grad/total sum/current sum/input params
    slaveObj_2.space_evolution_function.params = [0, 0, 0, 0, []]  # value/grad/total sum/current sum/input params
    slaveObj_2.space_devolution_function.params = [0, 0, 0, 0, []]  # value/grad/total sum/current sum/input params

    slaveObj_2.set_mode(mode="point")
    slaveObj_2.set_pos(pos=[75, 75])
    slaveObj_2.enable_time_evolution()
    slaveObj_2.enable_space_evolution()
    slaveObj_2.set_default_update_func(func=slaveObj_2.update_in_temperal)

    # =====================


    def person_callback_func_v1(Obj: EvolutionBase = None):
        Obj.time_evolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              [Obj.input_params]]
        Obj.current_sum = Obj.current_sum + Obj.get_value()

    def slave_side_evolution_func_v1(args):
        # print("args:", args)
        tmp = args[-1][0]
        print("tmp:", tmp)
        if tmp > 90:
            print("tmp>90")
            return -5
        elif tmp > 70 and tmp <= 90:
            print("tmp>70")
            return -3
        elif tmp > 50 and tmp <= 70:
            print("tmp>50")
            return -2
        elif tmp > 20 and tmp <= 50:
            print("tmp>20")
            return -1
        elif tmp > 0 and tmp <= 20:
            print("tmp>0")
            return -0.5
        else:
            return 0



    slaveObj_1.update_callback = person_callback_func_v1
    slaveObj_2.update_callback = person_callback_func_v1
    # slaveObj_1.set_default_update_func(func=slaveObj_1.update_in_temperal)

    slaveObj_1.time_evolution_function.add_functions(slave_side_evolution_func_v1)
    slaveObj_2.time_evolution_function.add_functions(slave_side_evolution_func_v1)

    slaveObj_1.time_evolution_function.params = [0, 0, 0, 0, [0]]
    slaveObj_2.time_evolution_function.params = [0, 0, 0, 0, [0]]


    HazardBaseObj = HazardBase()

    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=MasterObj,
        slave_hazard_object=slaveObj_1
    )

    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=MasterObj,
        slave_hazard_object=slaveObj_2
    )

    # HazardBaseObj.hazards_mappings_list[HazardMapping.get_default_mapping_name(master_name=MasterObj.get_name(), slave_name=slaveObj_1.get_name())].set_slave_callback_function()


    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig2 = plt.figure(num=2, figsize=(15, 8))

    def Evolution_plot(retval: np.ndarray, ):
        plt.subplot(2, 2, 1)
        meshval = retval.reshape([100, 100])
        im = plt.imshow(meshval, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=110)
        im = plt.plot(50, 50, color='red', marker="o")
        im = plt.plot(75, 75, color='green', marker="o")
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar()
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('热功率 单位(MW)')
        plt.title('热功率空间分布图')

        ax1 = plt.subplot(2, 2, 2)
        im = plt.plot(x, ya, "r-")
        im = plt.plot(x, yb, "g-")
        ax1.set_xlabel('时间(分钟)')
        ax1.set_ylabel('热功率（MW）')

        ax1 = plt.subplot(2, 2, 3)
        im = plt.plot(x, y1, "r-")
        ax1.set_xlabel('时间(分钟)')
        ax1.set_ylabel('生命值')

        ax1 = plt.subplot(2, 2, 4)
        im = plt.plot(x, y2, "g-")
        ax1.set_xlabel('时间(分钟)')
        ax1.set_ylabel('生命值')

        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        return im

    t = np.array(list(range(0, 120)))
    x, ya, yb, y1, y2 = [], [], [], [], []

    def init():
        pass

    def update_point(step):
        # DisasterEvolutionObj/SlaveEvolutionObj_1/SlaveEvolutionObj_2
        HazardBaseObj.update()
        x.append(step)
        ya.append(HazardBaseObj.value_list[MasterObj.get_name()][slaveObj_1.pos[0]][slaveObj_1.pos[1]])
        yb.append(HazardBaseObj.value_list[MasterObj.get_name()][slaveObj_2.pos[0]][slaveObj_2.pos[1]])
        y1.append(HazardBaseObj.value_list[slaveObj_1.get_name()])
        y2.append(HazardBaseObj.value_list[slaveObj_2.get_name()])

        # fig2.savefig(r"D:\Project\EmergencyDeductionEngine\docs\figs\imgs\img_{:0>2d}.png".format(step))
        return Evolution_plot(HazardBaseObj.value_list[MasterObj.get_name()])

    ani = FuncAnimation(fig2, update_point, frames=t,
                        init_func=init, interval=300, repeat=False)

    # ani.save(r"D:\Project\EmergencyDeductionEngine\docs\figs\space_evolution_with_different_stride.gif")
    plt.show()
    pass





def fire_scene():
    print("===== fire_scene_test =====")

    init_value = np.zeros([100, 100])
    # init_value[49:51, 49:51] = 0
    init_grad = np.ones([100, 100]) * 2
    init_dgrad = np.ones([100, 100]) * -0.1
    init_spread = [2, 2, 1]
    init_dspread = [1, 1, 1]
    total_sum = np.ones([100, 100]) * 4000
    # ===== Disaster: heat=====
    DisasterObj_1 = EvolutionBase(
        id="01",
        name="DisasterEvolutionObj_1",
        class_name="EvolutionBase",
        init_value=init_value,
        init_grad=init_grad,
        init_dgrad=init_dgrad,
        init_spread=init_spread,
        init_dspread=init_dspread,
        min_value=0,
        max_value=100,
        total_sum=total_sum,
        area=[100, 100, 100],
        stride=[2, 2, 1])
    DisasterObj_1.time_evolution_function.params = [DisasterObj_1.get_value(),  # value
                                                DisasterObj_1.grad,  # grad
                                                DisasterObj_1.total_sum,  # total sum
                                                DisasterObj_1.current_sum,  # current sum
                                                []  # input params
                                                ]
    DisasterObj_1.time_devolution_function.params = [DisasterObj_1.get_value(),  # value
                                                DisasterObj_1.grad,  # grad
                                                DisasterObj_1.total_sum,  # total sum
                                                DisasterObj_1.current_sum,  # current sum
                                                 []  # input params
                                                 ]
    DisasterObj_1.space_evolution_function.params = [DisasterObj_1.get_value(),  # value
                                                DisasterObj_1.grad,  # grad
                                                DisasterObj_1.total_sum,  # total sum
                                                DisasterObj_1.current_sum,  # current sum
                                                 []  # input params
                                                 ]
    DisasterObj_1.space_devolution_function.params = [DisasterObj_1.get_value(),  # value
                                                DisasterObj_1.grad,  # grad
                                                DisasterObj_1.total_sum,  # total sum
                                                DisasterObj_1.current_sum,  # current sum
                                                  []  # input params
                                                  ]
    DisasterObj_1.set_mode(mode="mesh")
    DisasterObj_1.evolution_localmesh.mask = (init_value > 0) * 1.0
    DisasterObj_1.devolution_localmesh.mask = np.zeros([100, 100])
    DisasterObj_1.enable_time_evolution()
    DisasterObj_1.enable_space_evolution()
    DisasterObj_1.set_default_update_func(DisasterObj_1.update)

    # ===== Disaster: co2=====
    init_value_2 = np.ones([100, 100]) * 400
    init_value_2[49:51, 49:51] = 800
    init_grad_2 = np.ones([100, 100]) * 200
    init_dgrad_2 = np.ones([100, 100]) * -0.1
    init_spread_2 = [2, 2, 1]
    init_dspread_2 = [1, 1, 1]
    total_sum_2 = total_sum * 2.3
    DisasterObj_2 = EvolutionBase(
        id="02",
        name="DisasterEvolutionObj_2",
        class_name="EvolutionBase",
        init_value=init_value_2,
        init_grad=init_grad_2,
        init_dgrad=init_dgrad_2,
        init_spread=init_spread_2,
        init_dspread=init_dspread_2,
        min_value=0,
        max_value=10000,
        total_sum=total_sum_2,
        area=[100, 100, 100],
        stride=[2, 2, 1])
    DisasterObj_2.time_evolution_function.params = [DisasterObj_2.get_value(),  # value
                                                    DisasterObj_2.grad,  # grad
                                                    DisasterObj_2.total_sum,  # total sum
                                                    DisasterObj_2.current_sum,  # current sum
                                                    []  # input params
                                                    ]
    DisasterObj_2.time_devolution_function.params = [DisasterObj_2.get_value(),  # value
                                                     DisasterObj_2.grad,  # grad
                                                     DisasterObj_2.total_sum,  # total sum
                                                     DisasterObj_2.current_sum,  # current sum
                                                     []  # input params
                                                     ]
    DisasterObj_2.space_evolution_function.params = [DisasterObj_2.get_value(),  # value
                                                     DisasterObj_2.grad,  # grad
                                                     DisasterObj_2.total_sum,  # total sum
                                                     DisasterObj_2.current_sum,  # current sum
                                                     []  # input params
                                                     ]
    DisasterObj_2.space_devolution_function.params = [DisasterObj_2.get_value(),  # value
                                                      DisasterObj_2.grad,  # grad
                                                      DisasterObj_2.total_sum,  # total sum
                                                      DisasterObj_2.current_sum,  # current sum
                                                      []  # input params
                                                      ]
    DisasterObj_2.set_mode(mode="mesh")
    DisasterObj_2.evolution_localmesh.mask = (init_value_2 > 400) * 1.0
    DisasterObj_2.devolution_localmesh.mask = np.zeros([100, 100])
    DisasterObj_2.enable_time_evolution()
    DisasterObj_2.enable_space_evolution()
    DisasterObj_2.set_default_update_func(DisasterObj_2.update)

    # ===== Anti_disaster =====
    Anti_disasterObj = EvolutionBase(
        id="04",
        name="AntiDisasterObj",
        class_name="EvolutionBase",
        init_value=0,
        init_grad=0,
        init_dgrad=-1,
        init_spread=[1, 1, 1],
        init_dspread=[1, 1, 1],
        min_value=0,
        max_value=5,
        total_sum=200,
        area=[100, 100, 100],
        stride=[2, 2, 1],
        pos=[90, 90]
    )
    Anti_disasterObj.time_evolution_function.params = [Anti_disasterObj.get_value(), Anti_disasterObj.grad, Anti_disasterObj.total_sum, Anti_disasterObj.current_sum, []]
    Anti_disasterObj.time_devolution_function.params = [Anti_disasterObj.get_value(), Anti_disasterObj.grad, Anti_disasterObj.total_sum, Anti_disasterObj.current_sum, []]
    Anti_disasterObj.space_evolution_function.params = [Anti_disasterObj.get_value(), Anti_disasterObj.grad, Anti_disasterObj.total_sum, Anti_disasterObj.current_sum, []]
    Anti_disasterObj.space_devolution_function.params = [Anti_disasterObj.get_value(), Anti_disasterObj.grad, Anti_disasterObj.total_sum, Anti_disasterObj.current_sum, []]
    Anti_disasterObj.set_mode(mode="point")
    # Anti_disasterObj.set_pos(pos=[90, 90])
    Anti_disasterObj.set_pt_pos(pt_pos=None)
    # Anti_disasterObj.set_value(value=2, mode="no_default")
    Anti_disasterObj.set_default_update_func(Anti_disasterObj.update_in_temperal)

    # ===== slaveObj_1 =====
    personObj_1 = EvolutionBase(
        id='02',
        name='SlaveEvolutionObj_1',
        class_name='EvolutionBase',
        init_value=100,
        init_grad=-1,
        init_dgrad=1,
        min_value=0,
        max_value=100,
        total_sum=100,
    )
    # Define a custom evolution function
    personObj_1.time_evolution_function.params = [personObj_1.get_value(), personObj_1.grad, personObj_1.total_sum, personObj_1.current_sum, []]  # value/grad/total sum/current sum/input params
    personObj_1.time_devolution_function.params = [personObj_1.get_value(), personObj_1.grad, personObj_1.total_sum, personObj_1.current_sum, []]  # value/grad/total sum/current sum/input params
    personObj_1.space_evolution_function.params = [personObj_1.get_value(), personObj_1.grad, personObj_1.total_sum, personObj_1.current_sum, []]  # value/grad/total sum/current sum/input params
    personObj_1.space_devolution_function.params = [personObj_1.get_value(), personObj_1.grad, personObj_1.total_sum, personObj_1.current_sum, []]  # value/grad/total sum/current sum/input params

    personObj_1.set_mode(mode="point")
    # personObj_1.set_pos(pos=[50, 50])
    personObj_1.set_pt_pos(pt_pos=[50, 50])
    personObj_1.enable_time_evolution()
    personObj_1.enable_space_evolution()
    personObj_1.set_default_update_func(func=personObj_1.update_in_temperal)

    # ===== slaveObj_2 =====
    personObj_2 = EvolutionBase(
        id='03',
        name='SlaveEvolutionObj_2',
        class_name='EvolutionBase',
        init_value=100,
        init_grad=-1,
        init_dgrad=1,
        min_value=0,
        max_value=100,
        total_sum=100,
    )
    # Define a custom evolution function
    personObj_2.time_evolution_function.params = [personObj_2.get_value(), personObj_2.grad, personObj_2.total_sum, personObj_2.current_sum, []]  # value/grad/total sum/current sum/input params
    personObj_2.time_devolution_function.params = [personObj_2.get_value(), personObj_2.grad, personObj_2.total_sum, personObj_2.current_sum, []]  # value/grad/total sum/current sum/input params
    personObj_2.space_evolution_function.params = [personObj_2.get_value(), personObj_2.grad, personObj_2.total_sum, personObj_2.current_sum, []]  # value/grad/total sum/current sum/input params
    personObj_2.space_devolution_function.params = [personObj_2.get_value(), personObj_2.grad, personObj_2.total_sum, personObj_2.current_sum, []]  # value/grad/total sum/current sum/input params

    personObj_2.set_mode(mode="point")
    # personObj_2.set_pos(pos=[75, 75])
    personObj_2.set_pt_pos(pt_pos=[75, 75])
    personObj_2.enable_time_evolution()
    personObj_2.enable_space_evolution()
    personObj_2.set_default_update_func(func=personObj_2.update_in_temperal)

    # ===== Medical unit =====
    medicalObj = EvolutionBase(
        id='05',
        name='medicalObj',
        class_name='EvolutionBase',
        init_value=0,
        init_grad=0,
        init_dgrad=0,
        min_value=0,
        max_value=100,
        total_sum=100,
    )

    medicalObj.time_evolution_function.params = [medicalObj.get_value(), medicalObj.grad, medicalObj.total_sum, medicalObj.current_sum, []]  # value/grad/total sum/current sum/input params
    medicalObj.time_devolution_function.params = [medicalObj.get_value(), medicalObj.grad, medicalObj.total_sum, medicalObj.current_sum, []]  # value/grad/total sum/current sum/input params
    medicalObj.space_evolution_function.params = [medicalObj.get_value(), medicalObj.grad, medicalObj.total_sum, medicalObj.current_sum, []]  # value/grad/total sum/current sum/input params
    medicalObj.space_devolution_function.params = [medicalObj.get_value(), medicalObj.grad, medicalObj.total_sum, medicalObj.current_sum, []]  # value/grad/total sum/current sum/input params
    medicalObj.set_mode(mode="point")
    medicalObj.set_pt_pos(pt_pos=[75, 75])
    medicalObj.enable_time_evolution()
    medicalObj.enable_space_evolution()
    medicalObj.set_default_update_func(func=medicalObj.update_in_temperal)

    def disaster_callback_func_v1(Obj: EvolutionBase = None):
        Obj.time_evolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              Obj.input_params]
        Obj.time_devolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              Obj.input_params]
        Obj.current_sum = Obj.current_sum + Obj.get_value()

    def anti_disaster_callback_func_v1(Obj: EvolutionBase = None):
        Obj.time_evolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              Obj.input_params]
        Obj.current_sum = Obj.current_sum + Obj.get_value()


    def person_callback_func_v1(Obj: EvolutionBase = None):
        Obj.time_evolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              Obj.input_params]
        Obj.time_devolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              Obj.input_params]
        Obj.current_sum = Obj.current_sum + Obj.get_value()

    def person_callback_func_v2(Obj: EvolutionBase = None):
        Obj.time_evolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              Obj.input_params]
        Obj.time_devolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              Obj.input_params]
        Obj.current_sum = Obj.current_sum + Obj.get_value()

    def medical_callback_func_v1(Obj: EvolutionBase = None):
        Obj.time_evolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              Obj.input_params]
        Obj.current_sum = Obj.current_sum + Obj.get_value()

    def disaster_evolution_func_v1(args):
        return (args[2]-args[3])/500

    def disaster_evolution_func_v2(args):
        # print("disaster_evolution_func_v2:", args)
        return 2

    def disaster_devolution_func_v1(args):
        # print("args:", args)
        # return args[-1][0]*-1
        return -10

    def anti_disaster_evoultion_func_v1(args):     #fireman status
        # 0-value, 1-grad, 2 total_sum, 3 current_sumtmp_value
        # print("anit args:", args)
        if (args[2]-args[3]) > 0.1*args[2]:
            return 0
        else:
            # return (args[2]-args[3])/2000
            return -1

    def medical_evolution_func_v1(args):            #medicalman status
        if (args[2] - args[3] > 0.1*args[2]):
            return 0
        else:
            return -1

    def personlife_evolution_func_v1(args):
        # print("personlife args:", args)
        tmp = args[-1][0]
        if tmp > 90:
            # print("tmp>90")
            return -5
        elif tmp > 70 and tmp <= 90:
            # print("tmp>70")
            return -3
        elif tmp > 50 and tmp <= 70:
            # print("tmp>50")
            return -2
        elif tmp > 20 and tmp <= 50:
            # print("tmp>20")
            return -1
        elif tmp > 0 and tmp <= 20:
            # print("tmp>0")
            return -0.5
        else:
            return 0

    def personlife_evolution_func_v2(args):
        if args[-1][0]<25 or args[-1][1]<2500:
            return 0
        else:
            return -(0.5 * (0.8 * float(args[-1][0]) / 10 + 0.2 * float(args[-1][1]) / 1000))


    def personlife_devolution_func_v1(args):
        #print("personlife_devolution_func_v1:", args)
        return args[-1][2]*0.4
        #return 0.8


    # ====== CALLBACK FUNCTION SETTINGS ======
    DisasterObj_1.update_callback = disaster_callback_func_v1
    DisasterObj_2.update_callback = disaster_callback_func_v1

    personObj_1.update_callback = person_callback_func_v2
    personObj_2.update_callback = person_callback_func_v2

    Anti_disasterObj.update_callback = anti_disaster_callback_func_v1
    medicalObj.update_callback = medical_callback_func_v1
    # slaveObj_1.set_default_update_func(func=slaveObj_1.update_in_temperal)

    # ====== TIME EVOLUTION FUNCTIONS SETTINGS ======
    DisasterObj_1.time_evolution_function.add_functions(disaster_evolution_func_v1)
    DisasterObj_1.time_evolution_function.add_functions(disaster_evolution_func_v2)
    DisasterObj_1.time_devolution_function.add_functions(disaster_devolution_func_v1)

    # DisasterObj_2.time_evolution_function.add_functions(disaster_evolution_func_v1)
    # DisasterObj_2.time_evolution_function.add_functions(disaster_evolution_func_v2)
    DisasterObj_2.time_devolution_function.add_functions(disaster_devolution_func_v1)

    personObj_1.time_evolution_function.add_functions(personlife_evolution_func_v2)
    personObj_2.time_evolution_function.add_functions(personlife_evolution_func_v2)

    Anti_disasterObj.time_evolution_function.add_functions(anti_disaster_evoultion_func_v1)
    medicalObj.time_evolution_function.add_functions(medical_evolution_func_v1)
    personObj_2.time_devolution_function.add_functions(personlife_devolution_func_v1)

    # ====== TIME EVOLUTION PARAMETERS INIT ======
    DisasterObj_1.time_evolution_function.params = [DisasterObj_1.get_value(), DisasterObj_1.grad, DisasterObj_1.total_sum, DisasterObj_1.current_sum, [Anti_disasterObj.get_value()]]
    DisasterObj_2.time_evolution_function.params = [DisasterObj_2.get_value(), DisasterObj_2.grad,
                                                    DisasterObj_2.total_sum, DisasterObj_2.current_sum,
                                                    [Anti_disasterObj.get_value()]]
    personObj_1.time_evolution_function.params = [personObj_1.get_value(), personObj_1.grad, personObj_1.total_sum, personObj_1.current_sum, [DisasterObj_1.get_value(personObj_1.get_pt_pos())]]
    personObj_2.time_evolution_function.params = [personObj_2.get_value(), personObj_2.grad, personObj_2.total_sum, personObj_2.current_sum, [DisasterObj_1.get_value(personObj_1.get_pt_pos())]]

    Anti_disasterObj.time_evolution_function.params = [Anti_disasterObj.get_value(), Anti_disasterObj.grad, Anti_disasterObj.total_sum, Anti_disasterObj.current_sum, [0]]
    # slaveObj_2.set_default_update_func(func=slaveObj_2.update_in_temperal)
    medicalObj.time_evolution_function.params = [medicalObj.get_value(), medicalObj.grad, medicalObj.total_sum, medicalObj.current_sum, [medicalObj.get_value()]]
    personObj_2.time_devolution_function.params = [personObj_2.get_value(), personObj_2.grad, personObj_2.total_sum, personObj_2.current_sum, [DisasterObj_1.get_value(personObj_1.get_pt_pos()), medicalObj.get_value(),0]] #目前只考虑了一种灾害
    # ====== CREATING OBJECT OF HAZARDBASE  =======
    HazardBaseObj = HazardBase()

    # ------ MAPPING THE SPECIFIC HAZARD ------


    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=DisasterObj_1,
        slave_hazard_object=personObj_1
    )

    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=DisasterObj_1,
        slave_hazard_object=personObj_2
    )

    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=DisasterObj_2,
        slave_hazard_object=personObj_1
    )

    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=DisasterObj_2,
        slave_hazard_object=personObj_2
    )

    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=Anti_disasterObj,
        slave_hazard_object=DisasterObj_1
    )
    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=Anti_disasterObj,
        slave_hazard_object=DisasterObj_2
    )

    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=medicalObj,
        slave_hazard_object=personObj_2
    )

    # # HazardBaseObj.hazards_mappings_list[HazardMapping.get_default_mapping_name(master_name=MasterObj.get_name(), slave_name=slaveObj_1.get_name())].set_slave_callback_function()

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig2 = plt.figure(num=2, figsize=(15, 8))

    max_step = 200

    def Evolution_plot(retval_1: np.ndarray, retval_2:np.ndarray):
        plt.subplot(2, 4, 1)
        meshval_1 = retval_1.reshape([100, 100])
        im = plt.imshow(meshval_1, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=110)
        im = plt.plot(50, 50, color='red', marker="o")
        im = plt.plot(75, 75, color='green', marker="o")
        plt.text(55,45, 'A', fontsize=12, color='red')
        plt.text(80, 70, 'B', fontsize=12, color='red')
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        cb = plt.colorbar()
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('thermal power(MW)')
        plt.title('Thermal power evolution', fontsize='small',fontweight='medium')

        plt.subplot(2, 4, 2)
        meshval_2 = retval_2.reshape([100, 100])
        # im = plt.imshow(meshval_2, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=10000)
        #im = plt.imshow((meshval_2>420)*1, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=1)
        im = plt.imshow(meshval_2, interpolation=None, cmap=plt.cm.BuGn, vmin=400, vmax=10000)
        im = plt.plot(50, 50, color='red', marker="o")
        im = plt.plot(75, 75, color='green', marker="o")
        plt.text(55,45, 'A', fontsize=12, color='red')
        plt.text(80, 70, 'B', fontsize=12, color='red')
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        cb = plt.colorbar()
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('CO concentration(PPM)')
        plt.title('CO concentration evolution', fontsize='small',fontweight='medium')


        ax1 = plt.subplot(2, 4, 3)
        im = plt.plot(x, ya, "r-")
        im = plt.plot(x, yb, "g-")
        ax1.set_xlabel('time(min)')
        ax1.set_ylabel('thermal power(MW)')
        plt.xlim(0, max_step)
        plt.ylim(DisasterObj_1.min_value, DisasterObj_1.max_value+10)
        plt.title('Thermal power for points', fontsize='small',fontweight='medium')

        ax1 = plt.subplot(2, 4, 4)
        im = plt.plot(x, yc, "r-")
        im = plt.plot(x, yd, "g-")
        ax1.set_xlabel('time(min)')
        ax1.set_ylabel('CO concentration(PPM)')
        plt.xlim(0, max_step)
        plt.ylim(DisasterObj_2.min_value, DisasterObj_2.max_value+1000)
        plt.title('CO concentration for points', fontsize='small',fontweight='medium')

        ax1 = plt.subplot(2, 4, 5)
        im = plt.plot(x, y1, "r-")
        ax1.set_xlabel('time(min)')
        ax1.set_ylabel('hit point')
        plt.xlim(0, max_step)
        plt.ylim(personObj_1.min_value, personObj_1.max_value+10)
        plt.title('Hit point of A', fontsize='small',fontweight='medium')

        ax1 = plt.subplot(2, 4, 6)
        im = plt.plot(x, y2, "g-")
        ax1.set_xlabel('time(min)')
        ax1.set_ylabel('hit point')
        plt.xlim(0, max_step)
        plt.ylim(personObj_2.min_value, personObj_2.max_value+10)
        plt.title('Hit point of B', fontsize='small',fontweight='medium')

        ax1 = plt.subplot(2, 4, 7)
        im = plt.plot(x, y_anti_d, "b-")
        ax1.set_xlabel('time(min)')
        ax1.set_ylabel('dosage of extinguishing agent')
        plt.xlim(0, max_step)
        plt.ylim(Anti_disasterObj.min_value, Anti_disasterObj.max_value+1)
        plt.title('Fire unit status', fontsize='small',fontweight='medium')

        ax1 = plt.subplot(2, 4, 8)
        im = plt.plot(x, y_medical, color="darkcyan", linestyle="-")
        ax1.set_xlabel('time(min)')
        ax1.set_ylabel('healing powers')
        plt.xlim(0, max_step)
        plt.ylim(Anti_disasterObj.min_value, Anti_disasterObj.max_value+1)
        plt.title('Medical unit status', fontsize='x-small',fontweight='medium')


        plt.subplots_adjust(wspace=0.6, hspace=0.6)
        return im

    t = np.array(list(range(0, max_step)))
    x, ya, yb, yc, yd, y1, y2, y_anti_d, y_medical = [], [], [], [], [], [], [], [], []


    def init():
        DisasterObj_1.disable_time_evolution()
        DisasterObj_1.disable_space_evolution()
        DisasterObj_2.enable_time_evolution()
        DisasterObj_2.enable_space_evolution()
        # Anti_disasterObj.enable_time_evolution()
        # Anti_disasterObj.disable_time_devolution()
        # Anti_disasterObj.set_value(value=2, mode="no_default")
        pass

    def update_point(step):
        # DisasterEvolutionObj/SlaveEvolutionObj_1/SlaveEvolutionObj_2
        HazardBaseObj.update()

        x.append(step)
        ya.append(HazardBaseObj.value_list[DisasterObj_1.get_name()][personObj_1.get_pt_pos()[0]][personObj_1.get_pt_pos()[1]])
        yb.append(HazardBaseObj.value_list[DisasterObj_1.get_name()][personObj_2.get_pt_pos()[0]][personObj_2.get_pt_pos()[1]])
        yc.append(HazardBaseObj.value_list[DisasterObj_2.get_name()][personObj_1.get_pt_pos()[0]][personObj_1.get_pt_pos()[1]])
        yd.append(HazardBaseObj.value_list[DisasterObj_2.get_name()][personObj_2.get_pt_pos()[0]][personObj_2.get_pt_pos()[1]])
        y1.append(HazardBaseObj.value_list[personObj_1.get_name()])
        y2.append(HazardBaseObj.value_list[personObj_2.get_name()])
        y_anti_d.append(HazardBaseObj.value_list[Anti_disasterObj.get_name()])
        y_medical.append(HazardBaseObj.value_list[medicalObj.get_name()])


        if step == 10:
            init_value[49:51, 49:51] = 20
            DisasterObj_1.set_value(init_value)
            # MasterObj.evolution_localmesh.mask = (init_value > 0) * 1.0
            DisasterObj_1.enable_time_evolution()
            DisasterObj_1.enable_space_evolution()

        if step == 50:
            Anti_disasterObj.set_value(2)
            Anti_disasterObj.enable_time_evolution()
            DisasterObj_1.enable_time_devolution()
            DisasterObj_1.devolution_localmesh.mask[70:75, 70:75] = 1
            DisasterObj_1.enable_space_devolution()

            DisasterObj_2.enable_time_devolution()
            DisasterObj_2.devolution_localmesh.mask[70:75, 70:75] = 1
            DisasterObj_2.enable_space_devolution()

        if step == 90:
            medicalObj.set_value(value=2, mode="no_mask")
            medicalObj.enable_time_evolution()
            personObj_2.enable_time_devolution()


        # fig2.savefig(r"D:\Project\EmergencyDeductionEngine\docs\figs\imgs\img_{:0>2d}.png".format(step))
        return Evolution_plot(HazardBaseObj.value_list[DisasterObj_1.get_name()],
                              HazardBaseObj.value_list[DisasterObj_2.get_name()])

    ani = FuncAnimation(fig2, update_point, frames=t,
                        init_func=init, interval=300, repeat=False)

    ani.save(r"F:\ffproject\EmergencyDeduce\v0.6\EmergencyDeductionEngine\docs\figs\fire_evolution.gif")
    # with open (r"E:\ins_project\EmergencyDeduce\v0.6\EmergencyDeductionEngine\docs\figs\multi_units_evolution_0520.html", "w") as f:
    #     print(ani.to_jshtml(), file = f)      #保存为html文件，可随时间回溯
    plt.show()
    print("===== Test accomplished! =====")
    pass



def road():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig2 = plt.figure(num=2, figsize=(5, 8))

    def Evolution_plot():
        ax1 = plt.subplot(1, 1, 1)
        # x = np.arange(0,10000,1)
        # x_test = x.reshape([100,100])
        x_ticks = np.arange(10, 40, 10)
        y_ticks = np.arange(0, 100, 10)
        plt.grid(b = True, axis = 'x')
        #x_test = np.zeros([100, 100])
        #im = plt.imshow(x_test)
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
        plt.xlabel('经度')
        plt.ylabel('纬度')
        im = plt.plot(50, 50, color='red', marker="o")
        im = plt.plot(75, 75, color='green', marker="o")
        #plt.title('灾害状态')
        #return im

    Evolution_plot()
    plt.show()
    pass

def test_delay():
    print("===== fire_scene_test =====")

    init_value = np.zeros([100, 100])
    # init_value[49:51, 49:51] = 0
    init_grad = np.ones([100, 100]) * 2
    init_dgrad = np.ones([100, 100]) * -0.1
    init_spread = [2, 2, 1]
    init_dspread = [1, 1, 1]
    total_sum = np.ones([100, 100]) * 4000
    # ===== Disaster: heat=====
    DisasterObj_1 = EvolutionBase(
        id="01",
        name="DisasterEvolutionObj_1",
        class_name="EvolutionBase",
        init_value=init_value,
        init_grad=init_grad,
        init_dgrad=init_dgrad,
        init_spread=init_spread,
        init_dspread=init_dspread,
        min_value=0,
        max_value=100,
        total_sum=total_sum,
        area=[100, 100, 100],
        stride=[2, 2, 1])
    DisasterObj_1.time_evolution_function.params = [DisasterObj_1.get_value(),  # value
                                                DisasterObj_1.grad,  # grad
                                                DisasterObj_1.total_sum,  # total sum
                                                DisasterObj_1.current_sum,  # current sum
                                                []  # input params
                                                ]
    DisasterObj_1.time_devolution_function.params = [DisasterObj_1.get_value(),  # value
                                                DisasterObj_1.grad,  # grad
                                                DisasterObj_1.total_sum,  # total sum
                                                DisasterObj_1.current_sum,  # current sum
                                                 []  # input params
                                                 ]
    DisasterObj_1.space_evolution_function.params = [DisasterObj_1.get_value(),  # value
                                                DisasterObj_1.grad,  # grad
                                                DisasterObj_1.total_sum,  # total sum
                                                DisasterObj_1.current_sum,  # current sum
                                                 []  # input params
                                                 ]
    DisasterObj_1.space_devolution_function.params = [DisasterObj_1.get_value(),  # value
                                                DisasterObj_1.grad,  # grad
                                                DisasterObj_1.total_sum,  # total sum
                                                DisasterObj_1.current_sum,  # current sum
                                                  []  # input params
                                                  ]
    DisasterObj_1.set_mode(mode="mesh")
    DisasterObj_1.evolution_localmesh.mask = (init_value > 0) * 1.0
    DisasterObj_1.devolution_localmesh.mask = np.zeros([100, 100])
    DisasterObj_1.enable_time_evolution()
    DisasterObj_1.enable_space_evolution()
    DisasterObj_1.set_default_update_func(DisasterObj_1.update)

    # ===== Disaster: co2=====
    init_value_2 = np.ones([100, 100]) * 400
    init_value_2[49:51, 49:51] = 800
    init_grad_2 = np.ones([100, 100]) * 200
    init_dgrad_2 = np.ones([100, 100]) * -0.1
    init_spread_2 = [2, 2, 1]
    init_dspread_2 = [1, 1, 1]
    total_sum_2 = total_sum * 2.3
    DisasterObj_2 = EvolutionBase(
        id="02",
        name="DisasterEvolutionObj_2",
        class_name="EvolutionBase",
        init_value=init_value_2,
        init_grad=init_grad_2,
        init_dgrad=init_dgrad_2,
        init_spread=init_spread_2,
        init_dspread=init_dspread_2,
        min_value=0,
        max_value=10000,
        total_sum=total_sum_2,
        area=[100, 100, 100],
        stride=[2, 2, 1])
    DisasterObj_2.time_evolution_function.params = [DisasterObj_2.get_value(),  # value
                                                    DisasterObj_2.grad,  # grad
                                                    DisasterObj_2.total_sum,  # total sum
                                                    DisasterObj_2.current_sum,  # current sum
                                                    []  # input params
                                                    ]
    DisasterObj_2.time_devolution_function.params = [DisasterObj_2.get_value(),  # value
                                                     DisasterObj_2.grad,  # grad
                                                     DisasterObj_2.total_sum,  # total sum
                                                     DisasterObj_2.current_sum,  # current sum
                                                     []  # input params
                                                     ]
    DisasterObj_2.space_evolution_function.params = [DisasterObj_2.get_value(),  # value
                                                     DisasterObj_2.grad,  # grad
                                                     DisasterObj_2.total_sum,  # total sum
                                                     DisasterObj_2.current_sum,  # current sum
                                                     []  # input params
                                                     ]
    DisasterObj_2.space_devolution_function.params = [DisasterObj_2.get_value(),  # value
                                                      DisasterObj_2.grad,  # grad
                                                      DisasterObj_2.total_sum,  # total sum
                                                      DisasterObj_2.current_sum,  # current sum
                                                      []  # input params
                                                      ]
    DisasterObj_2.set_mode(mode="mesh")
    DisasterObj_2.evolution_localmesh.mask = (init_value_2 > 400) * 1.0
    DisasterObj_2.devolution_localmesh.mask = np.zeros([100, 100])
    DisasterObj_2.enable_time_evolution()
    DisasterObj_2.enable_space_evolution()
    DisasterObj_2.set_default_update_func(DisasterObj_2.update)

    # ===== Anti_disaster =====
    Anti_disasterObj = EvolutionBase(
        id="04",
        name="AntiDisasterObj",
        class_name="EvolutionBase",
        init_value=0,
        init_grad=0,
        init_dgrad=-1,
        init_spread=[1, 1, 1],
        init_dspread=[1, 1, 1],
        min_value=0,
        max_value=5,
        total_sum=200,
        area=[100, 100, 100],
        stride=[2, 2, 1],
        pos=[90, 90]
    )
    Anti_disasterObj.time_evolution_function.params = [Anti_disasterObj.get_value(), Anti_disasterObj.grad, Anti_disasterObj.total_sum, Anti_disasterObj.current_sum, []]
    Anti_disasterObj.time_devolution_function.params = [Anti_disasterObj.get_value(), Anti_disasterObj.grad, Anti_disasterObj.total_sum, Anti_disasterObj.current_sum, []]
    Anti_disasterObj.space_evolution_function.params = [Anti_disasterObj.get_value(), Anti_disasterObj.grad, Anti_disasterObj.total_sum, Anti_disasterObj.current_sum, []]
    Anti_disasterObj.space_devolution_function.params = [Anti_disasterObj.get_value(), Anti_disasterObj.grad, Anti_disasterObj.total_sum, Anti_disasterObj.current_sum, []]
    Anti_disasterObj.set_mode(mode="point")
    # Anti_disasterObj.set_pos(pos=[90, 90])
    Anti_disasterObj.set_pt_pos(pt_pos=None)
    # Anti_disasterObj.set_value(value=2, mode="no_default")
    Anti_disasterObj.set_default_update_func(Anti_disasterObj.update_in_temperal)

    # ===== slaveObj_1 =====
    personObj_1 = EvolutionBase(
        id='02',
        name='SlaveEvolutionObj_1',
        class_name='EvolutionBase',
        init_value=100,
        init_grad=-1,
        init_dgrad=1,
        min_value=0,
        max_value=100,
        total_sum=100,
    )
    # Define a custom evolution function
    personObj_1.time_evolution_function.params = [personObj_1.get_value(), personObj_1.grad, personObj_1.total_sum, personObj_1.current_sum, []]  # value/grad/total sum/current sum/input params
    personObj_1.time_devolution_function.params = [personObj_1.get_value(), personObj_1.grad, personObj_1.total_sum, personObj_1.current_sum, []]  # value/grad/total sum/current sum/input params
    personObj_1.space_evolution_function.params = [personObj_1.get_value(), personObj_1.grad, personObj_1.total_sum, personObj_1.current_sum, []]  # value/grad/total sum/current sum/input params
    personObj_1.space_devolution_function.params = [personObj_1.get_value(), personObj_1.grad, personObj_1.total_sum, personObj_1.current_sum, []]  # value/grad/total sum/current sum/input params

    personObj_1.set_mode(mode="point")
    # personObj_1.set_pos(pos=[50, 50])
    personObj_1.set_pt_pos(pt_pos=[50, 50])
    personObj_1.enable_time_evolution()
    personObj_1.enable_space_evolution()
    personObj_1.set_default_update_func(func=personObj_1.update_in_temperal)

    # ===== slaveObj_2 =====
    personObj_2 = EvolutionBase(
        id='03',
        name='SlaveEvolutionObj_2',
        class_name='EvolutionBase',
        init_value=100,
        init_grad=-1,
        init_dgrad=1,
        min_value=0,
        max_value=100,
        total_sum=100,
    )
    # Define a custom evolution function
    personObj_2.time_evolution_function.params = [personObj_2.get_value(), personObj_2.grad, personObj_2.total_sum, personObj_2.current_sum, []]  # value/grad/total sum/current sum/input params
    personObj_2.time_devolution_function.params = [personObj_2.get_value(), personObj_2.grad, personObj_2.total_sum, personObj_2.current_sum, []]  # value/grad/total sum/current sum/input params
    personObj_2.space_evolution_function.params = [personObj_2.get_value(), personObj_2.grad, personObj_2.total_sum, personObj_2.current_sum, []]  # value/grad/total sum/current sum/input params
    personObj_2.space_devolution_function.params = [personObj_2.get_value(), personObj_2.grad, personObj_2.total_sum, personObj_2.current_sum, []]  # value/grad/total sum/current sum/input params

    personObj_2.set_mode(mode="point")
    # personObj_2.set_pos(pos=[75, 75])
    personObj_2.set_pt_pos(pt_pos=[75, 75])
    personObj_2.enable_time_evolution()
    personObj_2.enable_space_evolution()
    personObj_2.set_default_update_func(func=personObj_2.update_in_temperal)

    # ===== Medical unit =====
    medicalObj = EvolutionBase(
        id='05',
        name='medicalObj',
        class_name='EvolutionBase',
        init_value=0,
        init_grad=0,
        init_dgrad=0,
        min_value=0,
        max_value=100,
        total_sum=100,
    )

    medicalObj.time_evolution_function.params = [medicalObj.get_value(), medicalObj.grad, medicalObj.total_sum, medicalObj.current_sum, []]  # value/grad/total sum/current sum/input params
    medicalObj.time_devolution_function.params = [medicalObj.get_value(), medicalObj.grad, medicalObj.total_sum, medicalObj.current_sum, []]  # value/grad/total sum/current sum/input params
    medicalObj.space_evolution_function.params = [medicalObj.get_value(), medicalObj.grad, medicalObj.total_sum, medicalObj.current_sum, []]  # value/grad/total sum/current sum/input params
    medicalObj.space_devolution_function.params = [medicalObj.get_value(), medicalObj.grad, medicalObj.total_sum, medicalObj.current_sum, []]  # value/grad/total sum/current sum/input params
    medicalObj.set_mode(mode="point")
    medicalObj.set_pt_pos(pt_pos=[75, 75])
    medicalObj.enable_time_evolution()
    medicalObj.enable_space_evolution()
    medicalObj.set_default_update_func(func=medicalObj.update_in_temperal)

    def disaster_callback_func_v1(Obj: EvolutionBase = None):
        Obj.time_evolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              Obj.input_params]
        Obj.time_devolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              Obj.input_params]
        Obj.current_sum = Obj.current_sum + Obj.get_value()

    def anti_disaster_callback_func_v1(Obj: EvolutionBase = None):
        Obj.time_evolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              Obj.input_params]
        Obj.current_sum = Obj.current_sum + Obj.get_value()


    def person_callback_func_v1(Obj: EvolutionBase = None):
        Obj.time_evolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              Obj.input_params]
        Obj.time_devolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              Obj.input_params]
        Obj.current_sum = Obj.current_sum + Obj.get_value()

    def person_callback_func_v2(Obj: EvolutionBase = None):
        Obj.time_evolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              Obj.input_params]
        Obj.time_devolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              Obj.input_params]
        Obj.current_sum = Obj.current_sum + Obj.get_value()

    def medical_callback_func_v1(Obj: EvolutionBase = None):
        Obj.time_evolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum,
                                              Obj.input_params]
        Obj.current_sum = Obj.current_sum + Obj.get_value()

    def disaster_evolution_func_v1(args):
        return (args[2]-args[3])/500

    def disaster_evolution_func_v2(args):
        # print("disaster_evolution_func_v2:", args)
        return 2

    def disaster_devolution_func_v1(args):
        # print("args:", args)
        # return args[-1][0]*-1
        return -10

    def anti_disaster_evoultion_func_v1(args):     #fireman status
        # 0-value, 1-grad, 2 total_sum, 3 current_sumtmp_value
        # print("anit args:", args)
        if (args[2]-args[3]) > 0.1*args[2]:
            return 0
        else:
            # return (args[2]-args[3])/2000
            return -1

    def medical_evolution_func_v1(args):            #medicalman status
        if (args[2] - args[3] > 0.1*args[2]):
            return 0
        else:
            return -1

    def personlife_evolution_func_v1(args):
        # print("personlife args:", args)
        tmp = args[-1][0]
        if tmp > 90:
            # print("tmp>90")
            return -5
        elif tmp > 70 and tmp <= 90:
            # print("tmp>70")
            return -3
        elif tmp > 50 and tmp <= 70:
            # print("tmp>50")
            return -2
        elif tmp > 20 and tmp <= 50:
            # print("tmp>20")
            return -1
        elif tmp > 0 and tmp <= 20:
            # print("tmp>0")
            return -0.5
        else:
            return 0

    def personlife_evolution_func_v2(args):
        if args[-1][0]<25 or args[-1][1]<2500:
            return 0
        else:
            return -(0.5 * (0.8 * float(args[-1][0]) / 10 + 0.2 * float(args[-1][1]) / 1000))


    def personlife_devolution_func_v1(args):
        #print("personlife_devolution_func_v1:", args)
        return args[-1][2]*0.4
        #return 0.8


    # ====== CALLBACK FUNCTION SETTINGS ======
    DisasterObj_1.update_callback = disaster_callback_func_v1
    DisasterObj_2.update_callback = disaster_callback_func_v1

    personObj_1.update_callback = person_callback_func_v2
    personObj_2.update_callback = person_callback_func_v2

    Anti_disasterObj.update_callback = anti_disaster_callback_func_v1
    medicalObj.update_callback = medical_callback_func_v1
    # slaveObj_1.set_default_update_func(func=slaveObj_1.update_in_temperal)

    # ====== TIME EVOLUTION FUNCTIONS SETTINGS ======
    DisasterObj_1.time_evolution_function.add_functions(disaster_evolution_func_v1)
    DisasterObj_1.time_evolution_function.add_functions(disaster_evolution_func_v2)
    DisasterObj_1.time_devolution_function.add_functions(disaster_devolution_func_v1)

    # DisasterObj_2.time_evolution_function.add_functions(disaster_evolution_func_v1)
    # DisasterObj_2.time_evolution_function.add_functions(disaster_evolution_func_v2)
    DisasterObj_2.time_devolution_function.add_functions(disaster_devolution_func_v1)

    personObj_1.time_evolution_function.add_functions(personlife_evolution_func_v2)
    personObj_2.time_evolution_function.add_functions(personlife_evolution_func_v2)

    Anti_disasterObj.time_evolution_function.add_functions(anti_disaster_evoultion_func_v1)
    medicalObj.time_evolution_function.add_functions(medical_evolution_func_v1)
    personObj_2.time_devolution_function.add_functions(personlife_devolution_func_v1)

    # ====== TIME EVOLUTION PARAMETERS INIT ======
    DisasterObj_1.time_evolution_function.params = [DisasterObj_1.get_value(), DisasterObj_1.grad, DisasterObj_1.total_sum, DisasterObj_1.current_sum, [Anti_disasterObj.get_value()]]
    DisasterObj_2.time_evolution_function.params = [DisasterObj_2.get_value(), DisasterObj_2.grad,
                                                    DisasterObj_2.total_sum, DisasterObj_2.current_sum,
                                                    [Anti_disasterObj.get_value()]]
    personObj_1.time_evolution_function.params = [personObj_1.get_value(), personObj_1.grad, personObj_1.total_sum, personObj_1.current_sum, [DisasterObj_1.get_value(personObj_1.get_pt_pos())]]
    personObj_2.time_evolution_function.params = [personObj_2.get_value(), personObj_2.grad, personObj_2.total_sum, personObj_2.current_sum, [DisasterObj_1.get_value(personObj_1.get_pt_pos())]]

    Anti_disasterObj.time_evolution_function.params = [Anti_disasterObj.get_value(), Anti_disasterObj.grad, Anti_disasterObj.total_sum, Anti_disasterObj.current_sum, [0]]
    # slaveObj_2.set_default_update_func(func=slaveObj_2.update_in_temperal)
    medicalObj.time_evolution_function.params = [medicalObj.get_value(), medicalObj.grad, medicalObj.total_sum, medicalObj.current_sum, [medicalObj.get_value()]]
    personObj_2.time_devolution_function.params = [personObj_2.get_value(), personObj_2.grad, personObj_2.total_sum, personObj_2.current_sum, [DisasterObj_1.get_value(personObj_1.get_pt_pos()), medicalObj.get_value(),0]] #目前只考虑了一种灾害
    # ====== CREATING OBJECT OF HAZARDBASE  =======
    HazardBaseObj = HazardBase()

    # ------ MAPPING THE SPECIFIC HAZARD ------


    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=DisasterObj_1,
        slave_hazard_object=personObj_1
    )

    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=DisasterObj_1,
        slave_hazard_object=personObj_2
    )

    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=DisasterObj_2,
        slave_hazard_object=personObj_1
    )

    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=DisasterObj_2,
        slave_hazard_object=personObj_2
    )

    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=Anti_disasterObj,
        slave_hazard_object=DisasterObj_1
    )
    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=Anti_disasterObj,
        slave_hazard_object=DisasterObj_2
    )

    HazardBaseObj.register_hazard_mapping(
        master_hazard_object=medicalObj,
        slave_hazard_object=personObj_2
    )

    # # HazardBaseObj.hazards_mappings_list[HazardMapping.get_default_mapping_name(master_name=MasterObj.get_name(), slave_name=slaveObj_1.get_name())].set_slave_callback_function()


    max_step = 200



    t = np.array(list(range(0, max_step)))
    x, ya, yb, yc, yd, y1, y2, y_anti_d, y_medical = [], [], [], [], [], [], [], [], []
    import time


    def init():
        DisasterObj_1.disable_time_evolution()
        DisasterObj_1.disable_space_evolution()
        DisasterObj_2.enable_time_evolution()
        DisasterObj_2.enable_space_evolution()
        # Anti_disasterObj.enable_time_evolution()
        # Anti_disasterObj.disable_time_devolution()
        # Anti_disasterObj.set_value(value=2, mode="no_default")
        pass

    def update_point(step):
        # DisasterEvolutionObj/SlaveEvolutionObj_1/SlaveEvolutionObj_2
        HazardBaseObj.update()
        # cTime = time.time()
        # time_gap = cTime - pTime
        # pTime = cTime
        # print(time_gap)

        x.append(step)
        ya.append(HazardBaseObj.value_list[DisasterObj_1.get_name()][personObj_1.get_pt_pos()[0]][personObj_1.get_pt_pos()[1]])
        yb.append(HazardBaseObj.value_list[DisasterObj_1.get_name()][personObj_2.get_pt_pos()[0]][personObj_2.get_pt_pos()[1]])
        yc.append(HazardBaseObj.value_list[DisasterObj_2.get_name()][personObj_1.get_pt_pos()[0]][personObj_1.get_pt_pos()[1]])
        yd.append(HazardBaseObj.value_list[DisasterObj_2.get_name()][personObj_2.get_pt_pos()[0]][personObj_2.get_pt_pos()[1]])
        y1.append(HazardBaseObj.value_list[personObj_1.get_name()])
        y2.append(HazardBaseObj.value_list[personObj_2.get_name()])
        y_anti_d.append(HazardBaseObj.value_list[Anti_disasterObj.get_name()])
        y_medical.append(HazardBaseObj.value_list[medicalObj.get_name()])


        if step == 10:
            init_value[49:51, 49:51] = 20
            DisasterObj_1.set_value(init_value)
            # MasterObj.evolution_localmesh.mask = (init_value > 0) * 1.0
            DisasterObj_1.enable_time_evolution()
            DisasterObj_1.enable_space_evolution()

        if step == 50:
            Anti_disasterObj.set_value(2)
            Anti_disasterObj.enable_time_evolution()
            DisasterObj_1.enable_time_devolution()
            DisasterObj_1.devolution_localmesh.mask[70:75, 70:75] = 1
            DisasterObj_1.enable_space_devolution()

            DisasterObj_2.enable_time_devolution()
            DisasterObj_2.devolution_localmesh.mask[70:75, 70:75] = 1
            DisasterObj_2.enable_space_devolution()

        if step == 90:
            medicalObj.set_value(value=2, mode="no_mask")
            medicalObj.enable_time_evolution()
            personObj_2.enable_time_devolution()




        #print(HazardBaseObj.value_list[DisasterObj_1.get_name()])
        # fig2.savefig(r"D:\Project\EmergencyDeductionEngine\docs\figs\imgs\img_{:0>2d}.png".format(step))
        #return Evolution_plot(HazardBaseObj.value_list[DisasterObj_1.get_name()],
                              #HazardBaseObj.value_list[DisasterObj_2.get_name()])

    pTime = time.time()
    init()
    step = 0
    time_list = []
    sum_gap = 0

    while step < 200:
        update_point(step)
        step+=1
        cTime = time.time()
        gap = cTime - pTime
        pTime = cTime
        time_list.append(gap)

    print(sum(time_list)/len(time_list)*1000)

    pass




def EvolutionTest():
    """
    A test for evolution
    :return:
    Assuming that there are several units affecting the hazard.
    """
    print("===== EvolutionBase test ======")
    # EvolutionsTestCase_01()
    # EvolutionsTestCase_02()
    # EvolutionsTestCase_03()
    # EvolutionsTestCase_04()
    # EvolutionsTestCase_05()
    # EvolutionsTestCase_06()
    # EvolutionsTestCase_07()
    # EvolutionsTestCase_08()
    # EvolutionsTestCase_09()
    # EvolutionsTestCase_10()
    # EvolutionsTestCase_11()
    # Multi_hazard_multi_peopleTest()
    fire_scene()
    # hazard_scene()
    #hazard_scene_testv1()
    #hazard_scene_testv2()
    #road()

    #test_delay()


if __name__ == "__main__":
    EvolutionTest()
