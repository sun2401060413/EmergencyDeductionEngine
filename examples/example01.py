

import numpy as np
from modules.simulation.hazard import EvolutionBase
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


def space_evolution_func_test():
    MeshScene = LocalMeshScene(100, 100, 100, 1, 1, 1)
    MeshScene.mask = np.zeros([100, 100])
    # MeshScene.mask[49:51, 49:51] = 1
    # MeshScene.mask[74:76, 74:76] = 1
    # MeshScene.mask[:, 5:10] = 1
    # MeshScene.mask[5:10, :] = 1
    import random
    # MeshScene.mask[:, 5:10] = 1
    # MeshScene.mask[5:10, :] = 1
    # x = [random.randint(20, 99) for i in range(10)]
    # y = [random.randint(20, 99) for i in range(10)]
    # MeshScene.mask[x, y] = 1
    MeshScene.mask[49:51, 49:51] = 1

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig2 = plt.figure(num=2, figsize=(4, 4))

    kernel_up = np.array(([0.25, 0.5, 0.25], [0, 0, 0], [0, 0, 0]), dtype="float32")
    kernel_down = np.flip(kernel_up, axis=0)
    kernel_left = kernel_up.T
    kernel_right = kernel_down.T

    # print(kernel_up, kernel_down, kernel_left, kernel_right)
    kernels = np.ones([3, 3])
    kernels[1, 0] = 5
    kernels[2, 0] = 5
    kernels[2, 1] = 5

    kernels_v1 = np.ones([3, 3])
    kernels_v1[1, 0] = 0.2
    kernels_v1[2, 0] = 0.1
    kernels_v1[2, 1] = 0.2
    kernels_v1[2, 2] = 0.1
    kernels_v1[1, 2] = 0.2

    def ani_plot(val):
        plt.subplot(1, 1, 1)
        im = plt.imshow(val, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=2)
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar()
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('热功率 单位(MW)')
        plt.title('热功率空间分布图')
        return im

    def init():
        pass

    def update_point(step):
        # 3*3: 94 steps;
        # 5*5: 57 steps;
        # 7*7: 42 steps;
        # 9*9: 33 steps;
        # 11*11: 28 steps;
        # 13*13: 24 steps;
        # 49*49: 11 steps;
        if step < 20:
            MeshScene.mask = default_space_evolution_func_v2(value=MeshScene.mask, stride_x=1, stride_y=1,
                                                             kernels=kernels)
        else:
            MeshScene.mask = default_space_evolution_func_v2(value=MeshScene.mask, stride_x=1, stride_y=1,
                                                             kernels=kernels_v1)
        print(step)
        # MeshScene.mask = default_space_evolution_func_v2(value=MeshScene.mask, kernels=kernels)
        if MeshScene.mask[0, 0] == 1:
            print("end step:", step)
        return ani_plot(MeshScene.mask)

    t = np.array(list(range(0, 500)))
    ani = FuncAnimation(fig2, update_point, frames=t,
                        init_func=init, interval=300, repeat=False)
    # ani.save(r"D:\Project\EmergencyDeductionEngine\docs\figs\random_pts_spreading.gif")
    plt.show()

    pass


# </editor-fold>

def HazardMappingTest():
    print("====== HazardMapping test =====")
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
    MasterObj.time_evolution_function.params = [np.array([100, 100]),   # value
                                                np.array([100, 100]),   # grad
                                                np.array([100, 100]),   # total sum
                                                np.array([100, 100]),   # current sum
                                                []                      # input params
                                            ]
    MasterObj.time_devolution_function.params = [np.array([100, 100]),   # value
                                                np.array([100, 100]),   # dgrad
                                                np.array([100, 100]),   # total sum
                                                np.array([100, 100]),   # current sum
                                                []                      # input params
                                            ]
    MasterObj.space_evolution_function.params = [np.array([100, 100]),   # value
                                                np.array([100, 100]),   # spread
                                                np.array([100, 100]),   # total sum
                                                np.array([100, 100]),   # current sum
                                                []                      # input params
                                            ]
    MasterObj.space_devolution_function.params = [np.array([100, 100]),   # value
                                                np.array([100, 100]),   # dspread
                                                np.array([100, 100]),   # total sum
                                                np.array([100, 100]),   # current sum
                                                []                      # input params
                                            ]
    MasterObj.set_mode(mode="mesh")
    MasterObj.evolution_localmesh.mask = (init_value > 0)*1.0
    MasterObj.devolution_localmesh.mask = np.zeros([100, 100])
    MasterObj.enable_time_evolution()
    MasterObj.enable_space_evolution()
    slaveObj = EvolutionBase(
        id='02',
        name='SlaveEvolutionObj',
        class_name='EvolutionBase',
        init_value=100,
        init_grad=-1,
        init_dgrad=1,
        min_value=0,
        max_value=1000,
        total_sum=100,
    )
    # Define a custom evolution function
    slaveObj.time_evolution_function.params = [0, 0, 0, 0, []]  # value/grad/total sum/current sum/input params
    slaveObj.time_devolution_function.params = [0, 0, 0, 0, []]  # value/grad/total sum/current sum/input params
    slaveObj.space_evolution_function.params = [0, 0, 0, 0, []]  # value/grad/total sum/current sum/input params
    slaveObj.space_devolution_function.params = [0, 0, 0, 0, []]  # value/grad/total sum/current sum/input params

    slaveObj.set_mode(mode="point")
    slaveObj.enable_time_evolution()
    slaveObj.enable_space_evolution()

    HazardMappingObj = HazardMapping(master_side=MasterObj, slave_side=slaveObj)

    def slave_side_callback_func_v1(Obj: EvolutionBase=None):
        Obj.time_evolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum, [Obj.input_params]]
        Obj.current_sum = Obj.current_sum + Obj.get_value()


    def slave_side_evolution_func_v1(args):
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
        elif tmp > 20 and tmp <=50:
            # print("tmp>20")
            return -1
        elif tmp > 0 and tmp <= 20:
            # print("tmp>0")
            return -0.5
        else:
            return 0

    HazardMappingObj.set_slave_callback_function(slave_side_callback_func_v1)
    HazardMappingObj.slave_side.time_evolution_function.add_functions(slave_side_evolution_func_v1)
    HazardMappingObj.set_mapping_update_functions(HazardMappingObj.master_side.update, HazardMappingObj.slave_side.update_in_temperal)

    HazardMappingObj.slave_side.input_params = HazardMappingObj.master_side.get_value([50, 50])       # init
    HazardMappingObj.slave_side.time_evolution_function.params = [0, 0, 0, 0, [HazardMappingObj.master_side.get_value([50, 50])]]
    HazardMappingObj.update_mapping(master_params=HazardMappingObj.master_side.get_value([50, 50]))

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig2 = plt.figure(num=2, figsize=(128, 108))
    x, y = [], []

    def Evolution_plot(retval: np.ndarray,):
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
        ax1.set_xlabel('时间(分钟)')
        ax1.set_ylabel('燃烧功率(兆瓦)')

        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        return im

    t = np.array(list(range(0, 120)))
    x, y1 = [], []


    def init():
        pass

    def update_point(step):
        HazardMappingObj.update_mapping(master_params=HazardMappingObj.master_side.get_value(pt_pos=[50, 50]))
        # print("retval_master:", retval_master, "retval_slave:", retval_slave)
        print("params:", HazardMappingObj.slave_side.time_evolution_function.params[-1])
        x.append(step)
        # y1.append(retval_slave)

        # fig2.savefig(r"D:\Project\EmergencyDeductionEngine\docs\figs\imgs\img_{:0>2d}.png".format(step))
        # return Evolution_plot(retval_master)

    ani = FuncAnimation(fig2, update_point, frames=t,
                        init_func=init, interval=300, repeat=False)

    # ani.save(r"D:\Project\EmergencyDeductionEngine\docs\figs\space_evolution_with_different_stride.gif")
    plt.show()

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
    EvolutionsTestCase_11()


if __name__ == "__main__":
    EvolutionTest()
