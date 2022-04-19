# !/usr/bin/python3
# Coding:   utf-8
# @File:    hazard.py
# @Time:    2022/3/26 18:03
# @Author:  Sun Zhu
# @Version: 0.0.0

import numpy as np
from modules.simulation.simulator import Element
from modules.simulation.mesh import LocalMeshScene


class HazardBase(Element):
    """
    Base class of all kinds of hazards
    """
    def __init__(self,
                 id=None,
                 name=None,
                 class_name=None,
                 init_value=None,
                 init_grad=None,
                 max_value=None,
                 min_value=None,
                 max_grad=None,
                 min_grad=None,
                 total_value=None,
                 ):
        super().__init__(id, name, class_name)
        self._init_value = init_value
        self._init_grad = init_grad
        self._max_value = max_value
        self._min_value = min_value
        self._max_grad = max_grad
        self._min_grad = min_grad
        self._total_value = total_value

        self._is_stopped = False
        self._is_vanished = False

        self._mesh = None

        self._enable_hazard_human = False
        self._enable_hazard_vehicle = False
        self._enable_hazard_building = False
        self._enable_hazard_environment = False

        self._value = self._init_value
        self._grad = self._init_grad
        self._dgrad = 0
        self._hazard_human_value = 0
        self._hazard_human_grad = 0
        self._hazard_human_dgrad = 0
        self._hazard_vehicle_value = 0
        self._hazard_vehicle_grad = 0
        self._hazard_vehicle_dgrad = 0

        self._evolution_function = None
        self._evolution_function_name = None
        self._devolution_function = None
        self._devolution_function_name = None

        self._begin_time = 0
        self._end_time = None
        self._timestamp = 0
        self._step = 1

    def set_simulate_time(self, begin_time=0, end_time=None):
        self._begin_time, self._end_time = begin_time, end_time

    def get_simulate_time(self):
        return self._begin_time, self._end_time

    def set_step(self, step):
        self._step = step

    def get_step(self):
        return self._step

    # As a part of disaster
    def _delta_evolution(self):
        """
        Natural evolution
        :return:
        """
        # TODO: instenstiy evolution, affected space evolution
        if self._evolution_function is not None:
            return self._evolution_function
        else:
            return self._grad* self._step

    def set_evolution_function(self, func=None):
        pass

    def get_evolution_function_name(self):
        return self._evolution_function_name

    def _delta_devolution(self):
        """
        devolution
        :return:
        """
        # TODO: instenstiy evolution, affected space evolution
        if self._devolution_function is not None:
            return self._evolution_function
        else:
            return self._dgrad* self._step

    def set_devolution_function(self, func=None):
        pass

    def get_value(self):
        return self._value
    # for disasters_bearers
    ## for human
    def _delta_hazard_human(self):
        pass

    def set_delta_hazard_human_function(self, func=None):
        pass

    ## for vehicle
    def _delta_hazard_vehicle(self):
        pass

    def set_delta_hazard_vehicle_function(self, func=None):
        pass

    ## for building
    def _delta_hazard_building(self):
        pass

    def set_delta_hazard_building_function(self):
        pass

    ## for environment
    def _delta_hazard_environment(self):
        pass

    def set_delta_hazard_environment_function(self):
        pass
    # for disasters_resistants
    ## for human
    def _delta_remove_hazard_human(self):
        pass

    ## for vehicle
    def _delta_remove_hazard_vehicle(self):
        pass

    ## for building
    def _delta_remove_hazard_building(self):
        pass

    ## for environment
    def _delta_remove_hazard_environment(self):
        pass

    def update(self):
        pass


class EvolutionBase(Element):
    """
    """
    def __init__(self,
                 id=None,
                 name=None,
                 class_name=None,
                 init_value=0,          # single value or 2D list, as the single value refers the value of the center of the localmesh
                 init_grad=0,           # single value or 2D list, as the single value refers the value of the center of the localmesh
                 init_dgrad=0,          # single value or 2D list, as the single value refers the value of the center of the localmesh
                 init_spread=0,         # For describing the spatial distribution
                 init_dspread=0,        # For describing the spatial distribution
                 min_value=0,
                 max_value=100,
                 total_sum=None,
                 pos=[0, 0, 0],
                 area=[1, 1, 1],        # The length, width, height of effecting area (localmesh)
                 stride=[1, 1, 1],      # The mesh stride in length, width, and height axis
                 step=1,
                 end_time=None):
        super().__init__(id, name, class_name)
        # Init
        self._init_value = init_value
        self._init_grad = init_grad         # for increasing value in temporal
        self._init_dgrad = init_dgrad       # for decreasing value in temporal
        self._init_spread = init_spread     # for increasing value in spatial
        self._init_dspread = init_dspread   # for decreasing value in spatial
        self.max_value = max_value
        self.min_value = min_value
        self.total_sum = total_sum
        self.current_sum = 0
        self.pos = pos

        self.area = area                    # area: length, width, height
        self.evolution_localmesh = LocalMeshScene(area[0], area[1], area[2], stride[0], stride[1], stride[2])
        self.devolution_localmesh = LocalMeshScene(area[0], area[1], area[2], stride[0], stride[1], stride[2])
        self._mode = "point"         # mode: "point" or "mesh"
        self._enable_space_evolution = True
        self._enable_space_devolution = True

        # Current values
        self._value = self._init_value      # single value: value of the center of localmesh
        self.grad = self._init_grad
        self.dgrad = self._init_dgrad
        self._mask = np.zeros_like(self._value)
        self.spread = self._init_spread
        self.dspread = self._init_dspread
        self._pt_pos = [0, 0, 0]

        # time evolution
        self.time_evolution_function = FunctionsBase()
        self.time_devolution_function = FunctionsBase()

        # space evolution
        self.space_evolution_function = FunctionsBase()
        self.space_devolution_function = FunctionsBase()

        self.update_callback = None

        self._timestamp = 0
        self._begin_time = 0
        self._end_time = end_time
        self.step = step            # for time evolution
        self.stride = stride        # for space evolution

    def set_mode(self, mode="point"):
        self._mode = mode

    def get_mode(self):
        return self._mode

    # Enable and disable the space evolution and devolution
    def enable_space_evolution(self):
        self._enable_space_evolution = True
    def disable_space_evolution(self):
        self._enable_space_evolution = False
    def enable_space_devolution(self):
        self._enable_space_devolution = True
    def disable_space_devolution(self):
        self._enable_space_devolution = False

    def _delta_time_evolution(self):
        """
        Evolution of one step with multi-objects
        :return:
        """
        # mode : point or mesh
        if self._mode is "point":
            if len(self.time_evolution_function.functions_list) > 0:
                retval = 0
                for func in self.time_evolution_function.functions_list:
                    retval = retval + call_function(self.time_evolution_function.params, func)*self.step
                return retval
            else:
                return self.grad * self.step
        else:
            if len(self.time_evolution_function.functions_list) > 0:
                retval = np.zeros([self.area[0], self.area[1]])
                for func in self.time_evolution_function.functions_list:
                    retval = retval + call_function(self.time_evolution_function.params, func)*self.step
                return retval
            else:
                return self.grad * self.step

    def _delta_space_evolution(self):
        # mode: point or mesh
        if self._mode is "point":
            # # require the distance between target and center
            # if len(self.space_evolution_function.functions_list) > 0:
            #     retval = 0
            #     for func in self.space_evolution_function.functions_list:
            #         retval = retval + call_function(self.space_evolution_function.params, func)*self.stride
            #     return retval
            # else:
            #     return self.spread * self.stride    # HOW?
            pass
        else:
            if len(self.space_evolution_function.functions_list) > 0:
                retval = 0
                for func in self.space_evolution_function.functions_list:
                    retval = retval + call_function(self.space_evolution_function.params, func)
                return retval
            else:
                return default_space_evolution_func(self.evolution_localmesh.mask,
                                                    center_x_idx=self.evolution_localmesh.ct_x*self.stride[1],
                                                    center_y_idx=self.evolution_localmesh.ct_y*self.stride[0],
                                                    stride_x=self.spread[1], stride_y=self.spread[0],
                                                    enable=self._enable_space_evolution)

    def _delta_time_devolution(self):
        # mode: point or mesh
        if self._mode is "point":
            if len(self.time_devolution_function.functions_list) > 0:
                retval = 0
                for func in self.time_devolution_function.functions_list:
                    retval = retval + call_function(self.time_devolution_function.params, func)*self.step
                return retval
            else:
                return self.dgrad * self.step
        else:
            if len(self.time_devolution_function.functions_list) > 0:
                retval = np.zeros([self.area[0], self.area[1]])
                for func in self.time_devolution_function.functions_list:
                    retval = retval + call_function(self.time_devolution_function.params, func)*self.step
                return retval
            else:
                return self.dgrad * self.step

    def _delta_space_devolution(self):
        # mode: point or mesh
        if self._mode is "point":
            if len(self.space_devolution_function.functions_list) > 0:
                retval = 0
                for func in self.space_devolution_function.functions_list:
                    retval = retval + call_function(self.space_devolution_function.params, func)*self.stride
                return retval
            else:
                return self.dspread * self.stride
        else:
            if len(self.space_devolution_function.functions_list) > 0:
                retval = 0
                for func in self.space_devolution_function.functions_list:
                    retval = retval + call_function(self.space_devolution_function.params, func)*self.stride
                return retval
            else:
                return default_space_evolution_func(self.devolution_localmesh.mask,
                                                    center_x_idx=self.devolution_localmesh.ct_x*self.stride[1],
                                                    center_y_idx=self.devolution_localmesh.ct_y*self.stride[0],
                                                    stride_x=self.dspread[1], stride_y=self.dspread[0],
                                                    enable=self._enable_space_devolution)

    def update(self):
        # TODO: distribution in space
        # if self._mode is "point":
        #     self._value = np.round(np.clip(self._value + self._delta_time_evolution() + self._delta_time_devolution(), a_min=self.min_value, a_max=self.max_value), 3)
        #     if self.update_callback is not None:
        #         call_function(self, self.update_callback)
        #     return self._value
        # else:
        #     self._value = np.round(np.clip(self._value + self._delta_time_evolution() + self._delta_time_devolution(), a_min=self.min_value, a_max=self.max_value), 3)
        #     if self.update_callback is not None:
        #         call_function(self, self.update_callback)
        #     return self._value

        # self._value = np.round(np.clip(self._value + np.multiply(self._delta_time_evolution(), self._mask) + np.multiply(self._delta_time_devolution(), self._mask), a_min=self.min_value, a_max=self.max_value), 3)
        self._value = np.round(np.clip(self._value + np.multiply(self._delta_time_evolution(), self._mask) + np.multiply(self._delta_time_devolution(), self.devolution_localmesh.mask), a_min=self.min_value, a_max=self.max_value), 3)
        self.evolution_localmesh.mask, self.devolution_localmesh.mask = self._delta_space_evolution(), self._delta_space_devolution()
        self._mask = np.clip(self.evolution_localmesh.mask - self.devolution_localmesh.mask, a_min=0, a_max=1)
        if self.update_callback is not None:
            call_function(self, self.update_callback)
        return self._value

    def set_pt_pos(self, pt_pos=[0, 0, 0]):
        """
        :param pt_pos: point position for calculation
        :return: None
        """
        self._pt_pos = pt_pos

    def get_pt_pos(self):
        return self._pt_pos

    def set_value(self, value):
        self._value = value

    def get_value(self, pt_pos=None):
        if pt_pos is None:
            return self._value
        else:
            self._value             # TODO: SAME

    def set_mask(self, mask):
        self._mask = mask

    def get_mask(self):
        return self._mask

class FunctionsBase:
    """
    Evolution Functions
    """
    def __init__(self):
        self.params = []
        self.functions_list = []

    def add_functions(self, func):
        self.functions_list.append(func)

    def delete_functions(self, func):
        self.functions_list.remove(func)

    def clear_functions(self):
        self.functions_list = []


def call_function(args, f):
    """Callback function"""
    return f(args)

def default_space_evolution_func(value, center_x_idx=0, center_y_idx=0, center_z_idx=0, mode="2D", stride_x=1, stride_y=1, stride_z=1, enable=True):
    if not enable:
        return value
    else:
        stride_value = value.copy()
        center_value = np.max(stride_value[center_x_idx-stride_x:center_x_idx+stride_x, center_y_idx-stride_y:center_y_idx+stride_y])
        stride_value[center_y_idx-stride_y: center_y_idx+stride_y+1, center_x_idx-stride_y: center_x_idx+stride_y+1] = center_value
        h_offset, v_offset, hv_offset = stride_value.copy(), stride_value.copy(), stride_value.copy()
        if mode is "2D":
            h_offset[:, 0:center_x_idx-stride_x] = h_offset[:, stride_x:center_x_idx]           # x=4, 0:2, 2:4, [0, 1, 2, 3, 4, 5, 6, 7, 8]
            h_offset[:, center_x_idx + stride_x:-1] = h_offset[:, center_x_idx:-1*stride_x - 1]   # x=4, 6:8, 4:6
            # print(h_offset)

            v_offset[0:center_y_idx - stride_y, :] = v_offset[stride_y: center_y_idx, :]
            v_offset[center_y_idx + stride_y:-1, :] = v_offset[center_y_idx:-1*stride_y - 1, :]
            # print(v_offset)

            hv_offset[:, 0:center_x_idx - stride_x] = hv_offset[:, stride_x:center_x_idx]
            hv_offset[:, center_x_idx + stride_x:-1] = hv_offset[:, center_x_idx:-1*stride_x - 1]
            hv_offset[0:center_y_idx - stride_y, :] = hv_offset[stride_y: center_y_idx, :]
            hv_offset[center_y_idx + stride_y:-1, :] = hv_offset[center_y_idx:-1*stride_y - 1, :]
            # print(hv_offset)

            evolution_value = 0.25 * h_offset + 0.25 * v_offset + 0.5 * hv_offset
        return evolution_value

# ===== TEST CASE =====


def update_callback_test(Obj : EvolutionBase):
    Obj.time_evolution_function.params = [Obj.get_value()]


# <editor-fold desc="Test functions">
def EvolutionsTestCase_01():
    print("----- Single point test: without evolution functions -----")
    EvolutionBaseObj = EvolutionBase(id="01",
                                     name="EvolutionTest01",
                                     class_name="Hazardbase",
                                     init_value=0,
                                     init_grad=10,
                                     init_dgrad=-5,
                                     init_spread=-0.01,
                                     init_dspread=-0.01,
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
        return args[-1] if args[0] > 0 else 0

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

# </editor-fold>


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
        # Obj.time_evolution_function.params = [Obj.get_value()] # PASS
        Obj.time_evolution_function.params = [(Obj.total_sum - Obj.current_sum) / 10, Obj.grad]
        Obj.current_sum = Obj.current_sum + Obj.get_value()
        # Obj.localmesh.mask = (Obj.get_value() > 0) * 1
        # Obj.localmesh.mask = (Obj.init_value > 0) * 1
        pass

    EvolutionBaseObj.update_callback = update_callback

    def Ev_func1(args):
        return args[0] / 100
        # return 1

    def Ev_func2(args):
        return -5

    EvolutionBaseObj.time_evolution_function.add_functions(Ev_func1)


    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig2 = plt.figure(num=2, figsize=(128, 108))
    x, y = [], []

    pt_view= [[0, 0], [25, 25], [50, 50]]

    def Evolution_plot(retval: np.ndarray, evolution_mask:np.ndarray, devolution_mask:np.ndarray, mask:np.ndarray):
        plt.subplot(2, 3, 1)
        meshval = retval.reshape([100, 100])
        im = plt.imshow(meshval, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=110)
        im = plt.plot(pt_view[0][0], pt_view[0][1], "o", color="r")
        im = plt.plot(pt_view[1][0], pt_view[1][1], "o", color="g")
        im = plt.plot(pt_view[2][0], pt_view[2][1], "o", color="b")
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar()
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('热功率 单位(MW)')
        plt.title('热功率空间分布图')

        plt.subplot(2, 3, 2)
        im = plt.imshow(evolution_mask, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=1)
        im = plt.plot(pt_view[0][0], pt_view[0][1], "o", color="r")
        im = plt.plot(pt_view[1][0], pt_view[1][1], "o", color="g")
        im = plt.plot(pt_view[2][0], pt_view[2][1], "o", color="b")
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar()
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('影响程度')
        plt.title('EvolutionMask')

        plt.subplot(2, 3, 3)
        im = plt.imshow(devolution_mask, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=1)
        im = plt.plot(pt_view[0][0], pt_view[0][1], "o", color="r")
        im = plt.plot(pt_view[1][0], pt_view[1][1], "o", color="g")
        im = plt.plot(pt_view[2][0], pt_view[2][1], "o", color="b")
        plt.xlabel('经度方向坐标x')
        plt.ylabel('纬度方向坐标y')
        cb = plt.colorbar()
        plt.xticks(np.arange(0, 100, 10))  # fixed
        plt.yticks(np.arange(0, 100, 10))  # fixed
        cb.set_label('影响程度')
        plt.title('DevolutionMask')

        plt.subplot(2, 3, 4)
        im = plt.imshow(mask, interpolation=None, cmap=plt.cm.BuGn, vmin=0, vmax=1)
        im = plt.plot(pt_view[0][0], pt_view[0][1], "o", color="r")
        im = plt.plot(pt_view[1][0], pt_view[1][1], "o", color="g")
        im = plt.plot(pt_view[2][0], pt_view[2][1], "o", color="b")
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

    t = np.array(list(range(0, 150)))
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

        pass

    def update_point(step):
        retval = EvolutionBaseObj.update()
        x.append(step)
        y1.append(retval[pt_view[0][0]][pt_view[0][1]])
        y2.append(retval[pt_view[1][0]][pt_view[1][1]])
        y3.append(retval[pt_view[2][0]][pt_view[2][1]])

        if step == 20:
            # tmp = EvolutionBaseObj.get_mask()
            # tmp[60: 70, 60: 70] = 1
            # EvolutionBaseObj.set_mask(tmp)
            EvolutionBaseObj.devolution_localmesh.mask[60:70, 60:70] = 1
            EvolutionBaseObj.time_devolution_function.add_functions(Ev_func2)
        if step == 80:
            EvolutionBaseObj.disable_space_devolution()
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

    # ani.save(r"D:\Project\EmergencyDeductionEngine\docs\figs\space_evolution_with_different_stride.gif")
    plt.show()





if __name__=="__main__":
    EvolutionTest()
    pass