# !/usr/bin/python3
# Coding:   utf-8
# @File:    hazard.py
# @Time:    2022/3/26 18:03
# @Author:  Sun Zhu
# @Version: 0.0.0
import logging

import numpy as np
import cv2
from modules.simulation.simulator import Element
from modules.simulation.mesh import LocalMeshScene

# output_Hazard
# input_Hazard


class EvolutionBase(Element):
    """
    """
    def __init__(self,
                 id: str = None,
                 name: str = None,
                 class_name: str = None,
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
        self.current_sum = np.zeros_like(total_sum)
        self.pos = pos
        self.input_params: list = None

        self.area = area                    # area: length, width, height
        self.evolution_localmesh = LocalMeshScene(area[0], area[1], area[2], stride[0], stride[1], stride[2])
        self.devolution_localmesh = LocalMeshScene(area[0], area[1], area[2], stride[0], stride[1], stride[2])
        self._mode = "point"         # mode: "point" or "mesh"
        self._enable_time_evolution = True
        self._enable_time_devolution = False
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
        self.spread_kernel = np.ones([self.spread[0]*2+1, self.spread[1]*2+1]) if self._mode is not "point" else None
        self.dspread_kernel = np.ones([self.dspread[0]*2+1, self.dspread[1]*2+1]) if self._mode is not "point" else None
        self.retval_list = []


        # time evolution
        self.time_evolution_function = FunctionsBase()
        self.time_devolution_function = FunctionsBase()

        # space evolution
        self.space_evolution_function = FunctionsBase()
        self.space_devolution_function = FunctionsBase()

        self.update_callback = None
        self.default_update_func = self.update_in_temperal if self._mode is "point" else self.update

        self._timestamp = 0
        self._begin_time = 0
        self._end_time = end_time
        self.step = step            # for time evolution
        self.stride = stride        # for space evolution


    def set_mode(self, mode="point"):
        self._mode = mode
        self.spread_kernel = np.ones([self.spread[0]*2+1, self.spread[1]*2+1]) if self._mode is not "point" else None
        self.dspread_kernel = np.ones([self.dspread[0]*2+1, self.dspread[1]*2+1]) if self._mode is not "point" else None

    def get_mode(self):
        return self._mode

    # Enable and disable the time and space evolution and devolution
    def enable_time_evolution(self):
        self._enable_time_evolution = True
    def enable_time_devolution(self):
        self._enable_time_devolution = True
    def disable_time_evolution(self):
        self._enable_time_evolution = False
    def disable_time_devolution(self):
        self._enable_time_devolution = False

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
                self.retval_list = []
                for func in self.time_evolution_function.functions_list:
                    # retval = retval + call_function(self.time_evolution_function.params, func) * self.step * (self._enable_time_evolution * 1.0)
                    tmp_value = call_function(self.time_evolution_function.params, func) * self.step * (self._enable_time_evolution * 1.0)
                    self.retval_list.append(tmp_value)
                    retval = retval + tmp_value
                return retval
            else:
                return self.grad * self.step * (self._enable_time_evolution * 1.0)
        else:
            if len(self.time_evolution_function.functions_list) > 0:
                retval = np.zeros([self.area[0], self.area[1]])
                self.retval_list = []
                for func in self.time_evolution_function.functions_list:
                    # retval = retval + call_function(self.time_evolution_function.params, func) * self.step * (self._enable_time_evolution * 1.0)
                    tmp_value = call_function(self.time_evolution_function.params, func) * self.step * (
                                self._enable_time_evolution * 1.0)
                    self.retval_list.append(tmp_value)
                    retval = retval + tmp_value
                return retval
            else:
                return self.grad * self.step * (self._enable_time_evolution * 1.0)

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
                    retval = retval + call_function(self.space_evolution_function.params, func) * (self._enable_space_evolution * 1.0)
                return retval
            else:
                # return default_space_evolution_func(self.evolution_localmesh.mask,
                #                                     center_x_idx=self.evolution_localmesh.ct_x*self.stride[1],
                #                                     center_y_idx=self.evolution_localmesh.ct_y*self.stride[0],
                #                                     stride_x=self.spread[1], stride_y=self.spread[0],
                #                                     enable=self._enable_space_evolution)
                return default_space_evolution_func_v2(self.evolution_localmesh.mask,
                                                    kernel=self.spread_kernel,
                                                    stride_x=self.spread[1], stride_y=self.spread[0],
                                                    enable=self._enable_space_evolution)

    def _delta_time_devolution(self):
        # mode: point or mesh
        if self._mode is "point":
            if len(self.time_devolution_function.functions_list) > 0:
                retval = 0
                for func in self.time_devolution_function.functions_list:
                    retval = retval + call_function(self.time_devolution_function.params, func) * self.step * (self._enable_time_devolution * 1.0)
                return retval
            else:
                return self.dgrad * self.step * (self._enable_time_devolution * 1.0)
        else:
            if len(self.time_devolution_function.functions_list) > 0:
                retval = np.zeros([self.area[0], self.area[1]])
                for func in self.time_devolution_function.functions_list:
                    retval = retval + call_function(self.time_devolution_function.params, func) * self.step * (self._enable_time_devolution * 1.0)
                return retval
            else:
                return self.dgrad * self.step * (self._enable_time_devolution * 1.0)

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
                # return default_space_evolution_func(self.devolution_localmesh.mask,
                #                                     center_x_idx=self.devolution_localmesh.ct_x*self.stride[1],
                #                                     center_y_idx=self.devolution_localmesh.ct_y*self.stride[0],
                #                                     stride_x=self.dspread[1], stride_y=self.dspread[0],
                #                                     enable=self._enable_space_devolution)
                return default_space_evolution_func_v2(self.devolution_localmesh.mask,
                                                    kernel=self.dspread_kernel,
                                                    stride_x=self.dspread[1], stride_y=self.dspread[0],
                                                    enable=self._enable_space_devolution)

    def update(self):
        self._value = np.round(np.clip(self._value + np.multiply(self._delta_time_evolution(), self.evolution_localmesh.mask) + np.multiply(self._delta_time_devolution(), self.devolution_localmesh.mask), a_min=self.min_value, a_max=self.max_value), 3)
        self.evolution_localmesh.mask = self._delta_space_evolution() - self.devolution_localmesh.mask
        self.devolution_localmesh.mask = self._delta_space_devolution()
        # self._mask = np.clip(self.evolution_localmesh.mask - self.devolution_localmesh.mask, a_min=0, a_max=1)
        if self.update_callback is not None:
            call_function(self, self.update_callback)
        return self._value

    def update_in_temperal(self):
        self._value = np.round(np.clip(self._value + self._delta_time_evolution() + self._delta_time_devolution(), a_min=self.min_value, a_max=self.max_value), 3)
        if self.update_callback is not None:
            call_function(self, self.update_callback)
        return self._value

    def update_in_spatial(self):
        self.evolution_localmesh.mask = self._delta_space_evolution() - self.devolution_localmesh.mask
        self.devolution_localmesh.mask = self._delta_space_devolution()
        # self._mask = np.clip(self.evolution_localmesh.mask - self.devolution_localmesh.mask, a_min=0, a_max=1)
        if self.update_callback is not None:
            call_function(self, self.update_callback)
        return self._value

    def set_default_update_func(self, func=None):
        self.default_update_func = func

    def get_default_update(self):
        return call_function_without_params(self.default_update_func)
        # return self.default_update_func()

    def set_pt_pos(self, pt_pos=[0, 0, 0]):
        """
        :param pt_pos: point position for calculation
        :return: None
        """
        self._pt_pos = pt_pos

    def get_pt_pos(self):
        return self._pt_pos

    def set_value(self, value, mode="default"):
        self._value = value
        if mode is "default" and self._mode is "mesh":
            self.evolution_localmesh.mask = (self._value > 0)*1.0

    def get_value(self, pt_pos=None):
        if pt_pos is None:
            return self._value
        elif len(pt_pos) == 2:
            return self._value[pt_pos[0], pt_pos[1]]
        else:
            return self._value[pt_pos[0], pt_pos[1], pt_pos[2]]  # TODO: SAME

    def set_mask(self, mask):
        self._mask = mask

    def get_mask(self):
        return self._mask


class HazardBase(Element):
    """
    Base class of all kinds of hazards
    """
    def __init__(self,
                 id: str = None,
                 name: str = None,
                 class_name: str = None,
                 hazards_params: dict = None
                 ):
        super().__init__(id, name, class_name)
        self.hazards_params = hazards_params
        self.hazards_list = {}
        self.hazards_mappings_list = {}
        self.value_list = {}

    def register_hazard(self, hazard_object: EvolutionBase = None):
        """
        :param hazard_name:
        :param hazard_object:
        :return:
        """
        if hazard_object is None:
            print("The hazard_object is None")
        else:
            self.hazards_list[hazard_object.get_name()] = hazard_object
        return self.hazards_list

    def de_register_hazard(self, hazard_object: EvolutionBase = None):
        """
        :param :
        :return:
        """
        if hazard_object:
            if self.hazards_list.__contains__(hazard_object.get_name()):
                self.hazards_list.pop(hazard_object.get_name())
            else:
                print("The hazard:\"{}\" dosen't exist".format(hazard_object.get_name()))
        else:
            print("The de_register_hazard function requires a hazard_name")

    def get_hazard_by_name(self, hazard_name: str = None):
        return self.hazards_list[hazard_name]

    def register_hazard_mapping(self,
                                master_hazard_object: EvolutionBase = None,
                                slave_hazard_object: EvolutionBase = None):
        if not self.hazards_list.__contains__(master_hazard_object.get_name()):
            self.register_hazard(hazard_object=master_hazard_object)
        if not self.hazards_list.__contains__(slave_hazard_object.get_name()):
            self.register_hazard(hazard_object=slave_hazard_object)

        tmp_mapping_obj = HazardMapping(
            master_side=master_hazard_object,
            slave_side=slave_hazard_object
        )
        tmp_mapping_name = tmp_mapping_obj.get_mapping_name()
        self.hazards_mappings_list[tmp_mapping_name] = tmp_mapping_obj

    def de_register_hazard_mapping(self,
                                   master_hazard_object: EvolutionBase = None,
                                   slave_hazard_object: EvolutionBase = None):

        tmp_mapping_name = HazardMapping.get_default_mapping_name(master_name=master_hazard_object.get_name(),
                                                                  slave_name=slave_hazard_object.get_name())
        if self.hazards_mappings_list.__contains__(tmp_mapping_name):
            self.hazards_mappings_list.pop(tmp_mapping_name)

    def get_hazard_mapping_by_name(self, master_hazard_name: str = None, slave_hazard_name: str = None):
        return self.hazards_mappings_list[HazardMapping.get_default_mapping_name(master_name=master_hazard_name, slave_name=slave_hazard_name)]

    def parse_parameters(self):
        """
        :return:
        """
        if self.hazards_params:
            pass
        else:
            pass

    def update(self):
        # ==== clear last params =====
        for hazard in self.hazards_list:
            self.hazards_list[hazard].input_params = []

        # ==== update current params ====
        for hazard_mapping in self.hazards_mappings_list:
            if self.hazards_mappings_list[hazard_mapping].master_side.get_mode() is "mesh":
                self.hazards_mappings_list[hazard_mapping].update_mapping(
                    master_params=self.hazards_mappings_list[hazard_mapping].master_side.get_value(
                        self.hazards_mappings_list[hazard_mapping].slave_side.get_pt_pos()
                    )
                )
            else:
                self.hazards_mappings_list[hazard_mapping].update_mapping(
                    master_params=self.hazards_mappings_list[hazard_mapping].master_side.get_value()
                )

        # ==== update current value ====
        self.value_list = {}
        for hazard in self.hazards_list:
            self.value_list[self.hazards_list[hazard].get_name()] = self.hazards_list[hazard].get_default_update()

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


class HazardMapping(Element):
    """
    Base class of all hazard mappings
    """
    def __init__(self,
                 id: str = None,
                 name: str = None,
                 class_name: str = None,
                 master_side: EvolutionBase = None,
                 slave_side: EvolutionBase = None,
                 ):
        super().__init__(id, name, class_name)
        self.master_side = master_side
        self.slave_side = slave_side

        self.master_update_func = None
        self.slave_update_func = None

    def set_master_side(self, Obj:EvolutionBase=None):
        self.master_side = Obj

    def get_master_side(self):
        return self.master_side

    def set_slave_side(self, Obj:EvolutionBase=None):
        self.slave_side = Obj

    def get_slave_side(self):
        return self.slave_side

    def add_mapping_function(self, slave_func_Obj:FunctionsBase=None, func=None):
        slave_func_Obj.add_functions(func)

    def delete_mapping_function(self, slave_func_Obj: FunctionsBase=None, func=None):
        slave_func_Obj.delete_functions(func)

    def set_master_callback_function(self, func=None):
        """
        :param func:
        :return:
        """
        if func:
            self.master_side.update_callback = func
        else:
            self.master_side.update_callback = self.defalut_mapping_callback

    def set_slave_callback_function(self, func=None):
        """
        :param func:
        :return:
        """
        if func:
            self.slave_side.update_callback = func
        else:
            self.slave_side.update_callback = self.defalut_mapping_callback

    def update_mapping(self, master_params: list=None):
        # self.slave_side.input_params = master_params
        self.slave_side.input_params.append(master_params)
        # return call_function_without_params(self.master_update_func), call_function_without_params(self.slave_update_func)

    def set_mapping_update_functions(self, master_func=None, slave_func=None):
        self.master_update_func = master_func
        self.slave_update_func = slave_func

    def get_mapping_name(self):
        if self.master_side and self.slave_side:
            return self.get_default_mapping_name(master_name=self.master_side.name,
                                                 slave_name=self.slave_side.name)
        else:
            return None

    @staticmethod
    def get_default_mapping_name(master_name: str = None, slave_name: str = None):
        return "{0}_{1}".format(master_name, slave_name)

    @staticmethod
    def defalut_mapping_callback(Obj:EvolutionBase):
        Obj.time_evolution_function.params = [Obj.get_value(), Obj.grad, Obj.total_sum, Obj.current_sum, Obj.input_params]
        Obj.time_devolution_function.params = [Obj.get_value(), Obj.dgrad, Obj.total_sum, Obj.current_sum, Obj.input_params]
        Obj.space_evolution_function.params = [Obj.get_value(), Obj.spread, Obj.total_sum, Obj.current_sum, Obj.input_params]
        Obj.space_devolution_function.params = [Obj.get_value(), Obj.dspread, Obj.total_sum, Obj.current_sum, Obj.input_params]
        Obj.current_sum = Obj.current_sum + Obj.get_value()
        pass



def call_function(args, f):
    """Callback function"""
    return f(args)

def call_function_without_params(f):
    return f()


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


def default_space_evolution_func_v2(value, kernel=None, stride_x=1, stride_y=1, stride_z=1, enable=True):
    if not enable:
        return value
    else:
        retval = cv2.filter2D(value, -1, kernel)
        retval = np.clip(value + retval/((stride_x*2+1)*(stride_y*2+1)-1), a_min=0, a_max=1)
        return retval
# ===== TEST CASE =====


def update_callback_test(Obj : EvolutionBase):
    Obj.time_evolution_function.params = [Obj.get_value()]

# ===== HazardBaseTest =====
def HazardBaseTest_v1():
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
    fig2 = plt.figure(num=2, figsize=(128, 108))

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

def HazardBaseTest_v2():
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

def HazardBaseTest():
    print("===== HazardBaseTest =====")
    # HazardBaseTest_v1()
    # HazardBaseTest_v2()
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

    def anti_disaster_evoultion_func_v1(args):
        # 0-value, 1-grad, 2 total_sum, 3 current_sumtmp_value
        # print("anit args:", args)
        if (args[2]-args[3]) > 0.1*args[2]:
            return 0
        else:
            # return (args[2]-args[3])/2000
            return -1

    def medical_evolution_func_v1(args):
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


if __name__=="__main__":
    # EvolutionTest()
    # space_evolution_func_test()
    HazardBaseTest()
    # HazardMappingTest()
    # TODO: master:


    pass