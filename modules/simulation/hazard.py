
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
        self.input_params_tep: list = None

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
                    retval = retval + call_function(self.time_evolution_function.params, func) * self.step * (self._enable_time_evolution * 1.0)
                    # tmp_value = call_function(self.time_evolution_function.params, func) * self.step * (self._enable_time_evolution * 1.0)
                    # self.retval_list.append(tmp_value)
                    # retval = retval + tmp_value
                return retval
            else:
                return self.grad * self.step * (self._enable_time_evolution * 1.0)
        else:
            if len(self.time_evolution_function.functions_list) > 0:
                retval = np.zeros([self.area[0], self.area[1]])
                self.retval_list = []
                for func in self.time_evolution_function.functions_list:
                    retval = retval + call_function(self.time_evolution_function.params, func) * self.step * (self._enable_time_evolution * 1.0)
                    # tmp_value = call_function(self.time_evolution_function.params, func) * self.step * (
                    #             self._enable_time_evolution * 1.0)
                    # self.retval_list.append(tmp_value)
                    # retval = retval + tmp_value
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
        #print(self.hazards_list[hazard].input_params)

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
        #self.slave_side.input_params = master_params
        self.slave_side.input_params.append(master_params)
        return self.slave_side.input_params
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




if __name__=="__main__":
    # EvolutionTest()
    # space_evolution_func_test()
    # HazardBaseTest()
    # HazardBaseTest_v1()
    # HazardMappingTest()
    # TODO: master:


    pass