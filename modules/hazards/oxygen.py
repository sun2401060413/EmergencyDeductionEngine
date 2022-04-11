# !/usr/bin/python3
# Coding:   utf-8
# @File:    oxygen.py
# @Time:    2022/3/27 9:06
# @Author:  Sun Zhu
# @Version: 0.0.0

from modules.simulation.hazard import HazardBase

class OxygenHazard(HazardBase):

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
                 total_value=None):
        super().__init__(id, name, class_name, init_value, init_grad, max_value, min_value, max_grad, min_grad, total_value)


    def set_evolution_function(self, func=None, name=None):
        self._evolution_function = func
        self._evolution_function_name = name

    def get_evolution_function_name(self):
        return self._evolution_function_name

    def set_devolution_function(self, func=None, name=None):
        self._devolution_function = func
        self._devolution_function_name = name

    def get_devolution_function_name(self):
        return self._devolution_function_name

    def update(self):
        self._value = self._value + self._delta_evolution() + self._delta_devolution()


def OxygenHazardTest():
    OxygenHazardObj = OxygenHazard(
                    id="H01",
                    name="Test01",
                    class_name=None,
                    init_value=19,
                    init_grad=-19/60*60,
                    max_value=19,
                    min_value=0,
                    max_grad=19,
                    min_grad=0,
                    total_value=19)
    def func(x,a):
        return x+a*1
    OxygenHazardObj.set_evolution_function(func=func)
    varval = OxygenHazardObj.evolution()(19, -1)
    print(varval)

if __name__=="__main__":
    OxygenHazardTest()