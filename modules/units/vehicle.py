# !/usr/bin/python3
# Coding:   utf-8
# @File:    vehicle.py
# @Time:    2022/3/26 11:59
# @Author:  Sun Zhu
# @Version: 0.0.0

from modules.simulation.unit import UnitBase

class Vehicle(UnitBase):
    """
    Normal Vehicles Class
    """
    def __init__(self,
                 id=None,
                 name=None,
                 class_name=None,
                 pos=(0, 0, 0),
                 dir=(0, 0, 0),
                 simulatable=True,
                 vehicle_param={},
                 ):
        """
        :param vehicle_param: vehicle parameters
        """
        super().__init__(
            id=id,
            name=name,
            class_name=class_name,
            pos=pos,
            dir=dir,
            simulatable=simulatable
        )
        self._params = vehicle_param