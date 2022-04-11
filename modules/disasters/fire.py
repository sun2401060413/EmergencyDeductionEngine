# !/usr/bin/python3
# Coding:   utf-8
# @File:    fire.py
# @Time:    2022/3/26 11:12
# @Author:  Sun Zhu
# @Version: 0.0.0

from modules.simulation.unit import UnitBase

class FireDisaster(UnitBase):
    """
    FireDisaster Class
    Describing the basic condition of a fire disaster
    """
    def __init__(self,
                 uid=None,
                 name=None,
                 unit_class=None,
                 pos=(0, 0, 0),
                 dir=(0, 0, 0),
                 simulateable=True,
                 fire_param={},
                 ):
        """
        :param fire_param: fire parameters
        """
        super().__init__(
            uid=uid,
            name=name,
            unit_class=unit_class,
            pos=pos,
            dir=dir,
            simulatable=simulateable
        )
        self._params = fire_param