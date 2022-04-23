# !/usr/bin/python3
# Coding:   utf-8
# @File:    firefighter_unit.py
# @Time:    2022/3/26 11:10
# @Author:  Sun Zhu
# @Version: 0.0.0

from modules.simulation.unit import UnitBase

class FireFighter(UnitBase):
    """
    FireFighters
    """
    def __init__(self,
                 uid=None,
                 name=None,
                 unit_class=None,
                 pos=(0, 0, 0),
                 dir=(0, 0, 0),
                 simulatable=True,
                 firefighter_param={},
                 ):
        """
        :param fire fighter_param: fire fighter  parameters
        """
        super().__init__(
            uid=uid,
            name=name,
            unit_class=unit_class,
            pos=pos,
            dir=dir,
            simulatable=simulatable
        )
        self._params = firefighter_param