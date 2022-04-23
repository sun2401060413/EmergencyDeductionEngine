# !/usr/bin/python3
# Coding:   utf-8
# @File:    medical_unit.py
# @Time:    2022/3/26 11:09
# @Author:  Sun Zhu
# @Version: 0.0.0

from modules.simulation.unit import UnitBase

class MedicalUnit(UnitBase):
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
                 medical_unit_param={},
                 ):
        """
        :param medical_unit_param: Medical Unit  parameters
        """
        super().__init__(
            uid=uid,
            name=name,
            unit_class=unit_class,
            pos=pos,
            dir=dir,
            simulatable=simulatable
        )
        self._params = medical_unit_param