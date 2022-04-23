# !/usr/bin/python3
# Coding:   utf-8
# @File:    person.py
# @Time:    2022/3/26 11:58
# @Author:  Sun Zhu
# @Version: 0.0.0

from modules.simulation.unit import UnitBase

class PersonDisastersBearers(UnitBase):
    """
    Normal Person Class
    """
    def __init__(self,
                 uid=None,
                 name=None,
                 unit_class=None,
                 pos=(0, 0, 0),
                 dir=(0, 0, 0),
                 simulatable=True,
                 person_param={},
                 ):
        """
        :param person_param: person parameters
        """
        super().__init__(
            uid=uid,
            name=name,
            unit_class=unit_class,
            pos=pos,
            dir=dir,
            simulatable=simulatable
        )
        self._params = person_param

