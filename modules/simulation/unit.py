# !/usr/bin/python3
# Coding:   utf-8
# @File:    unit.py
# @Time:    2022/3/26 11:58
# @Author:  Sun Zhu
# @Version: 0.0.0

from modules.simulation.simulator import Element

class UnitBase(Element):
    """
    Base class of all kinds units
    """
    def __init__(self,
                 id=None,
                 name=None,
                 class_name=None,
                 pos=(0, 0, 0),
                 dir=(0, 0, 0),
                 simulatable=True):
        """
        :param uid: unit id
        :param name: unit name
        :param unit_class: unit class
        :param pos: unit position
        :param dir: unit direction
        :param simulatable: simulatable flag
        """
        super().__init__(id, name, class_name)
        self._pos = pos
        self._dir = dir
        self._simulateable = simulatable






