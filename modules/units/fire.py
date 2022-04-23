# !/usr/bin/python3
# Coding:   utf-8
# @File:    fire.py
# @Time:    2022/3/26 11:12
# @Author:  Sun Zhu
# @Version: 0.0.0

from modules.simulation.unit import UnitBase
from modules.simulation.hazard import HazardBase

class FireDisaster(UnitBase, HazardBase):
    """
    FireDisaster Class
    Describing the basic condition of a fire disaster
    """
    def __init__(self,
                 uid=None,
                 name=None,
                 class_name=None,
                 pos=(0, 0, 0),
                 dir=(0, 0, 0),
                 simulateable=True,
                 fire_param={},
                 ):
        """
        :param fire_param: fire parameters
        """
        super().__init__(
            id=id,
            name=name,
            class_name=class_name,
            pos=pos,
            dir=dir,
            simulatable=simulateable
        )
        self._params = fire_param




def FireDisaster_test():
    FireDisasterObj = FireDisaster()
    FireDisasterObj.set_id("1")
    print(FireDisasterObj.get_id())
    pass

if __name__=="__main__":
    FireDisaster_test()