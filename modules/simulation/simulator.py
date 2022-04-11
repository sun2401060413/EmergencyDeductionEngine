# !/usr/bin/python3
# Coding:   utf-8
# @File:    simulator.py
# @Time:    2022/3/25 22:58
# @Author:  Sun Zhu
# @Version: 0.0.0

class Simulator:
    """
    Simulator
    """
    def __init__(self):
        self._scene = None
        self._env = []
        self._disasters = []
        self._disasters_bearers = []
        self._disasters_resistants = []
        pass

    def start(self):
        pass

    def time(self):
        pass

    def add_disaster(self, disaster_obj):
        self._disasters.append(disaster_obj)

    def del_disaster(self, disaster_obj):
        for obj in self._disasters:
            # TODO
            # if obj.id is
            pass

    # TODO: ADD DISASTER BEARING ....

    def update(self, time, state):
        return

class Element():
    def __init__(self,
                 id = None,
                 name = None,
                 class_name =None):
        self.id = id
        self.name = name
        self.class_name = class_name

    def set_id(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_class_name(self, class_name):
        self.class_name = class_name

    def get_class_name(self):
        return self.class_name

def ElementTest():
    Obj = Element(id="01", name="Elem", class_name="Test")
    print(Obj.id, Obj.name, Obj.class_name)

if __name__=="__main__":
    ElementTest()
