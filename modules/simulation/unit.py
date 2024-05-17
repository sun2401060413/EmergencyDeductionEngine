from modules.simulation.simulator import Element

class UnitBase(Element):
    """
    Base class of all kinds units
    """
    def __init__(self,
                 id: str = None,
                 name: str = None,
                 class_name: str = None,
                 pos: list = [0, 0, 0],
                 dir: list = [0, 0, 0],
                 simulatable: bool = True):
        """
        :param id: unit id
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






