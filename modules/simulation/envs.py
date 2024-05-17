

import numpy as np
from modules.simulation.mesh import MeshScene
from modules.simulation.simulator import Element

class EnvBase(Element):
    """
    Base class of all kinds of environments
    """
    def __init__(self,
                 id=None,
                 name=None,
                 class_name=None,
                 pos=(0, 0, 0),
                 radius=None,
                 center_value=None,
                 outer_value=None,
                 simulatable=True):
        """
        :param eid: env id
        :param name: env name
        :param env_class: env class
        :param pos: position
        :param radius: range
        :param center_value: value on the center
        :param outer_value: value outside the radius
        :param simulatable: simulatable flag
        """
        super().__init__(id, name, class_name)
        self.pos = pos
        self.radius = radius
        self.center_value=center_value
        self.outer_value=outer_value
        self.simulatable = simulatable
        self.step = 1

        # For setting params sequence
        self._pos_params_seq = []
        self._radius_params_seq = []
        self._center_value_params_seq = []
        self._outer_value_params_seq = []
        self._time_value_params_seq = []

        # For saving the interpolation of params
        self._pos_seq = []
        self._radius_seq = []
        self._center_value_seq = []
        self._outer_value_seq = []
        self._time_value_seq = []

        self._is_value_seq_exist = False    # indicates whether the interpolation sequence exists

        # For update
        self._time_stamp = 0
        self._pt_pos = None
        self._pt_mode = "point"

    def set_meshes(self, pt_pos):
        self._pt_pos = pt_pos

    def set_pt_mode(self, mode="point"):
        self._pt_mode = mode

    def set_params(self, pos, radius, center_value, outer_value):
        """
        To get a transient value
        :param pos:
        :param radius:
        :param center_value:
        :param outer_value:
        :return:
        """
        self.pos = pos
        self.radius = radius
        self.center_value = center_value
        self.outer_value = outer_value
        pass

    def set_params_sequence(self,
                           pos_params_seq = [],
                           radius_params_seq = [],
                           center_value_params_seq = [],
                           outer_value_params_seq = [],
                           time_value_params_seq = []):
        """
        To get value sequence by using interpolation
        :param pos_params_seq: sequence of the key position parameters
        :param radius_params_seq: sequence of the key radius parameters
        :param center_value_params_seq: sequence of the key center value parameters
        :param outer_value_params_seq: sequence of the key outer value parameters
        :param time_value_params_seq: sequence of the key time value parameters
        :return:
        """
        self._pos_params_seq = pos_params_seq
        self._radius_params_seq = radius_params_seq
        self._center_value_params_seq = center_value_params_seq
        self._outer_value_params_seq = outer_value_params_seq
        self._time_value_params_seq = time_value_params_seq

        self.get_interpolation_params_sequence()
        pass

    def get_value(self, pt_pos=None, mode="point"):
        """
        :param pt_pos: points position for calculation
        :param mode: some options for calculation:
            - "point": evaluate the value for a single point;
            - "mesh" : evaluate the values for mesh grid;
        :return:
        """
        if self.radius is None or self.center_value is None or self.outer_value is None:
            return None

        # 2D calculation, TODO: 3D calculation
        if mode is "point":
            pts = [pt_pos]
        else:
            pts = pt_pos

        retval = []
        for i in range(len(pts)):
            distance = np.linalg.norm(np.array([self.pos[0], self.pos[1]]) - np.array([pts[i][0], pts[i][1]]))
            val = round(self.center_value + (self.outer_value - self.center_value)*distance / self.radius, 1)
            val = self.outer_value if distance > self.radius else val
            retval.append(val)
        return retval

    def get_value_sequence(self, pt_pos=None, cur_t=0, mode="point"):
        """
        :param pt_pos: points position for calculation
        :param cur_t: current time for calculation
        :param mode: calculation with point or mesh
        :return:
        """
        # 2D calculation
        if mode is "point":
            pts = [pt_pos]
        else:
            pts = pt_pos

        if not self._is_value_seq_exist:
            pass
        retval = []

        if not self._is_value_seq_exist:
            self.get_interpolation_params_sequence()

        self.set_params(pos=self._pos_seq[cur_t],
                        radius=self._radius_seq[cur_t],
                        center_value=self._center_value_seq[cur_t],
                        outer_value=self._outer_value_seq[cur_t])
        return self.get_value(pt_pos=pt_pos, mode=mode)

    def get_interpolation_params_sequence(self):
        if self._is_value_seq_exist:
            return

        # Processing the condition that the first key point of time is not "0"
        if self._time_value_params_seq[0] > 0:
            self._pos_params_seq.insert(0, self._pos_params_seq[0])
            self._radius_params_seq.insert(0, self._radius_params_seq[0])
            self._center_value_params_seq.insert(0, self._center_value_params_seq[0])
            self._outer_value_params_seq.insert(0, self._outer_value_params_seq[0])
            self._time_value_params_seq.insert(0, self._time_value_params_seq[0])


        self._pos_seq = []
        self._radius_seq = []
        self._center_value_seq = []
        self._outer_value_seq = []
        self._time_value_seq = []

        # Generate the time sequence firstly
        self._time_value_seq = range(min(self._time_value_params_seq), max(self._time_value_params_seq)+1)
        for i in range(len(self._time_value_params_seq)-1):
            self._pos_seq.extend(np.linspace(self._pos_params_seq[i],
                                             self._pos_params_seq[i+1],
                                             self._time_value_params_seq[i+1]-self._time_value_params_seq[i]+1))
            self._radius_seq.extend(np.linspace(self._radius_params_seq[i],
                                                self._radius_params_seq[i+1],
                                                self._time_value_params_seq[i+1]-self._time_value_params_seq[i]+1))
            self._center_value_seq.extend(np.linspace(self._center_value_params_seq[i],
                                                      self._center_value_params_seq[i+1],
                                                      self._time_value_params_seq[i+1]-self._time_value_params_seq[i]+1))
            self._outer_value_seq.extend(np.linspace(self._outer_value_params_seq[i],
                                                     self._outer_value_params_seq[i+1],
                                                     self._time_value_params_seq[i+1]-self._time_value_params_seq[i]+1))
            if i is not len(self._time_value_params_seq)-1:
                self._pos_seq.pop(-1)
                self._radius_seq.pop(-1)
                self._center_value_seq.pop(-1)
                self._outer_value_seq.pop(-1)

        self._is_value_seq_exist = True
        return

    def update(self):
        if self._is_value_seq_exist:
            retval = self.get_value_sequence(pt_pos=self._pt_pos, cur_t=self._time_stamp, mode=self._pt_mode)
            self._time_stamp = self._time_stamp + self.step
        else:
            retval = self.get_value(pt_pos=self._pt_pos)
        return retval

def EnvBaseTest():
    """
    Function for testing EnvBase
    :return:
    """
    import matplotlib.pyplot as plt

    font = {'family': 'SimHei',
            'weight': 'bold',
            'size': '12'}
    plt.rc('font', **font)  # 设置字体的更多属性
    plt.rc('axes', unicode_minus=False)  # 解决坐标轴负数的负号显示问题
    print("===== EnvBase Test =====")
    TestObj = EnvBase(id="env_01",
                      name="test_01",
                      class_name="Vis",
                      pos=[100, 100, 0],
                      radius=50,
                      center_value=50,
                      outer_value=100)

    print("----- get value test -----")
    print("TestObj.get_value([100, 100, 0]):", TestObj.get_value([100, 100, 0]))
    print("TestObj.get_value([75, 75, 0]):", TestObj.get_value([75, 75, 0]))
    print("TestObj.get_value([50, 50, 0]):", TestObj.get_value([50, 50, 0]))
    print("TestObj.get_value([25, 25, 0]):", TestObj.get_value([25, 25, 0]))

    print("----- set params test -----")
    TestObj.set_params(pos=[75, 75, 0], radius=50, center_value=100, outer_value=50)
    print("TestObj.get_value([125, 125, 0]):", TestObj.get_value([125, 125, 0]))
    print("TestObj.get_value([100, 100, 0]):", TestObj.get_value([100, 100, 0]))
    print("TestObj.get_value([75, 75, 0]):", TestObj.get_value([75, 75, 0]))
    print("TestObj.get_value([50, 50, 0]):", TestObj.get_value([50, 50, 0]))
    print("TestObj.get_value([25, 25, 0]):", TestObj.get_value([25, 25, 0]))

    print("----- get value (mesh grid) test -----")
    from modules.simulation.mesh import MeshScene
    Mesh = MeshScene(xrange=[25, 125], yrange=[25, 125], xcount=100, ycount=100)
    # pts_x, pts_y = np.meshgrid(range(25, 125, 1), range(25, 125, 1), indexing="xy")
    # pt_pos = list(zip(pts_x.flat, pts_y.flat))
    pt_pos = Mesh.get_meshgrid(mode="2D")
    meshval = TestObj.get_value(pt_pos=pt_pos, mode="mesh")
    meshval = np.array(meshval).reshape([100, 100])
    print("TestObj.get_value mesh:", meshval)

    import matplotlib.pyplot as plt

    # fig1 = plt.figure(num=1, figsize=(8, 6))
    # plt.imshow(meshval, interpolation='nearest', cmap=plt.cm.hot)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.colorbar()
    # plt.xticks(np.arange(25, 125, 100))
    # plt.yticks(np.arange(25, 125, 100))
    # plt.title('Mesh value test')
    # # plt.show()

    print("----- get value seq test -----")
    TestObj.set_params_sequence(pos_params_seq=[[30, 30], [30, 40], [50, 50], [60, 60], [70, 70]],
                                radius_params_seq=[20, 25, 30, 35, 40],
                                center_value_params_seq=[10, 20, 30, 40, 50],
                                outer_value_params_seq=[110, 120, 130, 140, 150],
                                time_value_params_seq=[0, 10, 20, 30, 40]
                                )
    TestObj.get_interpolation_params_sequence()

    from matplotlib.animation import FuncAnimation

    fig2 = plt.figure(num=2)
    im = plt.imshow(meshval, interpolation='nearest', cmap=plt.cm.BuGn, vmin=0, vmax=300)
    plt.xlabel('经度方向坐标x')
    plt.ylabel('纬度方向坐标y')
    cb = plt.colorbar()
    plt.xticks(np.arange(25, 125, 100))
    plt.yticks(np.arange(25, 125, 100))
    plt.grid()
    plt.xticks(np.arange(25, 125, 25))  # fixed
    plt.yticks(np.arange(25, 125, 25))  # fixed
    cb.set_label('能见度 单位(m)')
    plt.title('能见度空间分布图')

    t = np.arange(0, 38)

    TestObj.set_meshes(pt_pos=pt_pos)
    TestObj.set_pt_mode(mode="mesh")

    def init():
        # meshval = TestObj.get_value_sequence(pt_pos=pt_pos, cur_t=0, mode="mesh")
        meshval = TestObj.update()
        meshval = np.array(meshval).reshape([100, 100])
        im = plt.imshow(meshval, interpolation='nearest', cmap=plt.cm.BuGn, vmin=0, vmax=300)
        return im


    def update(step):
        # meshval = TestObj.get_value_sequence(pt_pos=pt_pos, cur_t=step, mode="mesh")
        meshval = TestObj.update()
        meshval = np.array(meshval).reshape([100, 100])
        im = plt.imshow(meshval, interpolation='nearest', cmap=plt.cm.BuGn, vmin=0, vmax=300)
        return im
    ani = FuncAnimation(fig2,
                        func=update,
                        init_func=init,
                        frames=t,
                        interval=10,
                        repeat=False)
    # ani.save(r"D:\Project\EmergencyDeductionEngine\docs\figs\envs_base.gif")
    plt.show()
    print("===== Test accomplished! =====")

    pass

if __name__=="__main__":
    EnvBaseTest()