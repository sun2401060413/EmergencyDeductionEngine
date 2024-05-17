
import numpy as np
from modules.simulation.simulator import Element


class MeshScene(Element):
    """
    Base class of all mesh
    """
    def __init__(self,
                 id=None,
                 name=None,
                 class_name=None,
                 xrange=None,
                 yrange=None,
                 zrange=None,
                 xcount=10,
                 ycount=10,
                 zcount=1,
                 mask=None):
        """
        :param xrange: start and end position on x-axis
        :param yrange: strat and end position on y-axis
        :param zrange: start and end position on z-axis
        :param xcount: grid count on x-axis
        :param ycount: grid count on y-axis
        :param zcount: grid count on z-axis
        """
        super().__init__(id, name, class_name)

        self.xrange = xrange
        self.yrange = yrange
        self.zrange = zrange
        self.xcount = xcount
        self.ycount = ycount
        self.zcount = zcount

        self.pts_x = None
        self.pts_y = None
        self.pts_z = None

        self.ct_x = 0
        self.ct_y = 0
        self.ct_z = 0

        self.mesh = np.zeros([self.xcount, self.ycount, self.zcount])       # save the value
        self.meshgrid = None                                                # save the coordinate

        self.localpos = None  # The origin of relative coordinate in real mesh coordinate
        self.localmesh = None  # relative coordinate of center

        self.mask = mask    # As same size as the MeshScene

    def set_params(self, xrange=None, yrange=None, zrange=None, xcount=None, ycount=None, zcount=None):
        self.xrange = xrange
        self.yrange = yrange
        self.zrange = zrange
        self.xcount = xcount
        self.ycount = ycount
        self.zcount = zcount

    def get_meshgrid(self, mode="2D"):
        """
        :param mode: option: "2D" or "3D" ...
            - "2D": get 2D meshgrid
            - "3D": get 3D meshgrid
        :return:
        """
        xinterval = int(abs(self.xrange[1] - self.xrange[0]) / self.xcount)
        xinterval = xinterval if xinterval >= 1 else 1
        yinterval = int(abs(self.yrange[1] - self.yrange[0]) / self.ycount)
        yinterval = yinterval if yinterval >= 1 else 1
        if mode is "2D":
            self.pts_x, self.pts_y = np.meshgrid(range(self.xrange[0], self.xrange[1], xinterval),
                                       range(self.yrange[0], self.yrange[1], yinterval),
                                       indexing="xy")
            self.meshgrid = list(zip(self.pts_x.flat, self.pts_y.flat))
        else:
            zinterval = int(abs(self.zrange[1] - self.zrange[0]) / self.zcount)
            zinterval = zinterval if zinterval >= 1 else 1
            self.pts_x, self.pts_y, self.pts_z = np.meshgrid(range(self.xrange[0], self.xrange[1], xinterval),
                                              range(self.yrange[0], self.yrange[1], yinterval),
                                              range(self.zrange[0], self.zrange[1], zinterval),
                                              indexing="xy")
            self.meshgrid = list(zip(self.pts_x.flat, self.pts_y.flat, self.pts_z.flat))
        return self.meshgrid

    def get_mesh(self, mode="2D"):
        if mode is "2D":
            return self.mesh[:, :, 0]
        else:
            return self.mesh

    @staticmethod
    def get_delta_xyz(ref_pos, target_pos):
        """
        :param ref_pos: position of base point
        :param target_pos: position of the point to be calculated
        :return: Relative distance
        """
        return (np.array(target_pos) - np.array(ref_pos)).tolist()
        pass

    def set_mask(self, mask=None):
        self.mask = mask

    def get_mask(self):
        return self.mask


class LocalMeshScene(MeshScene):
    """Local mesh"""
    def __init__(self, length=1, width=1, height=1, l_stride=1, w_stride=1, h_stride=1, l_start=None, w_start=None, h_start=None):
        """
        :param length:
        :param width:
        :param height:
        :param l_stride:
        :param w_stride:
        :param h_stride:
        :param l_start: the start number of mesh on the length axis
        :param w_start: the start number of mesh on the width axis
        :param h_start: the start number of mesh on the height axis
        """
        self.length = length
        self.width = width
        self.height = height

        self.l_stride = l_stride
        self.w_stride = w_stride
        self.h_stride = h_stride

        self.l_start = l_start if l_start is not None else -1 * int(self.length / 2)
        self.w_start = w_start if w_start is not None else -1 * int(self.width / 2)
        self.h_start = h_start if h_start is not None else -1 * int(self.height / 2)

        super().__init__(
            id=None,
            name=None,
            class_name=None,
            xrange=[self.w_start, self.w_start + self.width],
            yrange=[self.l_start, self.l_start + self.length],
            zrange=[self.h_start, self.h_start + self.height],
            xcount=int(self.width / self.w_stride),
            ycount=int(self.length / self.l_stride),
            zcount=int(self.height / self.h_stride),
        )

        self.get_meshgrid(mode="2D")
        self.get_origin_index(mode="2D")

    def reset_origin(self, mode="2D", l_start=None, w_start=None, h_start=None):
        """Reset the origin of the matrix, the default origin is the center of """
        self.l_start = l_start if l_start is not None else -1 * int(self.length / 2)
        self.w_start = w_start if w_start is not None else -1 * int(self.width / 2)
        self.h_start = h_start if h_start is not None else -1 * int(self.height / 2)

        self.xrange = [self.w_start, self.w_start + self.width]
        self.yrange = [self.l_start, self.l_start + self.length]
        self.zrange = [self.h_start, self.h_start + self.height]

        self.get_meshgrid(mode=mode)
        return self.get_origin_index(mode=mode)

    def get_origin_index(self, mode="2D"):
        """Get the index of the center in matrix"""
        if self.mesh is None:
            self.get_meshgrid(mode="2D")

        min_pts_x, min_pts_y = np.abs(self.pts_x).min(), np.abs(self.pts_y).min()
        self.ct_x = np.unique(np.where(self.pts_x == min_pts_x)[1])[0]
        self.ct_y = np.unique(np.where(self.pts_y == min_pts_y)[0])[0]
        if mode is "2D":
            return self.ct_x, self.ct_y
        else:
            self.ct_z = np.unique(np.where(self.pts_z == 0)[2])[0]
            return self.ct_x, self.ct_y, self.ct_z

def MeshSceneTest():
    """
    Function for testing MeshScene
    :return:
    """
    print("===== MeshScene Test =====")
    MeshSceneObj = MeshScene(xrange=[0, 10],
                             yrange=[0, 20],
                             zrange=[0, 5],
                             xcount=10,
                             ycount=20,
                             zcount=5)
    print("----- Get meshgrid test -----")                  # PASS
    print(MeshSceneObj.get_meshgrid(mode="2D"))
    pass
    print("===== LocalMesh Test =====")
    print("----- Get meshgrid test -----")  # PASS
    LocalMeshSceneObj = LocalMeshScene(
        length=10,
        width=10,
        height=10,
        l_stride=1,
        w_stride=1,
        h_stride=1
    )
    print(LocalMeshSceneObj.get_meshgrid(mode="2D"))
    print("size:", len(LocalMeshSceneObj.get_meshgrid(mode="2D")))

    print("----- Get meshgrid with custom settings -----")  # PASS
    LocalMeshSceneObj = LocalMeshScene(
        length=10,
        width=10,
        height=10,
        l_stride=1,
        w_stride=1,
        h_stride=1,
        l_start=-1,
        w_start=-1,
        h_start=-1
    )

    print(LocalMeshSceneObj.get_meshgrid(mode="2D"))
    print("size:", len(LocalMeshSceneObj.get_meshgrid(mode="2D")))

    print("----- Reset origin test -----")  # PASS
    LocalMeshSceneObj.reset_origin(mode="2D", l_start=-3, w_start=-3, h_start=-3)
    print(LocalMeshSceneObj.get_meshgrid(mode="2D"))
    print("size:", len(LocalMeshSceneObj.get_meshgrid(mode="2D")))


    print("----- Get_delta_xyz test -----")                 # PASS
    a = [1, 1, 1]
    b = [1, 2, 3]
    c = [-1, -2, -3]
    print(MeshScene.get_delta_xyz(a, b))
    print(MeshScene.get_delta_xyz(a, c))

    print("----- Get mesh value test -----")                # PASS
    # print(LocalMeshSceneObj.get_mesh(mode="2D"))
    # print(LocalMeshSceneObj.get_mesh(mode="3D"))

    print("----- Get the index of origin test -----")       # PASS
    print(LocalMeshSceneObj.get_origin_index())             # 2D test
    LocalMeshSceneObj.get_meshgrid(mode="3D")               # 3D test
    print(LocalMeshSceneObj.get_origin_index(mode="3D"))

    print("----- Mesh visualization -----")
    LocalMeshSceneObj.reset_origin(mode="2D", l_start=-6, w_start=-1, h_start=-1)
    LocalMeshSceneObj.get_meshgrid(mode="2D")               # 3D test
    x, y = LocalMeshSceneObj.pts_x, LocalMeshSceneObj.pts_y
    sigma = 2
    thresh = 5
    z = np.round(np.array(1 / (2 * np.pi * (sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)))*1000, 3)
    LocalMeshSceneObj.mesh = 1 - (z > thresh)*1

    print(LocalMeshSceneObj.mesh.size)
    import matplotlib.pyplot as plt
    font = {'family': 'SimHei',
            'weight': 'bold',
            'size': '12'}
    plt.rc('font', **font)  # 设置字体的更多属性
    plt.rc('axes', unicode_minus=False)  # 解决坐标轴负数的负号显示问题
    plt.figure(num=1)
    # plt.imshow(z, interpolation=None, cmap=plt.cm.GnBu, aspect='auto', vmin=5, vmax=25)
    plt.imshow(LocalMeshSceneObj.mesh, interpolation=None, cmap=plt.cm.GnBu, aspect='auto', vmin=0, vmax=5)
    plt.xlabel('经度方向坐标x')
    plt.ylabel('纬度方向坐标y')
    cb = plt.colorbar()
    plt.xticks(np.arange(0, 10, 1))  # fixed
    plt.yticks(np.arange(0, 10, 1))  # fixed
    cb.set_label('强度')
    plt.title('变量空间分布图')
    plt.show()



if __name__ == "__main__":
    MeshSceneTest()
