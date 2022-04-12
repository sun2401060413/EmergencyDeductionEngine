# !/usr/bin/python3
# Coding:   utf-8
# @File:    mesh.py
# @Time:    2022/3/26 18:31
# @Author:  Sun Zhu
# @Version: 0.0.0

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
                 zcount=10):
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

        self.mesh = None

        self.localpos = None  # The origin of relative coordinate in real mesh coordinate
        self.localmesh = None  # relative coordinate of center

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
            pts_x, pts_y = np.meshgrid(range(self.xrange[0], self.xrange[1], xinterval),
                                       range(self.yrange[0], self.yrange[1], yinterval),
                                       indexing="xy")
            self.mesh = list(zip(pts_x.flat, pts_y.flat))
        else:
            zinterval = int(abs(self.zrange[1] - self.zrange[0]) / self.zcount)
            zinterval = zinterval if zinterval >= 1 else 1
            pts_x, pts_y, pts_z = np.meshgrid(range(self.xrange[0], self.xrange[1], xinterval),
                                              range(self.yrange[0], self.yrange[1], yinterval),
                                              range(self.zrange[0], self.zrange[1], zinterval),
                                              indexing="xy")
            self.mesh = list(zip(pts_x.flat, pts_y.flat, pts_z.flat))
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


class LocalMeshScene(MeshScene):
    """Local mesh"""

    def __init__(self, length=1, width=1, height=1, l_stride=1, w_stride=1, h_stride=1):
        self.length = length
        self.width = width
        self.height = height

        self.l_stride = l_stride
        self.w_stride = w_stride
        self.h_stride = h_stride
        super().__init__(
            id=None,
            name=None,
            class_name=None,
            xrange=[-1 * int(self.width / 2), int(self.width / 2)],
            yrange=[-1 * int(self.length / 2), int(self.length / 2)],
            zrange=[-1 * int(self.height / 2), int(self.height / 2)],
            xcount=int(self.width / self.w_stride),
            ycount=int(self.length / self.l_stride),
            zcount=int(self.height / self.h_stride),
        )
        self.mask = None  # As same size as the AreaScene


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
    print("----- get meshgrid test -----")
    print(MeshSceneObj.get_meshgrid(mode="2D"))  # PASS
    pass
    print("===== LocalMesh Test =====")
    LocalMeshSceneObj = LocalMeshScene(
        length=10,
        width=10,
        height=10,
        l_stride=1,
        w_stride=1,
        h_stride=1
    )
    print("----- get meshgrid test -----")
    print(LocalMeshSceneObj.get_meshgrid(mode="2D"))  # PASS
    print("size:", len(LocalMeshSceneObj.get_meshgrid(mode="2D")))

    print("----- get_delta_xyz test -----")
    A = [1, 1, 1]
    B = [1, 2, 3]
    C = [-1, -2, -3]
    print(MeshScene.get_delta_xyz(A, B))
    print(MeshScene.get_delta_xyz(A, C))


if __name__ == "__main__":
    MeshSceneTest()
