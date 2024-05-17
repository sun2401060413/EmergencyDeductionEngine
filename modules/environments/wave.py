# !/usr/bin/python3
# Coding:   utf-8
# @File:    wave.py
# @Time:    2022/4/11 18:24
# @Author:  Gao Peng
# @Version: 0.0.0
import os

import matplotlib

from modules.simulation.envs import EnvBase
import matplotlib.pyplot as plt
from modules.simulation.mesh import MeshScene
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

class WaveEnv:
    """
    Class for describing all wave conditions, maybe including wave_height and wave_direction
    """
    def __init__(self):
        self._wave_height = EnvBase()
        self._wave_direction = EnvBase()
        self._tide_height = EnvBase()
        self._wave_lifecycle = EnvBase()
        self._wave_speed = EnvBase()


        #simulation flag
        self._enable_wave_height = False
        self._enable_wave_direction = False
        self._enable_tide_height = False
        self._enable_wave_lifecycle = False
        self._enable_wave_speed = False

    def reset_enable_flag(self):
        self._enable_wave_height = False
        self._enable_wave_direction = False
        self._enable_tide_height = False
        self._enable_wave_lifecycle = False
        self._enable_wave_speed = False

    def set_params(self, params_dict={}):
        self.reset_enable_flag()
        if params_dict.__contains__("wave_height"):
            self._enable_wave_height = True
            p_dict = params_dict["wave_height"]
            self._set_params_via_dict(obj=self._wave_height,
                                     param_dict=p_dict)
        if params_dict.__contains__("wave_direction"):
            self._enable_wave_height = True
            p_dict = params_dict["wave_direction"]
            self._set_params_via_dict(obj=self._wave_direction,
                                     param_dict=p_dict)
        if params_dict.__contains__("tide_height"):
            self._enable_wave_height = True
            p_dict = params_dict["tide_height"]
            self._set_params_via_dict(obj=self._wave_direction,
                                     param_dict=p_dict)
        if params_dict.__contains__("wave_lifecycle"):
            self._enable_wave_height = True
            p_dict = params_dict["wave_lifecycle"]
            self._set_params_via_dict(obj=self._wave_direction,
                                     param_dict=p_dict)
        if params_dict.__contains__("wave_speed"):
            self._enable_wave_height = True
            p_dict = params_dict["wave_speed"]
            self._set_params_via_dict(obj=self._wave_direction,
                                      param_dict=p_dict)

    def _set_params_via_dict(self, obj, param_dict):
        if param_dict.__contains__("id"):
            obj.set_id(param_dict["id"])
        if param_dict.__contains__("name"):
            obj.set_name(param_dict["name"])
        obj.set_params(
            pos=param_dict["pos"],
            radius=param_dict["radius"],
            center_value=param_dict["center_value"],
            outer_value=param_dict["outer_value"]
        )
    def get_value(self, pt_pos=None, mode="point"):
        # 2D calculation
        if mode is "point":
            pts = [pt_pos]
        else:
            pts = pt_pos

        retval = {}
        if self._enable_wave_height:
            retval["wave_height"] = self._wave_height.get_value(pt_pos=pt_pos, mode=mode)
        if self._enable_wave_direction:
            retval["wave_direction"] = self._wave_direction.get_value(pt_pos=pt_pos, mode=mode)
        if self._enable_tide_height:
            retval["tide_height"] = self._tide_height.get_value(pt_pos=pt_pos, mode=mode)
        if self._enable_wave_lifecycle:
            retval["wave_lifecycle"] = self._wave_lifecycle.get_value(pt_pos=pt_pos, mode=mode)
        if self._enable_wave_speed:
            retval["wave_speed"] = self._wave_speed.get_value(pt_pos=pt_pos, mode=mode)


        return retval
    def set_params_sequence(self, params_dict={}):
        self.reset_enable_flag()
        if params_dict.__contains__("wave_height"):
            self._enable_wave_height = True
            p_dict = params_dict["wave_height"]
            self._set_params_sequence_via_dict(obj=self._wave_height,
                                      param_dict=p_dict)
        if params_dict.__contains__("wave_direction"):
            self._enable_wave_direction = True
            p_dict = params_dict["wave_direction"]
            self._set_params_sequence_via_dict(obj=self._wave_direction,
                                      param_dict=p_dict)
        if params_dict.__contains__("tide_height"):
            self._enable_tide_height = True
            p_dict = params_dict["tide_height"]
            self._set_params_sequence_via_dict(obj=self._tide_height,
                                      param_dict=p_dict)
        if params_dict.__contains__("wave_lifecycle"):
            self._enable_wave_lifecycle = True
            p_dict = params_dict["wave_lifecycle"]
            self._set_params_sequence_via_dict(obj=self._wave_lifecycle,
                                      param_dict=p_dict)
        if params_dict.__contains__("wave_speed"):
            self._enable_wave_speed = True
            p_dict = params_dict["wave_speed"]
            self._set_params_sequence_via_dict(obj=self._wave_speed,
                                               param_dict=p_dict)


    def _set_params_sequence_via_dict(self, obj, param_dict):
        if param_dict.__contains__("id"):
            obj.set_id(param_dict["id"])
        if param_dict.__contains__("name"):
            obj.set_name(param_dict["name"])
        obj.set_params_sequence(
            pos_params_seq=param_dict["pos"],
            radius_params_seq=param_dict["radius"],
            center_value_params_seq=param_dict["center_value"],
            outer_value_params_seq=param_dict["outer_value"],
            time_value_params_seq=param_dict["time_value"]
        )

    def get_value_sequence(self, pt_pos=None, cur_t=0, mode="point"):#序列数据下加入了时间信息
        # 2D calculation
        if mode is "point":
            pts = [pt_pos]
        else:
            pts = pt_pos

        retval = {}
        if self._enable_wave_height:
            retval["wave_height"] = self._wave_height.get_value_sequence(pt_pos=pt_pos, cur_t=cur_t, mode=mode)
        if self._enable_wave_direction:
            retval["wave_direction"] = self._wave_direction.get_value_sequence(pt_pos=pt_pos, cur_t=cur_t, mode=mode)
        if self._enable_tide_height:
            retval["tide_height"] = self._tide_height.get_value_sequence(pt_pos=pt_pos, cur_t=cur_t, mode=mode)
        if self._enable_wave_lifecycle:
            retval["wave_lifecycle"] = self._wave_lifecycle.get_value_sequence(pt_pos=pt_pos, cur_t=cur_t, mode=mode)
        if self._enable_wave_height:
            retval["wave_speed"] = self._wave_speed.get_value_sequence(pt_pos=pt_pos, cur_t=cur_t, mode=mode)


        return retval

def WaveEnvTest():
    """
    A case for resting WaveEnv class
    :return:
    """


    font = {'family': 'SimHei',
            'weight': 'bold',
            'size': '12'}
    plt.rc('font', **font)  # 设置字体的更多属性
    plt.rc('axes', unicode_minus=False)  # 解决坐标轴负数的负号显示问题

    # params for describing the weather conditions, set outside data
    params_dict = {
        "wave_height" : {
            "id" : "env07",
            "name" : "test_07",
            "pos" :[75, 75, 0],
            "radius" : 50,
            "center_value" : 10,
            "outer_value" : 3,
        },
        "wave_direction": {
            "id": "env08",
            "name": "test_08",
            "pos": [75, 75, 0],
            "radius": 50,
            "center_value": 10,
            "outer_value": 3,
        },
        "tide_height": {
            "id": "env09",
            "name": "test_09",
            "pos": [75, 75, 0],
            "radius": 200,
            "center_value": 50,
            "outer_value": 20,
        },
        "wave_lifecycle": {
            "id": "env10",
            "name": "test_10",
            "pos": [75, 75, 0],
            "radius": 50,
            "center_value": 20,
            "outer_value": 5,
        },
        "wave_speed": {
            "id": "env11",
            "name": "test_11",
            "pos": [75, 75, 0],
            "radius": 50,
            "center_value": 30,
            "outer_value": 10,
        },
    }
    WaveEnvObj = WaveEnv()
    print("===== WaveEnv Test =====")
    print("----- get value test -----")
    WaveEnvObj.set_params(params_dict=params_dict)
    #set area and boundary
    Mesh = MeshScene(xrange=[25, 125], yrange=[25, 125], xcount=100, ycount=100)
    pt_pos = Mesh.get_meshgrid(mode="2D")

    params_dict_seq = {
        "wave_height" : {
            "id": "env_07",
            "name" : "test_07",
            "pos" : [[30, 30], [50, 50], [70, 70], [90, 90], [110, 110]],
            "radius" : [50, 100, 150, 100, 50],
            "center_value" : [3, 5, 10, 5, 3],
            "outer_value" : [1, 3, 6, 3, 1],
            "time_value" : [0, 10, 20, 30, 40],
        },
        "wave_direction":{
            "id": "env_08",
            "name": "test_08",
            "pos": [[30, 30], [50, 50], [70, 70], [90, 90], [110, 110]],
            "radius": [50, 50, 50, 50, 50],
            "center_value": [50, 100, 150, 200, 250],
            "outer_value": [40, 80, 120, 180, 220],
            "time_value": [0, 10, 20, 30, 40],
        },
        "tide_height": {
            "id": "env_09",
            "name": "test_09",
            "pos": [[30, 30], [50, 50], [70, 70], [90, 90], [110, 110]],
            "radius": [100, 100, 100, 100, 100],
            "center_value": [-40, 50, 150, 50, -50],
            "outer_value": [-50, 30, 120, 30, -50],
            "time_value": [0, 10, 20, 30, 40],
        },
        "wave_lifecycle": {
            "id": "env_10",
            "name": "test_10",
            "pos": [[30, 30], [50, 50], [70, 70], [90, 90], [110, 110]],
            "radius": [50, 50, 50, 50, 50],
            "center_value": [5, 10, 20, 10, 5],
            "outer_value": [2, 6, 12, 6, 2],
            "time_value": [0, 10, 20, 30, 40],
        },
        "wave_speed": {
            "id": "env_11",
            "name": "test_11",
            "pos": [[30, 30], [50, 50], [70, 70], [90, 90], [110, 110]],
            "radius": [50, 50, 50, 50, 50],
            "center_value": [15, 30, 50, 30, 15],
            "outer_value": [10, 20, 30, 20, 10],
            "time_value": [0, 10, 20, 30, 40],
        },
    }
    WaveEnvObj.set_params_sequence(params_dict=params_dict_seq)

    fig2 = plt.figure(num = 2, figsize=(15,8))

    def wave_condition_plot(retval,ttext):
        """
        Some settings are workable only in this program. see[# fixed]
        :param retval:
        :return:
        """


        centers = [25, 125, 125, 25]            # fixed
        #centers = [0, 100, 100, 0]            # fixed
        dx, = np.diff(centers[:2]) / (100 - 1)  # fixed
        dy, = -np.diff(centers[2:]) / (100 - 1) # fixed
        extent = [centers[0] - dx / 2, centers[1] + dx / 2, centers[2] + dy / 2, centers[3] - dy / 2]
        t = ttext

        # plot wave_height
        plt.subplot(2, 3, 1)
        meshval_wave_height = np.array(retval["wave_height"]).reshape([100, 100]) # fixed
        im = plt.imshow(meshval_wave_height, interpolation=None, cmap=plt.cm.Blues, extent=extent, aspect='auto', vmin=0, vmax=15)   # fixed
        plt.text(1, 3, 't=%.1f'%t, fontsize=20)
        #x, y, v, u = draw_wave_direction(meshval_wave_direction)
        #plt.quiver(x, y, u, v, color="blue", pivot="tip", units="inches")
        #plt.scatter(x, y, color="b", s=0.05)
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        cb = plt.colorbar()
        plt.grid()
        plt.xticks(np.arange(25, 150, 25))  # fixed
        plt.yticks(np.arange(50, 150, 25))  # fixed
        plt.gca().invert_yaxis()
        cb.set_label('wave height(m)')
        plt.title('Spatial distribution of wave height', fontsize='small',fontweight='medium')

        # plot wave_direction

        plt.subplot(2, 3, 2)
        meshval_wave_direction = np.array(retval["wave_direction"]).reshape([100, 100]) # fixed
        #plt.imshow(meshval_wave_height, interpolation=None, cmap=plt.cm.BuGn, extent=extent, aspect='auto', vmin=0, vmax=15)   # fixed
        plt.cla()
        x, y, v, u = draw_wave_direction(meshval_wave_direction)
        plt.quiver(x, y, u, v, color="red", pivot="tip", units="inches")
        plt.scatter(x, y, color="r", s=0.05)
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.title('Spatial distribution of wave direction', fontsize='small',fontweight='medium')

        #plot tide_height
        plt.subplot(2, 3, 3)    # fixed
        meshval_tide_height = np.array(retval["tide_height"]).reshape([100, 100]) # fixed
        plt.imshow(meshval_tide_height, interpolation=None, cmap=plt.cm.BuPu, extent=extent, aspect='auto', vmin=-50, vmax=250)   # fixed
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        cb = plt.colorbar()
        plt.grid()
        plt.xticks(np.arange(25, 150, 25))  # fixed
        # reverse y axis
        #lt.yticks(np.arange(25, 125, 25))  # fixed
        plt.yticks(np.arange(50, 150, 25))  # fixed
        plt.gca().invert_yaxis()
        cb.set_label('tides height(cm)')
        plt.title('Spatial distribution of tides height', fontsize='small',fontweight='medium')

        #plot wave_lifecycle
        plt.subplot(2, 3, 4)    # fixed
        meshval_wave_lifecycle = np.array(retval["wave_lifecycle"]).reshape([100, 100]) # fixed
        plt.imshow(meshval_wave_lifecycle, interpolation=None, cmap=plt.cm.BuGn, extent=extent, aspect='auto', vmin=0, vmax=25)   # fixed
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        cb = plt.colorbar()
        plt.grid()
        plt.xticks(np.arange(25, 150, 25))  # fixed
        plt.yticks(np.arange(50, 150, 25))  # fixed
        plt.gca().invert_yaxis()
        cb.set_label('wave cycles(s)')
        plt.title('Spatial distribution of wave cycles', fontsize='small',fontweight='medium')

        #plot wave_speed
        plt.subplot(2, 3, 5)    # fixed
        meshval_wave_speed = np.array(retval["wave_speed"]).reshape([100, 100]) # fixed
        plt.imshow(meshval_wave_speed, interpolation=None, cmap=plt.cm.Greens, extent=extent, aspect='auto', vmin=10, vmax=50)   # fixed
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        cb = plt.colorbar()
        plt.grid()
        plt.xticks(np.arange(25, 150, 25))  # fixed
        plt.yticks(np.arange(50, 150, 25))  # fixed
        plt.gca().invert_yaxis()
        cb.set_label('wave speed(km/h)')
        plt.title('Spatial distribution of wave speed', fontsize='small',fontweight='medium')
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        return im

    def draw_wave_direction(retval):
        a = np.arange(25, 125, 10)  # fixed
        b = np.arange(25, 125, 10)  # fixed
        x, y = np.meshgrid(a, b)
        z = retval[x - 25, y - 25]  # fixed
        v, u = np.cos(z * np.pi / 180), np.sin(z * np.pi / 180)
        return x, y, v, u

    t = np.arange(0, 40)


    def init():
        retval = WaveEnvObj.get_value_sequence(pt_pos=pt_pos, cur_t=0, mode="mesh")
        ttext = 0
        return wave_condition_plot(retval,ttext)

    def update(step):
        retval = WaveEnvObj.get_value_sequence(pt_pos=pt_pos, cur_t=step, mode="mesh")
        ttext = step
        return wave_condition_plot(retval,ttext)

    ani = FuncAnimation(fig2,
                        func=update,
                        init_func = init,
                        frames = t,
                        interval=10,
                        repeat=False
                        ) #绘制动图
    #ani.save(r"E:\fproject\docs\figs\wave_test.gif", writer='pillow') #保存为gif文件
    #ffmpegpath = os.path.abspath(r"D:\google_download\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe")
    #matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath
    #writer = animation.FFMpegWriter()
    # ani.save(r"E:\fproject\docs\figs\wave_test.mp4", writer=writer) #保存为mp4文件
    ani.save(r"F:\ffproject\EmergencyDeduce\v0.6\EmergencyDeductionEngine\docs\figs\wave_evolution.gif")


    # with open (r"F:\ffproject\EmergencyDeducationEngine\docs\figs\wave_test.html", "w") as f:
    #     print(ani.to_jshtml(), file = f)      #保存为html文件，可随时间回溯

    #plt.show()
    print("===== Test accomplished! =====")

if __name__=="__main__":
    WaveEnvTest()