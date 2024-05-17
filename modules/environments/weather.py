
from modules.simulation.envs import EnvBase

class WeatherEnv:
    """
    Class for describing all weather conditions
    """
    def __init__(self):
        self._visibility = EnvBase()
        self._brightness = EnvBase()
        self._temperature = EnvBase()
        self._wind_speed = EnvBase()
        self._wind_direction = EnvBase()        # Wind direction 插值的问题需要单独处理
        self._rainfall = EnvBase()

        # simulatable flag
        self._enable_visibility = False
        self._enable_brightness = False
        self._enable_temperature = False
        self._enable_wind_speed = False
        self._enable_wind_direction = False
        self._enable_rainfall = False

    def reset_enable_flag(self):
        self._enable_visibility = False
        self._enable_brightness = False
        self._enable_temperature = False
        self._enable_wind_speed = False
        self._enable_wind_direction = False
        self._enable_rainfall = False

    def set_params(self, params_dict={}):
        self.reset_enable_flag()
        if params_dict.__contains__("visibility"):
            self._enable_visibility = True
            p_dict = params_dict["visibility"]
            self._set_params_via_dict(obj=self._visibility,
                                     param_dict=p_dict)
        if params_dict.__contains__("brightness"):
            self._enable_brightness = True
            p_dict = params_dict["brightness"]
            self._set_params_via_dict(obj=self._brightness,
                                     param_dict=p_dict)
        if params_dict.__contains__("temperature"):
            self._enable_temperature = True
            p_dict = params_dict["temperature"]
            self._set_params_via_dict(obj=self._temperature,
                                     param_dict=p_dict)
        if params_dict.__contains__("wind_speed"):
            self._enable_wind_speed = True
            p_dict = params_dict["wind_speed"]
            self._set_params_via_dict(obj=self._wind_speed,
                                     param_dict=p_dict)
        if params_dict.__contains__("wind_direction"):
            self._enable_wind_direction = True
            p_dict = params_dict["wind_direction"]
            self._set_params_via_dict(obj=self._wind_direction,
                                     param_dict=p_dict)
        if params_dict.__contains__("rainfall"):
            self._enable_rainfall = True
            p_dict = params_dict["rainfall"]
            self._set_params_via_dict(obj=self._rainfall,
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
        if self._enable_visibility:
            retval["visibility"] = self._visibility.get_value(pt_pos=pt_pos, mode=mode)
        if self._enable_brightness:
            retval["brightness"] = self._brightness.get_value(pt_pos=pt_pos, mode=mode)
        if self._enable_temperature:
            retval["temperature"] = self._temperature.get_value(pt_pos=pt_pos, mode=mode)
        if self._enable_wind_speed:
            retval["wind_speed"] = self._wind_speed.get_value(pt_pos=pt_pos, mode=mode)
        if self._enable_wind_direction:
            retval["wind_direction"] = self._wind_direction.get_value(pt_pos=pt_pos, mode=mode)
        if self._enable_rainfall:
            retval["rainfall"] = self._rainfall.get_value(pt_pos=pt_pos, mode=mode)

        return retval

    def set_params_sequence(self, params_dict={}):
        self.reset_enable_flag()
        if params_dict.__contains__("visibility"):
            self._enable_visibility = True
            p_dict = params_dict["visibility"]
            self._set_params_sequence_via_dict(obj=self._visibility,
                                      param_dict=p_dict)
        if params_dict.__contains__("brightness"):
            self._enable_brightness = True
            p_dict = params_dict["brightness"]
            self._set_params_sequence_via_dict(obj=self._brightness,
                                      param_dict=p_dict)
        if params_dict.__contains__("temperature"):
            self._enable_temperature = True
            p_dict = params_dict["temperature"]
            self._set_params_sequence_via_dict(obj=self._temperature,
                                      param_dict=p_dict)
        if params_dict.__contains__("wind_speed"):
            self._enable_wind_speed = True
            p_dict = params_dict["wind_speed"]
            self._set_params_sequence_via_dict(obj=self._wind_speed,
                                      param_dict=p_dict)
        if params_dict.__contains__("wind_direction"):
            self._enable_wind_direction = True
            p_dict = params_dict["wind_direction"]
            self._set_params_sequence_via_dict(obj=self._wind_direction,
                                      param_dict=p_dict)
        if params_dict.__contains__("rainfall"):
            self._enable_rainfall = True
            p_dict = params_dict["rainfall"]
            self._set_params_sequence_via_dict(obj=self._rainfall,
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

    def get_value_sequence(self, pt_pos=None, cur_t=0, mode="point"):
        # 2D calculation
        if mode is "point":
            pts = [pt_pos]
        else:
            pts = pt_pos

        retval = {}
        if self._enable_visibility:
            retval["visibility"] = self._visibility.get_value_sequence(pt_pos=pt_pos, cur_t=cur_t, mode=mode)
        if self._enable_brightness:
            retval["brightness"] = self._brightness.get_value_sequence(pt_pos=pt_pos, cur_t=cur_t, mode=mode)
        if self._enable_temperature:
            retval["temperature"] = self._temperature.get_value_sequence(pt_pos=pt_pos, cur_t=cur_t, mode=mode)
        if self._enable_wind_speed:
            retval["wind_speed"] = self._wind_speed.get_value_sequence(pt_pos=pt_pos, cur_t=cur_t, mode=mode)
        if self._enable_wind_direction:
            retval["wind_direction"] = self._wind_direction.get_value_sequence(pt_pos=pt_pos, cur_t=cur_t, mode=mode)
        if self._enable_rainfall:
            retval["rainfall"] = self._rainfall.get_value_sequence(pt_pos=pt_pos, cur_t=cur_t, mode=mode)

        return retval


def WeatherEnvTest():
    """
    A case for testing WeatherEnv class
    :return:
    """
    import matplotlib.pyplot as plt

    font = {'family': 'SimHei',
            'weight': 'bold',
            'size': '12'}
    plt.rc('font', **font)  # 设置字体的更多属性
    plt.rc('axes', unicode_minus=False)  # 解决坐标轴负数的负号显示问题

    # params for describing the weather conditions
    params_dict = {
        "visibility" : {
            "id" : "env_01",
            "name" : "test_01",
            "pos" : [75, 75, 0],
            "radius" : 50,
            "center_value" : 50,
            "outer_value" : 100
        },
        "brightness" : {
            "id": "env_02",
            "name": "test_02",
            "pos": [75, 75, 0],
            "radius": 1000,
            "center_value": 1200,
            "outer_value": 1100
        },
        "temperature" : {
            "id": "env_03",
            "name": "test_03",
            "pos": [75, 75, 0],
            "radius": 200,
            "center_value": 18,
            "outer_value": 17
        },
        "wind_speed" : {
            "id": "env_03",
            "name": "test_03",
            "pos": [75, 75, 0],
            "radius": 200,
            "center_value": 2.5,
            "outer_value": 2
        },
        "wind_direction" : {
            "id": "env_03",
            "name": "test_03",
            "pos": [75, 75, 0],
            "radius": 200,
            "center_value": 45,
            "outer_value": 60
        },
        "rainfall" : {
            "id": "env_03",
            "name": "test_03",
            "pos": [75, 75, 0],
            "radius": 200,
            "center_value": 20,
            "outer_value": 10
        },
    }

    WeatherEnvObj = WeatherEnv()
    print("===== WeatherEnv Test =====")
    print("----- get value test -----")
    WeatherEnvObj.set_params(params_dict=params_dict)
    from modules.simulation.mesh import MeshScene
    Mesh = MeshScene(xrange=[25, 125], yrange=[25, 125], xcount=100, ycount=100)
    pt_pos = Mesh.get_meshgrid(mode="2D")
    # retval = WeatherEnvObj.get_value(pt_pos=pt_pos, mode="mesh")
    # print("WeatherEnvObj.get_value mesh:", retval)

    import matplotlib.pyplot as plt
    import numpy as np

    # fig1 = plt.figure(num=1)
    # # plot visibilty
    # plt.subplot(2, 3, 1)
    # meshval_visibility = np.array(retval["visibility"]).reshape([100, 100])
    # plt.imshow(meshval_visibility, interpolation="nearest", cmap=plt.cm.hot)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.colorbar()
    # plt.xticks(np.arange(25, 125, 100))
    # plt.yticks(np.arange(25, 125, 100))
    # plt.title('Mesh value of visibility')
    #
    # # plot brightness
    # plt.subplot(2, 3, 2)
    # meshval_brightness = np.array(retval["brightness"]).reshape([100, 100])
    # plt.imshow(meshval_brightness, interpolation="nearest", cmap=plt.cm.hot)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.colorbar()
    # plt.xticks(np.arange(25, 125, 100))
    # plt.yticks(np.arange(25, 125, 100))
    # plt.title('Mesh value of brightness')
    #
    # # plot temperature
    # plt.subplot(2, 3, 3)
    # meshval_temperature = np.array(retval["temperature"]).reshape([100, 100])
    # plt.imshow(meshval_temperature, interpolation="nearest", cmap=plt.cm.hot)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.colorbar()
    # plt.xticks(np.arange(25, 125, 100))
    # plt.yticks(np.arange(25, 125, 100))
    # plt.title('Mesh value of temperature')
    #
    # # plot wind speed
    # plt.subplot(2, 3, 4)
    # meshval_wind_speed = np.array(retval["wind_speed"]).reshape([100, 100])
    # plt.imshow(meshval_wind_speed, interpolation="nearest", cmap=plt.cm.hot)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.colorbar()
    # plt.xticks(np.arange(25, 125, 100))
    # plt.yticks(np.arange(25, 125, 100))
    # plt.title('Mesh value of wind_speed')
    #
    # # plot wind direction
    # plt.subplot(2, 3, 5)
    # meshval_wind_direction = np.array(retval["wind_direction"]).reshape([100, 100])
    # plt.imshow(meshval_wind_direction, interpolation="nearest", cmap=plt.cm.hot)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.colorbar()
    # plt.xticks(np.arange(25, 125, 100))
    # plt.yticks(np.arange(25, 125, 100))
    # plt.title('Mesh value of wind_direction')
    #
    # # plot rainfall
    # plt.subplot(2, 3, 6)
    # meshval_rainfall = np.array(retval["rainfall"]).reshape([100, 100])
    # plt.imshow(meshval_rainfall, interpolation="nearest", cmap=plt.cm.hot)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.colorbar()
    # plt.xticks(np.arange(25, 125, 100))
    # plt.yticks(np.arange(25, 125, 100))
    # plt.title('Mesh value of rainfall')
    #
    # plt.show()

    params_dict_seq = {
        "visibility": {
            "id": "env_01",
            "name": "test_01",
            "pos": [[30, 30], [60, 40], [60, 80], [80, 100], [100, 100]],
            "radius": [40, 35, 30, 35, 40],
            "center_value": [200, 100, 50, 100, 200],
            "outer_value": [300, 250, 200, 250, 300],
            "time_value" : [0, 10, 20, 30, 40]
        },
        "brightness": {
            "id": "env_02",
            "name": "test_02",
            "pos": [[30, 30], [50, 50], [70, 70], [90, 90], [110, 110]],
            "radius": [100, 120, 150, 120, 100],
            "center_value": [1000, 1050, 1100, 1150, 1200],
            "outer_value": [900, 950, 1000, 1050, 1100],
            "time_value" : [0, 10, 20, 30, 40],
        },
        "temperature": {
            "id": "env_03",
            "name": "test_03",
            "pos":  [[30, 30], [50, 50], [70, 70], [90, 90], [110, 110]],
            "radius":  [100, 120, 150, 120, 100],
            "center_value": [18, 20, 22, 24, 25],
            "outer_value": [16, 18, 20, 22, 23],
            "time_value" : [0, 10, 20, 30, 40],
        },
        "wind_speed": {
            "id": "env_03",
            "name": "test_03",
            "pos": [[30, 30], [50, 50], [70, 70], [90, 90], [110, 110]],
            "radius": [100, 100, 100, 100, 100],
            "center_value": [0.5, 1.0, 1.5, 2.0, 2.5],
            "outer_value": [0.25, 0.5, 1.0, 1.5, 2.0],
            "time_value" : [0, 10, 20, 30, 40],
        },
        "wind_direction": {
            "id": "env_03",
            "name": "test_03",
            "pos": [[30, 30], [50, 50], [70, 70], [90, 90], [110, 110]],
            "radius": [100, 100, 100, 100, 100],
            "center_value": [50, 100, 150, 200, 250],
            "outer_value": [40, 80, 120, 180, 220],
            "time_value" : [0, 10, 20, 30, 40],
        },
        "rainfall": {
            "id": "env_03",
            "name": "test_03",
            "pos": [[30, 30], [30, 40], [50, 50], [60, 60], [70, 70]],
            "radius": [120, 140, 160, 180, 200],
            "center_value": [5, 10, 15, 20, 25],
            "outer_value": [3, 6, 9, 12, 15],
            "time_value" : [0, 10, 20, 30, 40],
        },
    }
    WeatherEnvObj.set_params_sequence(params_dict=params_dict_seq)
    # retval = WeatherEnvObj.get_value_sequence(pt_pos=pt_pos, cur_t=0, mode="mesh")
    # print("WeatherEnvObj.get_value mesh:", retval)
    from matplotlib.animation import FuncAnimation
    import math


    #fig2 = plt.figure(num=2, figsize=(128, 108))
    fig2 = plt.figure(num=2, figsize=(15, 8))

    def weather_condition_plot(retval):
        """
        Some settings are workable only in this program. see[# fixed]
        :param retval:
        :return:
        """
        # plot visibility

        centers = [25, 125, 125, 25]            # fixed
        dx, = np.diff(centers[:2]) / (100 - 1)  # fixed
        dy, = -np.diff(centers[2:]) / (100 - 1) # fixed
        extent = [centers[0] - dx / 2, centers[1] + dx / 2, centers[2] + dy / 2, centers[3] - dy / 2]

        plt.subplot(2, 3, 1)    # fixed
        meshval_visibility = np.array(retval["visibility"]).reshape([100, 100]) # fixed
        im = plt.imshow(meshval_visibility, interpolation=None, cmap=plt.cm.BuGn, extent=extent, aspect='auto', vmin=0, vmax=500)   # fixed
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        cb = plt.colorbar()
        plt.grid()
        #plt.xticks(np.arange(25, 125, 25))  # fixed
        #plt.yticks(np.arange(25, 125, 25))  # fixed
        plt.xticks(np.arange(25, 150, 25))  # fixed
        plt.yticks(np.arange(50, 150, 25))  # fixed
        plt.gca().invert_yaxis()
        cb.set_label('visibility(m)')
        plt.title('Spatial distribution of visibility', fontsize='small',fontweight='medium')

        # plot brightness
        plt.subplot(2, 3, 2)    # fixed
        meshval_brightness = np.array(retval["brightness"]).reshape([100, 100]) # fixed
        plt.imshow(meshval_brightness, interpolation=None, cmap=plt.cm.hot, extent=extent, aspect='auto', vmin=900, vmax=1200) # fixed
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        cb = plt.colorbar()
        plt.grid()
        # plt.xticks(np.arange(25, 125, 25))  # fixed
        # plt.yticks(np.arange(25, 125, 25))  # fixed
        plt.xticks(np.arange(25, 150, 25))  # fixed
        plt.yticks(np.arange(50, 150, 25))  # fixed
        plt.gca().invert_yaxis()
        cb.set_label('brightness(lux)')
        plt.title('Spatial distribution of brightness', fontsize='small',fontweight='medium')

        # plot temperature
        plt.subplot(2, 3, 3)    # fixed
        meshval_temperature = np.array(retval["temperature"]).reshape([100, 100])   # fixed
        plt.imshow(meshval_temperature, interpolation=None, cmap=plt.cm.hot, extent=extent, aspect='auto', vmin=15, vmax=25)    # fixed
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        cb = plt.colorbar()
        plt.grid()
        # plt.xticks(np.arange(25, 125, 25))  # fixed
        # plt.yticks(np.arange(25, 125, 25))  # fixed
        plt.xticks(np.arange(25, 150, 25))  # fixed
        plt.yticks(np.arange(50, 150, 25))  # fixed
        plt.gca().invert_yaxis()
        cb.set_label('temperature(℃)')
        plt.title('Spatial distribution of temperature', fontsize='small',fontweight='medium')

        # plot wind speed
        plt.subplot(2, 3, 4)    # fixed
        meshval_wind_speed = np.array(retval["wind_speed"]).reshape([100, 100]) # fixed
        plt.imshow(meshval_wind_speed, interpolation=None, cmap=plt.cm.Reds, extent=extent, aspect='auto', vmin=0, vmax=5)   # fixed
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        cb = plt.colorbar()
        plt.grid()
        # plt.xticks(np.arange(25, 125, 25))  # fixed
        # plt.yticks(np.arange(25, 125, 25))  # fixed
        plt.xticks(np.arange(25, 150, 25))  # fixed
        plt.yticks(np.arange(50, 150, 25))  # fixed
        plt.gca().invert_yaxis()
        cb.set_label('wind speed(m/s)')
        plt.title('Spatial distribution of wind speed', fontsize='small',fontweight='medium')

        # plot wind direction
        plt.subplot(2, 3, 5)    # fixed
        meshval_wind_direction = np.array(retval["wind_direction"]).reshape([100, 100]) # fixed
        # plt.imshow(meshval_wind_direction, interpolation=None, cmap=plt.cm.hot, extent=extent, aspect='auto', vmin=50, vmax=70)
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.colorbar()
        # plt.grid()
        # plt.xticks(np.arange(25, 125, 25))
        # plt.yticks(np.arange(25, 125, 25))
        plt.cla()
        x, y, v, u = draw_wind_direction(meshval_wind_direction)
        plt.quiver(x, y, u, v, color="blue", pivot="tip", units="inches")
        plt.scatter(x, y, color="b", s=0.05)
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.title('Spatial distribution of wind direction', fontsize='small',fontweight='medium')

        # plot rainfall
        plt.subplot(2, 3, 6)    # fixed
        meshval_rainfall = np.array(retval["rainfall"]).reshape([100, 100]) # fixed
        im = plt.imshow(meshval_rainfall, interpolation=None, cmap=plt.cm.GnBu, extent=extent, aspect='auto', vmin=5, vmax=25)   # fixed
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        cb = plt.colorbar()
        plt.grid()
        # plt.xticks(np.arange(25, 125, 25))  # fixed
        # plt.yticks(np.arange(25, 125, 25))  # fixed
        plt.xticks(np.arange(25, 150, 25))  # fixed
        plt.yticks(np.arange(50, 150, 25))  # fixed
        plt.gca().invert_yaxis()
        cb.set_label('rainfall(mm/24h)')
        plt.title('Spatial distribution of rainfall', fontsize='small',fontweight='medium')
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        return im

    def draw_wind_direction(retval):
        a = np.arange(25, 125, 10)  # fixed
        b = np.arange(25, 125, 10)  # fixed
        x, y = np.meshgrid(a, b)
        z = retval[x-25, y-25]  # fixed
        v, u = np.cos(z*np.pi/180), np.sin(z*np.pi/180)
        return x, y, v, u

    t = np.arange(0, 40)    # fixed

    def init():
        retval = WeatherEnvObj.get_value_sequence(pt_pos=pt_pos, cur_t=0, mode="mesh")
        return weather_condition_plot(retval)

    def update(step):
        retval = WeatherEnvObj.get_value_sequence(pt_pos=pt_pos, cur_t=step, mode="mesh")
        #fig2.savefig(r"D:\Project\EmergencyDeductionEngine\docs\figs\imgs\img_{:0>2d}".format(step))
        return weather_condition_plot(retval)

    ani = FuncAnimation(fig2,
                        func=update,
                        init_func=init,
                        frames=t,
                        interval=10,
                        repeat=False)
    ani.save(r"F:\ffproject\EmergencyDeduce\v0.6\EmergencyDeductionEngine\docs\figs\weather_evolution.gif")
    #with open (r"F:\ffproject\EmergencyDeducationEngine\docs\figs\weather_test.html", "w") as f:
        #print(ani.to_jshtml(), file = f)      #保存为html文件，可随时间回溯
    #plt.show()
    print("===== Test accomplished! =====")

if __name__=="__main__":
    WeatherEnvTest()






