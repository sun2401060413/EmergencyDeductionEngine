from lib.datatype import *
class CoreFunc:
    def __init__(self):
        pass

    def __format_ret(self, cid, oid, algo, data):
        retval = dict()
        retval[KEY_CLIENT_ID] = cid
        retval[KEY_OBJECT_ID] = oid
        retval[KEY_ALGORITHM] = algo
        retval[KEY_RESULT] = data
        return retval

    def core_func(self, format_data: dict):
        print("core_func")
        client_id = format_data.get(KEY_CLIENT_ID)
        object_id = format_data.get(KEY_OBJECT_ID)
        algorithm = format_data.get(KEY_ALGORITHM)
        # start_time = int(format_data.get(KEY_START_TIME))
        # stop_time = int(format_data.get(KEY_STOP_TIME))
        # step_time = int(format_data.get(KEY_STEP_TIME))
        value = ["sz test"]
        if(algorithm == ALGORITHM_LIST_FIRE):
            pass
            # burning_material_key = format_data.get(KEY_BURNING_MATERIAL)
            # burning_mass_value = format_data.get(KEY_BURNING_MASS)
            # history_fighting_time_list = format_data.get(
            #     KEY_HISTORY_FIGHTING_TIME_LIST)
            # history_fighting_material_list = format_data.get(
            #     KEY_HISTORY_FIGHTING_MATERIAL_LIST)
            # value = self.fire_burning.fire_stub(burning_material_key, burning_mass_value, history_fighting_time_list,
            #                                      history_fighting_material_list, start_time, stop_time, step_time)

        elif(algorithm == ALGORITHM_LIST_VEHICLE_HEAT):
            pass
            # burning_cause = format_data.get(KEY_BURNING_CAUSE)
            # vehicle_type = format_data.get(KEY_VEHICLE_TYPE)
            # value = self.vh.vehicle_heat_stub(start_time, stop_time, step_time,
            #                                    burning_cause, vehicle_type)

        elif(algorithm == ALGORITHM_LIST_BRIGHTNESS):
            pass
            # init_brightness = format_data.get(KEY_INIT_BRIGHTNESS)
            # weather_status = format_data.get(KEY_WEATHER_STATUS)
            # location = format_data.get(KEY_LOCATION)
            # value = self.bt.brightness_stub(start_time, stop_time, step_time,
            #                                  init_brightness, weather_status, location)

        elif(algorithm == ALGORITHM_LIST_OXYGEN):
            pass
            # init_oxygen = format_data.get(KEY_INITOXYGEN)
            # burn_strength = format_data.get(KEY_BURN_STRENGTH)
            # distance = format_data.get(KEY_DISTANCE)
            # value = self.oxyden.oxygen_stub(start_time, stop_time, step_time,
            #                                  init_oxygen, burn_strength, distance)

        elif(algorithm == ALGORITHM_LIST_PEOPLE_STATUS):
            pass
            # init_people_status = format_data.get(KEY_INIT_PEOPLE_STATUS)
            # burning_strength = format_data.get(KEY_BURN_STRENGTH)
            # distance = format_data.get(KEY_DISTANCE)
            # start_stop_fight_time = format_data.get(KEY_START_STOP_FIGHT_TIME)
            # rescue_material = format_data.get(KEY_RESCUE_MATERIAL)
            # age = format_data.get(KEY_AGE)
            # vo2 = format_data.get(KEY_VO2)
            # location = format_data.get(KEY_LOCATION)
            # value = self.ps.people_in_vehicle_status_stub(start_time, stop_time, step_time,
            #                                     init_people_status, age,vo2)

        elif(algorithm == ALGORITHM_LIST_VEHICLE_STATUS):
            pass
            # init_vehicle_status = format_data.get(KEY_INIT_VEHICLE_STATUS)
            # burning_strength = format_data.get(KEY_BURN_STRENGTH)
            # distance = format_data.get(KEY_DISTANCE)
            # location = format_data.get(KEY_LOCATION)
            # vehicle_type = format_data.get(KEY_VEHICLE_TYPE)
            # value = self.vehicle.vehicle_stub(start_time, stop_time, step_time, vehicle_type,
            #                                    init_vehicle_status, burning_strength, distance, location)

        elif(algorithm == ALGORITHM_LIST_EVN_TEMP):
            pass
            # distance = format_data.get(KEY_DISTANCE)
            # value = env_temp_stub(start_time, stop_time, step_time, distance)

        elif(algorithm == ALGORITHM_LIST_VISIBILITY):
            pass
            # burning_cause = format_data.get(KEY_BURNING_CAUSE)
            # vehicle_type = format_data.get(KEY_VEHICLE_TYPE)
            # distance = format_data.get(KEY_DISTANCE)
            # value = self.visibility.visibility_stub(start_time, stop_time, step_time,
            #                                          distance, burning_cause, vehicle_type)

        else:
            pass
        retval = self.__format_ret(client_id,object_id,algorithm,value)
        return retval

    def test_func(self, format_data: dict):
        return "test_func"