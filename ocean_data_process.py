import numpy as np
import pandas as pd
import time
import datetime
import math


def ocean_data_load():
    atlantic_2011_2021 = np.load("/root/Ocean_data_process/atlantic_Y_2011_2021.npy", allow_pickle=True)
    india_2011_2021 = np.load("/root/Ocean_data_process/india_Y_2011_2021.npy", allow_pickle=True)
    pacific_2011_2021 = np.load("/root/Ocean_data_process/pacific_Y_2011_2021.npy", allow_pickle=True)
    data_frame = []
    for i in atlantic_2011_2021:
        for j in i:
            data_frame.append(j)
    for i in india_2011_2021:
        for j in i:
            data_frame.append(j)
    for i in pacific_2011_2021:
        for j in i:
            data_frame.append(j)
    dataf = pd.DataFrame(columns=['time', 'float_id', 'lat', 'lon', 'temp'], data=data_frame)
    return dataf


def data_sort_and_process(data):
    ds = data.sort_values(['float_id', 'time'])
    ds = ds.reset_index(drop=True)
    ds["lat"] = pd.to_numeric(ds["lat"])
    ds["lon"] = pd.to_numeric(ds["lon"])
    ds["temp"] = pd.to_numeric(ds["temp"], errors='coerce')
    ds['time'] = pd.to_datetime(ds['time'], format='%Y%m%d')
    ds = ds[~ds['float_id'].isin(['n/a'])]
    return ds


def calcaute_float_num(data):
    float_list = data.drop_duplicates(subset='float_id')['float_id'].to_list()
    return len(float_list)


def delete_float_id_by_lat_lon(data):
    float_list = data.drop_duplicates(subset='float_id')['float_id'].to_list()
    not_used_float = []
    for float_id in float_list:
        try:
            d = data[data['float_id'] == float_id]
            lat_d = list(d['lat'])
            lon_d = list(d['lon'])
            # temp_d = list(data[data['float_id'] == float_id]['temp'])
            lat_d_1 = np.absolute(np.array(lat_d[:-1]))
            lat_d_2 = np.absolute(np.array(lat_d[1:]))
            lon_d_1 = np.absolute(np.array(lon_d[:-1]))
            lon_d_2 = np.absolute(np.array(lon_d[1:]))
            # temp_d_1 = np.array(temp_d[:-1])
            # temp_d_2 = np.array(temp_d[1:])
            lat_d_3 = math.fabs((lat_d_2 - lat_d_1).max())
            lon_d_3 = math.fabs((lon_d_2 - lon_d_1).max())
            # temp_d_3 = math.fabs((temp_d_2 - temp_d_1).max())
            if lat_d_3 > 5 or lon_d_3 > 5:
                not_used_float.append(float_id)
                print('not_float_id',float_id,lat_d_3,lon_d_3)
        except:
            not_used_float.append(float_id)
            print(float_id)
            continue
        # lat_d = list(data[data['float_id'] == float_id]['lat'])
        # lon_d = list(data[data['float_id'] == float_id]['lon'])
        # temp_d = list(data[data['float_id'] == float_id]['temp'])
        # for i in range(len(lat_d) - 1):
        #     tmp_lat = math.fabs(lat_d[i + 1]) - math.fabs(lat_d[i])
        #     tmp_lon = math.fabs(lon_d[i + 1]) - math.fabs(lon_d[i])
        #     tmp_temp = math.fabs(temp_d[i + 1]) - math.fabs(temp_d[i])
        #     if math.fabs(tmp_lat) > 5 or math.fabs(tmp_lon) > 5 or math.fabs(tmp_temp) > 5:
        #         not_used_float.append(float_id)
        #         break
    print("delete_float_id_by_lat_lon_num is", len(not_used_float))
    ds = data[~data['float_id'].isin(not_used_float)]
    return ds


def delete_float_id_by_num(data, frequnet: 100):
    g = data.groupby('float_id').size() > frequnet
    s = pd.DataFrame(g)
    y = s.reset_index()
    y.columns = ['float', 'true']
    float_list = list(y[y['true'] == True]['float'])
    ds = data[data['float_id'].isin(float_list)]
    return ds


def data_time_process(data):
    time_all = []
    start_time = datetime.datetime(2011, 1, 1)
    end_time = datetime.datetime(2021, 12, 1)
    while 1:
        if start_time > end_time:
            break
        else:
            time_all.append(start_time)
            start_time += datetime.timedelta(days=10)
    float_dict = []
    a = float('nan')
    float_list = data.drop_duplicates(subset='float_id')['float_id'].to_list()
    float_sensor_num = np.full([len(float_list), len(time_all), 3], np.nan)
    for float_index in range(len(float_list)-1):
        d = data[data['float_id'] == float_list[float_index]]
        tmp_time = 0
        for row in d.iterrows():
            time = row[1]['time']
            temp = row[1]['temp']
            lon = row[1]['lon']
            lat = row[1]['lat']
            while tmp_time < len(time_all)-1 and time_all[tmp_time+1] < time:
                tmp_time += 1
            if np.isnan(float_sensor_num[float_index][tmp_time][0]):
                float_sensor_num[float_index][tmp_time] = np.array([temp, lon, lat])
            else:
                float_sensor_num[float_index][tmp_time] = (float_sensor_num[float_index][tmp_time] + np.array([temp, lon, lat])) / 2
    np.save("float_sensor_num.npy", float_sensor_num)
    return 0

    def data_time_process(data):
        time_all = []
        start_time = datetime.datetime(2011, 1, 1)
        end_time = datetime.datetime(2021, 12, 1)
        while 1:
            if start_time > end_time:
                break
            else:
                time_all.append(start_time)
                start_time += datetime.timedelta(days=10)
        float_dict = []
        a = float('nan')
        float_list = data.drop_duplicates(subset='float_id')['float_id'].to_list()
        float_sensor_num = np.full([len(float_list), len(time_all), 3], np.nan)
        for float_index in range(len(float_list) - 1):
            d = data[data['float_id'] == float_list[float_index]]
            tmp_time = 0
            for row in d.iterrows():
                time = row[1]['time']
                temp = row[1]['temp']
                lon = row[1]['lon']
                lat = row[1]['lat']
                while tmp_time < len(time_all) - 1 and time_all[tmp_time + 1] < time:
                    tmp_time += 1
                if np.isnan(float_sensor_num[float_index][tmp_time][0]):
                    float_sensor_num[float_index][tmp_time] = np.array([temp, lon, lat])
                else:
                    float_sensor_num[float_index][tmp_time] = (float_sensor_num[float_index][tmp_time] + np.array(
                        [temp, lon, lat])) / 2
        np.save("float_sensor_num.npy", float_sensor_num)
        return 0


def data_time_process_old(data):
    time_all = []
    start_time = datetime.datetime(2011, 1, 1)
    end_time = datetime.datetime(2021, 12, 1)
    while 1:
        if start_time > end_time:
            break
        else:
            time_all.append(start_time)
            start_time += datetime.timedelta(days=10)
    float_dict = []
    a = float('nan')
    float_list = data.drop_duplicates(subset='float_id')['float_id'].to_list()
    for float_index in float_list:
        print(float_index)
        d = data[data['float_id'] == float_index]
        ts = 0
        dicts = []
        for ss in range(len(time_all)):
            dicts.append([])
        index = 1
        while index < len(d) + 1:
            if ts >= len(time_all) - 1:
                break
            s_time = time_all[ts]
            e_time = time_all[ts + 1]
            row = d.iloc[index - 1:index]
            time = row['time'].tolist()[0]
            temp = row['temp'].tolist()[0]
            lon = row['lon'].tolist()[0]
            lat = row['lat'].tolist()[0]
            if s_time <= time < e_time:
                if len(dicts[ts]) == 0:
                    dicts[ts].append([time, float_index, lat, lon, temp])
                    index += 1
                else:
                    tmp_1 = dicts[ts][0]
                    dicts[ts] = []
                    tmp_time = tmp_1[0]
                    tmp_lat = (tmp_1[2] + lat) / 2
                    tmp_lon = (tmp_1[3] + lon) / 2
                    tmp_temp = (tmp_1[4] + temp) / 2
                    dicts[ts].append([tmp_time, float_index, tmp_lat, tmp_lon, tmp_temp])
                    index += 1
            else:
                if len(dicts[ts]) == 0:
                    dicts[ts].append([s_time+datetime.timedelta(days=5), float_index, a, a, a])
                ts += 1
        while ts <= len(time_all) - 1:
            dicts[ts].append([s_time, float_index, a, a, a])
            ts += 1
        float_dict.append(dicts)
    data_process = []
    for i in float_dict:
        for j in i:
            data_process.append(j[0])
    dateframe = pd.DataFrame(columns=['time', 'float_id', 'lat', 'lon', 'temp'], data=data_process)
    # dateframe['temp'] = dateframe['temp'].interpolate()
    # dateframe['lat'] = dateframe['lat'].interpolate()
    # dateframe['lon'] = dateframe['lon'].interpolate()
    return dateframe


def numpy_to_csv():
    df = pd.read_csv('/root/Ocean_sensor_model/data_delete_by_lat_lon_2011_2021.csv')
    # ds['time'] = pd.to_datetime(ds['time'], format='%Y-%m-%d')
    # data_process = data_time_process(ds)
    # data_process.to_csv('/root/Ocean_sensor_model/data_process_2011_2021_null.csv', index=None)
    data = np.load("float_sensor_num.npy")
    float_list = df.drop_duplicates(subset='float_id')['float_id'].to_list()
    dataframe_list = []
    time_all = []
    start_time = datetime.datetime(2011, 1, 1)
    end_time = datetime.datetime(2021, 12, 1)
    while 1:
        if start_time > end_time:
            break
        else:
            time_all.append(start_time)
            start_time += datetime.timedelta(days=10)
    for i in range(len(float_list)):
        float_id = float_list[i]
        data_float = data[i]
        for j in range(len(time_all)):
            dataframe_list.append([float_id, time_all[j], data_float[j][1], data_float[j][2], data_float[j][0]])
    data_full = pd.DataFrame(columns=['float_id', 'time', 'lat', 'lon', 'temp'], data=dataframe_list)
    data_full.to_csv('/root/Ocean_sensor_model/data_process_2011_2021_null_numpy.csv', index=None)


def create_float_nan_array():
    data = np.load("float_sensor_num.npy")
    data_float = np.zeros([data.shape[0], data.shape[1]])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if ~np.isnan(data[i][j][0]):
                data_float[i][j] = 1
    np.save('float_nan.npy', data_float)


if __name__ == '__main__':
    # data = ocean_data_load()
    # data_sort = data_sort_and_process(data)
    # data_sort.to_csv('/root/Ocean_sensor_model/all_ocean_2011_2021.csv', index=None)
    # data_delete_by_lat_lon = delete_float_id_by_lat_lon(data_sort)
    # data_delete_by_lat_lon.to_csv('/root/Ocean_sensor_model/data_delete_by_lat_lon_2011_2021.csv', index=None)
    # ds = pd.read_csv('/root/Ocean_sensor_model/data_delete_by_lat_lon_2011_2021.csv')
    # ds['time'] = pd.to_datetime(ds['time'], format='%Y-%m-%d')
    # data_process = data_time_process(ds)
    # data_process.to_csv('/root/Ocean_sensor_model/data_process_2011_2021_null.csv', index=None)
    data = np.load("float_nan.npy")
    data_count = np.sum(data, axis=0)
    data_sort_count = np.sort(data_count)
    print(1)





