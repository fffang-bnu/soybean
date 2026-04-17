import os
import xarray as xr
import numpy as np
import progressbar


### 一、面积加权平均

def Area_Mean(data, lat, lon):
    """
    data: 要进行区域加权平均的变量，支持2、3维  2D: [lat, lon]  3D：[time, lat, lon]
    lat: data2D对应的纬度 1D
    lon: data2D对应的经度 1D
    """

    if data.ndim == 2:
        y_weight2D = abs(np.cos(lat * np.pi / 180))
        weight2D = np.expand_dims(y_weight2D, 1).repeat(len(lon), axis=1)
        # print(weight2D)
        new_data = np.average(data, weights=weight2D)
        return new_data
    elif data.ndim == 3:
        y_weight2D = abs(np.cos(lat * np.pi / 180))
        weight2D = np.expand_dims(y_weight2D, 1).repeat(len(lon), axis=1)
        weight3D = np.expand_dims(weight2D, 0).repeat(len(data[:, 0, 0]), axis=0)
        # print(weight3D.shape)
        new_data = np.average(data, weights=weight3D, axis=(-1, -2))
        return new_data
    else:
        print("输入数据非2&3维")


gcms = ["GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL"]
scenarios = ["historical", "ssp126", "ssp370", "ssp585"]
path = r"F:\data_soybean\temperature"

# tas_set_his计算示例，tas_set_ssp126/370/585类推
gcm = gcms[0]
scenario = scenarios[0]
tas_set_his = xr.Dataset(data_vars={"tas_area_mean": (["year"], [])}, coords={"year": []})

for filewalks in os.walk(path + "\\" + gcm):
    for files in filewalks[2]:
        if scenario in files:
            bar = progressbar.ProgressBar(
                widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()]
            )
            bar.start()

            # 读取nc4文件
            file_path = os.path.join(filewalks[0], files)
            xr_data = xr.open_dataset(file_path)
            try:
                xr_data_year = xr_data.groupby("time.year")
                xr_data_yearlymean = xr_data_year.mean(dim=("time"), skipna=True)
                year_num = len(xr_data_yearlymean.year.values)
                tas = xr_data_yearlymean.tas
                lat = xr_data_yearlymean.lat
                lon = xr_data_yearlymean.lon
                year = xr_data_yearlymean.year

                # 面积加权平均
                tas_yearlymean = xr.Dataset(
                    data_vars={"tas_area_mean": (["year"], Area_Mean(tas, lat, lon))},
                    coords={"year": year},
                )
                tas_set_his = xr.concat([tas_set_his, tas_yearlymean], dim="year")
            except:
                print("文件" + str(file_path) + "读取失败")
            bar.finish()

tas_set_his = tas_set_his.sortby("year")
tas_set_his.to_netcdf(path + "\\" + gcm + "_tas_set_his.nc")


### 二、20年滑动平均
# 基准期（1995-2014年）温度参考值
tas_set_his_mean = tas_set_his.sel(year=slice(1995, 2014)).mean().tas_area_mean.values

# SSP126示例
# 滑动平均 20年 （min_periods=2避免出现空值）【结合历史】
tas_set_ssp126his = xr.Dataset(
    data_vars={"tas_area_mean": (["year"], [])}, coords={"year": []},
)
tas_set_ssp126his = xr.concat([tas_set_his, tas_set_ssp126], dim="year")
tas_set_ssp126_roll = tas_set_ssp126his.rolling(year=20, center=True).mean()
# 计算与基准期的温差
tempDiff_ssp126 = tas_set_ssp126_roll - tas_set_his_mean
tempDiff_ssp126_df = tempDiff_ssp126.to_dataframe()

try:
    centralyear_o99_ssp126 = tempDiff_ssp126_df[
        tempDiff_ssp126_df.tas_area_mean >= 0.99-0.85
    ].index[0]
    period_o99_ssp126 = [
        centralyear_o99_ssp126 - 10,
        centralyear_o99_ssp126 + 9,
    ]
    print("SSP1-2.6 " + str(gcm) + " 0.99°C对应的时间段: " + str(period_o99_ssp126))
except:
    period_o99_ssp126 = []
    print("SSP1-2.6 " + str(gcm) + " 0.99°C对应的时间段不存在！！")

try:
    centralyear_15_ssp126 = tempDiff_ssp126_df[
        tempDiff_ssp126_df.tas_area_mean >= 0.65
    ].index[0]
    period_15_ssp126 = [
        centralyear_15_ssp126 - 10,
        centralyear_15_ssp126 + 9,
    ]
    print("SSP1-2.6 " + str(gcm) + " 1.5°C对应的时间段: " + str(period_15_ssp126))
except:
    period_15_ssp126 = []
    print("SSP1-2.6 " + str(gcm) + " 1.5°C对应的时间段不存在！！")

try:
    centralyear_20_ssp126 = tempDiff_ssp126_df[
        tempDiff_ssp126_df.tas_area_mean >= 1.15
    ].index[0]
    period_20_ssp126 = [
        centralyear_20_ssp126 - 10,
        centralyear_20_ssp126 + 9,
    ]
    print("SSP1-2.6 " + str(gcm) + " 2.0°C对应的时间段: " + str(period_20_ssp126))
except:
    period_20_ssp126 = []
    print("SSP1-2.6 " + str(gcm) + " 2.0°C对应的时间段不存在！！")
