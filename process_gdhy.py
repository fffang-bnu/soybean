import geopandas as gpd
import rasterio
import numpy as np
from rasterio.mask import mask
import matplotlib.pyplot as plt
import os
from rasterstats import zonal_stats
import pandas as pd
import pycountry
from rasterio import features

### 一、计算各单元GDHY历史平均单产

# --- 标注聚合层级 ---
def get_group_level(row):
    country = row["Country"]
    admin1 = row.get("Admin1", None)

    if country == "China":
        if admin1 in china_admin2:
            return "admin2"
        else:
            return "admin1"

    elif country in keep_county:
        return "admin2"
    elif country in agg_admin1:
        return "admin1"
    else:
        return "country"

gdhy_mean_df_copy = gdhy_mean_df.copy()
gdhy_mean_df_copy["agg_level"] = gdhy_mean_df_copy.apply(get_group_level, axis=1)

# --- 加权平均函数（稳健处理 NaN / area sum == 0） ---
def weighted_mean(df, value_col="GDHY_yield_weighted_mean", weight_col="harvested_area_total"):
    # 只保留 value 与 weight 都不是 NaN 的行
    v = df[value_col]
    w = df[weight_col]
    mask = (~v.isna()) & (~w.isna())
    if not mask.any():
        return np.nan
    w2 = w[mask]
    v2 = v[mask]
    denom = w2.sum()
    if denom == 0 or np.isclose(denom, 0.0):
        return np.nan
    return (v2 * w2).sum() / denom

# --- 聚合结果容器 ---
result_list = []

# 1) 县级：保持原样（但也确保字段类型已处理）
county_df = gdhy_mean_df_copy[gdhy_mean_df_copy["agg_level"] == "admin2"].copy()
# 确保 county 保留 Admin2、GID_2、HASC_2
# 保留列顺序：Country, Admin1, qAdmin2, GID_2, HASC_2, GDHY..., harvested_area_total, agg_level
county_df = county_df[
    ["Country", "Admin1", "Admin2", "GID_2", "HASC_2",
     "GDHY_yield_weighted_mean", "harvested_area_total", "agg_level"]
].reset_index(drop=True)
result_list.append(county_df)

# 2) 市级（Admin1）：按 Country + Admin1 分组加权求单产，汇总面积
admin1_df = (
    gdhy_mean_df_copy[gdhy_mean_df_copy["agg_level"] == "admin1"]
    .groupby(["Country", "Admin1"], as_index=False)
    .apply(lambda g: pd.Series({
        "Admin2": np.nan,
        "GID_2": np.nan,
        "HASC_2": np.nan,
        "GDHY_yield_weighted_mean": weighted_mean(g),
        "harvested_area_total": g["harvested_area_total"].sum(),
        "agg_level": "admin1"
    }))
    .reset_index(drop=True)
)
# 重新排列列
admin1_df = admin1_df[
    ["Country", "Admin1", "Admin2", "GID_2", "HASC_2",
     "GDHY_yield_weighted_mean", "harvested_area_total", "agg_level"]
]
result_list.append(admin1_df)

# 3) 国家级（Country）：按 Country 分组加权求单产，汇总面积
country_df = (
    gdhy_mean_df_copy[gdhy_mean_df_copy["agg_level"] == "country"]
    .groupby("Country", as_index=False)
    .apply(lambda g: pd.Series({
        "Admin1": np.nan,
        "Admin2": np.nan,
        "GID_2": np.nan,
        "HASC_2": np.nan,
        "GDHY_yield_weighted_mean": weighted_mean(g),
        "harvested_area_total": g["harvested_area_total"].sum(),
        "agg_level": "country"
    }))
    .reset_index(drop=True)
)

country_df = country_df[
    ["Country", "Admin1", "Admin2", "GID_2", "HASC_2",
     "GDHY_yield_weighted_mean", "harvested_area_total", "agg_level"]
]
result_list.append(country_df)

# 合并所有结果
gdhy_agg_df = pd.concat(result_list, ignore_index=True, sort=False)

### 二、计算各单元历史平均统计单产
# 将1995–2014年列转为数值类型（非数值自动变成NaN）
Top_df.loc[:, 1995:2014] = Top_df.loc[:, 1995:2014].apply(pd.to_numeric, errors='coerce')

# # 计算1995到2014年的平均值
# 选取年份列
year_cols = Top_df.loc[:, 1995:2014]
# 1. 统计每一行的非空值个数
valid_count = year_cols.count(axis=1)
# 2. 至少 5 个非空值才计算均值，否则为 NaN
Top_df["Aveyield_1995_2014"] = year_cols.mean(axis=1).where(valid_count >= 5)
Top_df_clean = Top_df[Top_df["Aveyield_1995_2014"].notnull()].copy()
Top_df_clean.reset_index(drop=True, inplace=True)

### 三、匹配各行政单元历史平均统计单产与GDHY单产
# === 确保匹配字段都是字符串类型，并去除多余空格 ===
for df in [gdhy_agg_df, Top_df_clean]:
    df["Country"] = df["Country"].astype(str).str.strip()

gdhy_agg_df["Admin1"] = gdhy_agg_df["Admin1"].astype(str).str.strip()
gdhy_agg_df["Admin2"] = gdhy_agg_df["Admin2"].astype(str).str.strip()

Top_df_clean["State/Province"] = Top_df_clean["State/Province"].astype(str).str.strip()
Top_df_clean["County/District"] = Top_df_clean["County/District"].astype(str).str.strip()

# === 合并操作 ===
merged_df = gdhy_agg_df.merge(
    Top_df_clean,
    how="left",
    left_on=["Country", "Admin1", "Admin2"],
    right_on=["Country", "State/Province", "County/District"]
)

### 四、计算&分配比例因子
merged_df["scaling_factor"] = merged_df.apply(
    lambda row: row["Aveyield_1995_2014"]/row["GDHY_yield_weighted_mean"]
                if pd.notnull(row["Aveyield_1995_2014"]) and pd.notnull(row["GDHY_yield_weighted_mean"]) and row["GDHY_yield_weighted_mean"] > 0
                else np.nan,
    axis=1
)
# === 1️⃣ 输出网格 ===
scaling_grid = np.full(gdhy_raw.shape, np.nan)

# === 2️⃣ 按层级拆分并转为字典 ===
df_country = merged_df[merged_df["agg_level"] == "country"]
df_admin1  = merged_df[merged_df["agg_level"] == "admin1"]
df_admin2  = merged_df[merged_df["agg_level"] == "admin2"]

dict_admin2 = df_admin2.set_index("GID_2")["scaling_factor"].to_dict()
dict_admin1 = df_admin1.set_index(["Country", "Admin1"])["scaling_factor"].to_dict()
dict_country = df_country.set_index("Country")["scaling_factor"].to_dict()

# === 3️⃣ 遍历 world（矢量边界） ===
for idx, row in world.iterrows():
    geom = row["geometry"]
    gid2 = row.get("GID_2")
    admin1 = row.get("NAME_1")
    country = row.get("NAME_0")

    factor = np.nan

    # --- 优先 admin2 ---
    if gid2 in dict_admin2:
        factor = dict_admin2[gid2]

    # --- 若无，则尝试 admin1 ---
    if np.isnan(factor) and (country, admin1) in dict_admin1:
        factor = dict_admin1[(country, admin1)]

    # --- 若仍无，则尝试 country ---
    if np.isnan(factor) and country in dict_country:
        factor = dict_country[country]

    if np.isnan(factor):
        continue  # 无匹配则跳过

    # === 4️⃣ 创建掩膜并赋值 ===
    mask = features.geometry_mask(
        [geom],
        out_shape=gdhy_raw.shape,
        transform=raster_transform,
        invert=True
    )
    scaling_grid[mask] = factor

gdhy_calibrated = gdhy_raw * scaling_grid
