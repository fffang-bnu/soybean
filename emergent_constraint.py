import numpy as np
from scipy import stats

## 定义作物模型、气候模式、时段等
cm = ["acea", "epic-iiasa", "ldndc", "pdssat", "pepic", "simplace-lintul5"]
# 气候模式
gcm = ["gfdl-esm4", "ipsl-cm6a-lr", "mpi-esm1-2-hr", "mri-esm2-0", "ukesm1-0-ll"]
# 情景
scenario = ["ssp126", "ssp370", "ssp585"]
# 时段
period = ['historical', '1.5 °C', '2.0 °C']

## Warming level alignment：计算温升时段的模拟yield均值
## 360*720*2，2个时间段（1.5°C和2.0°C）
AVE_cm_lr = np.full([360, 720, 2], np.nan)  # 均值_线性回归
STD_cm_lr = np.full([360, 720, 2], np.nan)  # 标准差_线性回归
R_CO2 = np.full([360, 720, 2], np.nan)  # 相关系数^2

for i in range(360):
    for j in range(720):
        pixel = AVE_YIELD_copy[i, j, :, :, :, :]  # raw multi-model simulations
        Pixel = pixel.reshape(len(gcm)*(len(cm))*len(scenario), 3)

        # 各个时间段下的多个模型模拟的产量结果
        x = Pixel[:, 0]  # 历史
        y1_v0 = Pixel[:, 1]  # 1.5°C
        y1 = y1_v0[~(np.isnan(x) | np.isnan(y1_v0))]  # 将x和y对应的nan值均删去
        x1 = x[~(np.isnan(x) | np.isnan(y1_v0))]
        y2_v0 = Pixel[:, 2]  # 2°C
        y2 = y2_v0[~(np.isnan(x) | np.isnan(y2_v0))]  # 将x和y对应的nan值均删去
        x2 = x[~(np.isnan(x) | np.isnan(y2_v0))]

        if ((not np.all(x1 == 0)) and (not np.isnan(GDHY_corrected[i, j]))):
            # 线性回归
            slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x1, y1)  # 拟合线性关系 x1与y1
            r2_linear1 = pow(r_value1, 2)  # 计算 R²

            if p_value1<0.05:
                AVE_cm_lr[i, j, 0] = slope1 * GDHY_corrected[i, j] + intercept1  # 实测单产带入拟合方程中
                y1f = slope1 * x1 + intercept1  # 预测值 1.5°C
                R_CO2[i, j, 0] = r2_linear1  # 历史单产与温升单产的相关系数

                # std
                m1 = len(x1[~np.isnan(x1)])
                aa1 = np.cumsum(pow(y1 - y1f, 2))[-1]  # 模拟值与预测值差距的平方 的 累积和
                s1 = pow(aa1/(m1-2), 0.5)  # 获得s
                o = np.cumsum(pow(x1-np.mean(x1), 2))[-1]
                oo = pow((GDHY_corrected[i, j] - np.mean(x1)), 2) / o
                ooo = pow(1 + 1/(m1) + oo, 0.5)
                o1 = s1 * ooo
                STD_cm_lr[i, j, 0] = o1  # 1.5°C

        else:
            AVE_cm_lr[i, j, 0] = np.nan
            STD_cm_lr[i, j, 0] = np.nan
            R_CO2[i, j, 0] = np.nan

        if ((not np.all(x2 == 0)) and (not np.isnan(GDHY_corrected[i, j]))):
            # 线性回归
            slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x2, y2)  # 拟合线性关系 x2与y2
            r2_linear2 = pow(r_value2, 2)  # 计算 R²

            if p_value2<0.05:
                AVE_cm_lr[i, j, 1] = slope2 * GDHY_corrected[i, j] + intercept2  # 实测单产带入拟合方程中
                y2f = slope2 * x2 + intercept2  # 预测值 2.0°C
                R_CO2[i, j, 1] = r2_linear2  # 历史单产与温升单产的相关系数

                # std
                m2 = len(x2[~np.isnan(x2)])
                aa2 = np.cumsum(pow(y2 - y2f, 2))[-1]  # 模拟值与预测值差距的平方 的 累积和
                s2 = pow(aa2/(m2-2), 0.5)  # 获得s
                o = np.cumsum(pow(x2-np.mean(x2), 2))[-1]
                oo = pow((GDHY_corrected[i, j] - np.mean(x2)), 2) / o
                ooo = pow(1 + 1/(m2) + oo, 0.5)
                o2 = s2 * ooo
                STD_cm_lr[i, j, 1] = o2  # 2.0°C

        else:
            AVE_cm_lr[i, j, 1] = np.nan
            STD_cm_lr[i, j, 1] = np.nan
            R_CO2[i, j, 1] = np.nan