import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import data_load as dl
import pmdarima as pm
from model_analyze import *
from statsmodels.tsa.arima.model import ARIMA


# 差分阶数d选择
def choose_d(data, value):
    '''
    :param data: 数据集提取出来的DataFrame对象
    :param value:DataFrame对象中需要预测的值的列索引
    :return:None
    '''
    # 时序图，用于可视化比较不同和差分阶数下数据平稳性
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    data["diff1"] = data[value].diff(1).dropna()  # 一阶差分
    data["diff2"] = data["diff1"].diff(1).dropna()  # 二阶差分
    data = data.loc[:, [value, "diff1", "diff2"]]
    data.plot(subplots=True, figsize=(18, 12), title="ARIMA差分图")
    plt.show()

    # 单位根检验，确定数据是否平稳，选择的差分阶数需要使得数据平稳，源代码数据集中无需差分数据已平稳
    # adfuller返回值
    # 第一个是adt检验的结果，也就是t统计量的值。
    # 第二个是t统计量的P值。
    # 第三个是计算过程中用到的延迟阶数。
    # 第四个是用于ADF回归和计算的观测值的个数。
    # 第五个是配合第一个一起看的，是在99%，95%，90%置信区间下的临界的ADF检验的值。如果第一个数值比“1%”下的则证明平稳。
    # 参考：https://blog.csdn.net/qq_36707798/article/details/88640684
    print("ARIMA单位根检验:\n")
    print('0阶差分：', adfuller(data[value].dropna()))  # dropna删除所有包含NaN的行
    print('1阶差分：', adfuller(data['diff1'].dropna()))
    print('2阶差分：', adfuller(data['diff2'].dropna()))

    # 白噪声检验，确定数据是否可分析,源代码数据集中数据均可分析
    # 只需要检查前12个自相关系数
    # 因为平稳时间序列具有短期相关性，短期内没关系，长期就更不可能有关系，随着步长增加，自相关系数很快趋于0
    # 如果检验结果大于0.05，说明无法拒绝原假设（该序列是白噪声序列），不具有分析意义。
    # 参考：https://zhuanlan.zhihu.com/p/439539311
    print("ARIMA白噪声检验:\n")
    print('0阶差分：', acorr_ljungbox(data[value].dropna(), lags=[i for i in range(1, 12)]))
    print('1阶差分：', acorr_ljungbox(data['diff1'].dropna(), lags=[i for i in range(1, 12)]))
    print('2阶差分：', acorr_ljungbox(data['diff2'].dropna(), lags=[i for i in range(1, 12)]))

    return data


# ARIMA作图定阶
def choose_pq(data, value):
    '''
    :param data: 数据集提取出来的DataFrame对象
    :param value: DataFrame对象中需要预测的值的列索引
    :return: None
    '''
    # 作acf图和pacf图
    # q值为acf图截尾，p值为pacf图截尾
    # 截尾是指时间序列的自相关函数（ACF）或偏自相关函数（PACF）在某阶后均为0的性质；拖尾是ACF或PACF并不在某阶后均为0的性质
    # 参考：https: // blog.csdn.net / qq_45176548 / article / details / 116771331
    # 本例中使用的0阶差分的数据（即原数据temp）来作图，选的是p = 9 ,q = 3, 使用我们自己的训练集需要根据重新作图的结果修改这两个参数
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)

    # ACF
    fig = sm.graphics.tsa.plot_acf(data[value].dropna(), lags=20, ax=ax1)  # 显示范围通过修改lags来调节
    ax2 = fig.add_subplot(212)

    # PACF
    fig = sm.graphics.tsa.plot_pacf(data[value].dropna(), lags=20, ax=ax2)
    plt.show()

    return


# 选定阶数上限和信息准则计算方法，自动选择一个最优的模型
def auto_choose_model(data, value, TRAIN_SPLIT):
    '''
    :param data: 数据集提取出来的DataFrame对象
    :param value: DataFrame对象中需要预测的值的列索引
    :return: None
    '''
    # 参考：https://www.cnblogs.com/anai/p/13139256.html
    train = data.iloc[TRAIN_SPLIT - 1000:TRAIN_SPLIT]
    test = data.iloc[TRAIN_SPLIT:]

    # auto_arima自动选择最佳参数
    model = pm.auto_arima(train[value], start_p=1, start_q=1,
                          information_criterion='bic',  # 选定的模型评价信息准则，建议使用aic或者bic
                          test='adf',  # 差分阶数选择方法，这里是adftest
                          max_p=10, max_q=10,  # 通过choose_pq大致的选定p和q的范围
                          m=30,  # 季节周期
                          d=None,  # None时机器自动选择，也可以我们自己通过choose_d选择
                          seasonal=False,  # No Seasonality
                          start_P=0,
                          D=1,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)

    return model


# 手动选择模型
# 若自动选择效果不理想，可手动选择参数
def choose_model(data, value, TRAIN_SPLIT):
    '''
    :param data: 数据集提取出来的DataFrame对象
    :param value: DataFrame对象中需要预测的值的列索引
    :return: None
    '''
    # 参考：https://www.cnblogs.com/anai/p/13139256.html
    train = data.iloc[TRAIN_SPLIT - 1000: TRAIN_SPLIT]
    test = data.iloc[TRAIN_SPLIT:]

    # 参考：https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.fit.html#statsmodels.tsa.arima.model.ARIMA.fit
    # 参数选择
    Model = ARIMA(train[value], order=(3, 1, 4))
    model = Model.fit()

    # 打印模型参数
    # 若系数很小，比如ma部分系数太小，不宜使用此模型
    # 若P > | z |列下的P - Value值远大于0.05，不宜使用此模型
    # 若AIC BIC太大，不宜使用此模型
    print(model.summary())

    # 预测
    fc = model.forecast(len(test))
    print(fc)

    # 作图
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(test[value], label='actual')
    plt.plot(fc, label='forecast')
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()

    # 残差图、残差直方密度图、QQ图、ACF图
    model_evaluation(fc, test[value])

    # 评价参数：MSE\RMSE\MAE\MAPE
    print(forecast_accuracy(fc, test[value]))

    return model







if __name__ == '__main__':
    file_name = 'dataset/compare.csv'
    data = dl.read_csv_data('Date', file_name)
    data = dl.data_wash(data)
    data = choose_d(data, 'Ave.T')
    choose_pq(data, 'diff1')
    model = auto_choose_model(data, 'Ave.T',4500)
    # model = choose_model(data, 'Ave.T',4500)


