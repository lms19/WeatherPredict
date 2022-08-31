import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox


# 读取文件
def read_csv_data(x, file_name):
    '''
    :param x: 横轴，作为索引
    :param file_name: 读取的文件
    :return: 读取后构建的DataFrame对象
    '''
    data = pd.read_csv(file_name)
    data.set_index(x, inplace=True)
    data.index = pd.DatetimeIndex(data.index)
    data = pd.DataFrame(data, dtype=np.float64)
    data = data.iloc[::-1]

    return data


# 清洗数据
def data_wash(data):
    '''
    :param data: 数据集提取出来的DataFrame对象
    :return:None
    '''
    # 参考：https://zhuanlan.zhihu.com/p/60241672
    # 参考：https://blog.csdn.net/weixin_42163563/article/details/121071827?ops_request_misc=&request_id=&biz_id=102&utm_term=arima%E6%95%B0%E6%8D%AE%E6%B8%85%E6%B4%97&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-121071827.142^v42^pc_rank_34_1,185^v2^control&spm=1018.2226.3001.4187

    # 查看清洗前数据
    print(data.describe())
    print(data.info())

    # 插值
    data.interpolate(method='time', limit_direction='both', inplace=True)

    # 去除重复值
    # data.drop_duplicates(inplace=True)

    # 查看清洗结果
    print(data.describe())
    print(data.info())

    return data


# 查看原始数据
def show_fig(data):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    data = data.loc[:, ['Ave.T', 'Max.T', 'Min.T', 'Prec(mm)', 'Press(Hpa)', 'Wind dir', 'Wind sp (km/h)', 'Cloud.c']]
    data.plot(subplots=True, figsize=(18, 12), title="气象图")
    plt.show()

    return


# 原始数据的白噪声检验
def data_test(data):
    print("白噪声检验:\n")
    print('Ave.T：', acorr_ljungbox(data['Ave.T'].dropna(), lags=[i for i in range(1, 12)]))
    print('Max.T：', acorr_ljungbox(data['Max.T'].dropna(), lags=[i for i in range(1, 12)]))
    print('Min.T；', acorr_ljungbox(data['Min.T'].dropna(), lags=[i for i in range(1, 12)]))
    print('Prec(mm):', acorr_ljungbox(data['Prec(mm)'].dropna(), lags=[i for i in range(1, 12)]))
    print('Press(Hpa)：', acorr_ljungbox(data['Press(Hpa)'].dropna(), lags=[i for i in range(1, 12)]))
    print('Wind dir', acorr_ljungbox(data['Wind dir'].dropna(), lags=[i for i in range(1, 12)]))
    print('Wind sp (km/h)：', acorr_ljungbox(data['Wind sp (km/h)'].dropna(), lags=[i for i in range(1, 12)]))
    print('Cloud.c', acorr_ljungbox(data['Cloud.c'].dropna(), lags=[i for i in range(1, 12)]))

    return


if __name__ == '__main__':
    file_name = 'beijing.csv'
    data = read_csv_data('Date', file_name)
    data = data_wash(data)
    show_fig(data)
    data_test(data)
