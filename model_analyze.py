import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.graphics.api import qqplot
from sklearn import metrics


# 模型准确性评价
def forecast_accuracy(pred,actual):

    #模型评价参数
    #主要看MAE和MAPE，越小说明拟合效果越小
    #参考：https://blog.51cto.com/tecdat/2770694
    mse = metrics.mean_squared_error(actual, pred)
    rmse = metrics.mean_squared_error(actual, pred) ** 0.5
    mae = metrics.mean_absolute_error(actual, pred)
    mape = metrics.mean_absolute_percentage_error(actual, pred)
    result ={'MSE':mse,'RMSE':rmse,'MAE':mae,'MAPE':mape,}
    return result


# 残差检验
def model_evaluation(pred,actual):
    residual = (pred - actual)
    # 创建一个画像和4个子图
    fig, (axes) = plt.subplots(2, 2, figsize=(16, 8), dpi=100)

    # 绘制残差图
    # 残余误差似乎应在零均值附近波动，并且具有均匀的方差。
    axes[0, 0].plot(residual)
    axes[0, 0].set_title('residual')

    # 绘制残差直方密度图
    # 密度图建议均值为零的正态分布
    sns.distplot(residual, kde=True, fit=stats.norm,ax = axes[0][1])
    axes[0, 1].set_title('residual_plot')

    # 绘制QQ图
    # 所有圆点应与红线一致。任何明显的偏差都意味着分布偏斜
    qqplot(residual, line='q',  fit=True,ax = axes[1][0])
    axes[1, 0].set_title('QQ')

    # 绘制ACF图
    # 大部分点应落在蓝色区域内并接近于0 否则说明残差仍然相关
    sm.graphics.tsa.plot_acf(residual,ax = axes[1][1])
    axes[1, 1].set_title('ACF')

    # 自动调整子图间距
    fig.tight_layout()
    plt.show()

    #白噪声检验
    print('残差白噪声检验：', acorr_ljungbox(residual, lags=[i for i in range(1, 12)]))
    # 设置横坐标间隔为10
    # ax1.xaxis.set_major_locator(MultipleLocator(10))
    # 横坐标字体旋转
    # ax1.tick_params(axis='x', labelrotation=45)

    return


# 展示预测结果
def model_fig(pred,actual,features):

    for i in range(0, len(features)):
        plt.figure(figsize=(16, 8))
        plt.plot(actual[:,i], label='actual')
        plt.plot(pred[:,i], label='prediction')
        plt.legend(loc='upper left', fontsize=8)
        plt.title(f"LSTM:{features[i]}")
        plt.show()

    return


