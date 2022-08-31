from lstm import *
from arima import *
import data_load as dl
import joblib


def lstm_model(data, TRAIN_SPLIT):
    # 数据标准化
    dataset, data_std, data_mean = data_standardization(data)

    # 历史步长与预测步长
    past_history = 20
    future_target = 1

    # 特征与标签切片
    x_train, y_train = data_slice(dataset, 0, TRAIN_SPLIT,
                                  past_history,
                                  future_target)
    x_val, y_val = data_slice(dataset, TRAIN_SPLIT - past_history, None,
                              past_history,
                              future_target)

    # 让用户选择是否重新拟合LSTM
    while (1):
        use_lstm = input("是否重新拟合LSTM？(y/n)")
        if use_lstm == 'y':
            # 模型保存
            model = multivariate_model(dataset, x_train, y_train, x_val, y_val)
            model.save('model/compare_lstm.h5')
            break
        if use_lstm == 'n':
            model = tf.keras.models.load_model('model/compare_lstm.h5')
            break
        else:
            print('输入错误！')

    pred = data_reduct(model.predict(x_val).reshape(dataset.shape[0] - TRAIN_SPLIT - future_target, dataset.shape[1]),
                       data_std[:dataset.shape[1]], data_mean[:dataset.shape[1]])
    actual = data_reduct(y_val.reshape(dataset.shape[0] - TRAIN_SPLIT - future_target, dataset.shape[1]),
                         data_std[:dataset.shape[1]], data_mean[:dataset.shape[1]])

    # 作图显示预测结果
    model_fig(pred, actual, features)
    # 残差图、残差直方密度图、QQ图、ACF图、白噪声检验
    model_evaluation(pred.reshape(-1), actual.reshape(-1))
    # MSE、RMSE、MAE、MAPE
    result = forecast_accuracy(pred, actual)
    print('LSTM: ',result)

    return


def arima_model(data, TRAIN_SPLIT):
    # 选择参数d,ADF检验,白噪声检验
    data = choose_d(data, 'Ave.T')
    # 绘制ACF/PACF图,选择参数p、q
    choose_pq(data, 'diff1')
    # 划分测试集
    test = data.iloc[TRAIN_SPLIT:]

    # 用户选择是否重新拟合ARIMA
    while (1):
        use_lstm = input("是否重新拟合ARIMA？(y/n)")
        if use_lstm == 'y':
            # 模型保存
            model = auto_choose_model(data, 'Ave.T', TRAIN_SPLIT)
            joblib.dump(model, 'model/compare_arima.pkl')
            break
        if use_lstm == 'n':
            model = joblib.load('model/compare_arima.pkl')
            break
        else:
            print('输入错误！')

    # 打印模型参数
    print('ARIMA模型参数: ')
    print(model.summary())
    # 预测结果和置信区间
    fc, confint = model.predict(n_periods=len(test), return_conf_int=True)

    # 调整index
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(confint[:, 0], index=test.index)  # 95%置信区间的下限
    upper_series = pd.Series(confint[:, 1], index=test.index)  # 95%置信区间的上限

    # 作图显示预测结果
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(test[features], label='actual')
    plt.plot(fc_series, label='forecast')
    plt.fill_between(lower_series.index,

                     lower_series,
                     upper_series,
                     color='k', alpha=.15)
    plt.title('ARIMA: Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()

    # 残差图、残差直方密度图、QQ图、ACF图
    model_evaluation(fc_series, test[features[0]])
    # 评价参数：MSE\RMSE\MAE\MAPE
    print('ARIMA: ',forecast_accuracy(fc_series, test[features[0]]))

    return


if __name__ == '__main__':
    # 划分训练集
    TRAIN_SPLIT = 4500

    # 读取数据
    data = dl.read_csv_data('Date', 'dataset/compare.csv')
    data = dl.data_wash(data)

    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 用来正常显示负号
    plt.rcParams['axes.unicode_minus'] = False

    # 特征选择
    features = ['Ave.T']
    data = data[features]

    # LSTM模型
    lstm_model(data, TRAIN_SPLIT)

    # ARIMA模型
    arima_model(data, TRAIN_SPLIT)
