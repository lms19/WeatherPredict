import data_get as dg
import datetime
from lstm import *
import pandas as pd
import os
import pickle

def handleCityAndModel(city, model):
    print("handleCityAndModel方法收到的city:", city)
    print("handleCityAndModel方法收到的model:", model)

    # watherDataHeader = ['date', 'Ave.T', 'Max.T', 'Min.T', 'Prec(mm)', 'Press(Hpa)', 'Wind dir', 'Wind sp(km/h)']
    weatherData1 = ['2022.8.27', '11', '12', '13', '-', '15', '16', '17']
    weatherData2 = ['2022.8.28', '21', '22', '23', '-', '25', '26', '27']
    weatherData3 = ['2022.8.29', '31', '32', '33', '-', '35', '36', '37']
    weatherData4 = ['2022.8.30', '41', '42', '43', '-', '45', '46', '47']
    weatherData5 = ['2022.8.31', '51', '52', '53', '-', '55', '56', '57']
    weatherData6 = ['2022.9.1', '61', '62', '63', '-', '65', '66', '67']
    weatherData7 = ['2022.9.2', '71', '72', '73', '-', '75', '76', '77']
    weatherDatas = [weatherData1, weatherData2, weatherData3, weatherData4, weatherData5, weatherData6, weatherData7]

    # 调用模型的处理
    if city == '北京':
        city = 'beijing'
        ind = 54511
    if city == '重庆':
        city = 'chongqing'
        ind = 57516
    if city == '广州':
        city = 'guangzhou'
        ind = 59287
    if city == '南京':
        city = 'nanjing'
        ind = 58238
    if city == '青岛':
        city = 'qingdao'
        ind = 54857
    if city == '三亚':
        city = 'sanya'
        ind = 59948
    if city == '上海':
        city = 'shanghai'
        ind = 58362
    if city == '深圳':
        city = 'shenzhen'
        ind = 59493
    if city == '武汉':
        city = 'wuhan'
        ind = 57494

    if model == 'LSTM':
        # 爬取前20天的数据
        end = datetime.datetime.now()
        start = (end - datetime.timedelta(days=20)).date()
        dg.getdata(str(ind), city, start, end)
        data = dl.read_csv_data('Date', f'{city}.csv')
        os.remove(f'{city}.csv')

        # 由于原网站的日期有bug，需要重新生成日期索引
        data.index = pd.date_range(start=start, end=end - datetime.timedelta(days=1), freq='1D')

        # 选择天气特征
        features = ['Ave.T', 'Max.T', 'Min.T', 'Prec(mm)', 'Press(Hpa)', 'Wind dir', 'Wind sp (km/h)']
        data = data[features]

        # 输入数据处理
        dataset, data_std, data_mean = data_standardization(data)
        fill_ndarray(dataset)
        dataset = dataset.reshape(1, 20, 7)

        # 调用模型
        model = tf.keras.models.load_model(f'model/{city}7.h5')
        pred = data_reduct(model.predict(dataset), data_std, data_mean).reshape(7, 7)
        pred = np.around(pred, decimals=2)
        pred = pred.astype(str)

        # 将预测结果写入weatherDatas
        j = 0
        for i in weatherDatas:
            i[0] = (datetime.datetime.now() + datetime.timedelta(days=j)).strftime('%Y.%m.%d')
            i[1:8] = pred[j]
            j = j + 1

    if model == 'ARIMA':
        # 预测范围
        start = datetime.datetime.now()
        end = (start + datetime.timedelta(days=7)).date()
        range = pd.date_range(start, end, freq='1D')

        # 将预测结果写入weatherDatas
        j = 0
        for i in range:
            loaded_model1 = pickle.load(open(f'model/{city} Ave.T.pkl', 'rb'))
            loaded_model2 = pickle.load(open(f'model/{city} Max.T.pkl', 'rb'))
            loaded_model3 = pickle.load(open(f'model/{city} Min.T.pkl', 'rb'))
            loaded_model4 = pickle.load(open(f'model/{city} Press(Hpa).pkl', 'rb'))
            loaded_model5 = pickle.load(open(f'model/{city} Wind dir.pkl', 'rb'))
            loaded_model6 = pickle.load(open(f'model/{city} Wind sp.pkl', 'rb'))
            weatherDatas[j][0] = i.strftime('%Y.%m.%d')
            # 保留两位小数
            weatherDatas[j][1] = str(round(loaded_model1.predict(i).values[0], 2))
            weatherDatas[j][2] = str(round(loaded_model2.predict(i).values[0], 2))
            weatherDatas[j][3] = str(round(loaded_model3.predict(i).values[0], 2))
            weatherDatas[j][5] = str(round(loaded_model4.predict(i).values[0], 2))
            weatherDatas[j][6] = str(round(loaded_model5.predict(i).values[0], 2))
            weatherDatas[j][7] = str(round(loaded_model6.predict(i).values[0], 2))
            j += 1

    print("handleCityAndModel方法返回的：", weatherDatas)
    return weatherDatas


if __name__ == "__main__":
    handleCityAndModel('北京', 'LSTM')
