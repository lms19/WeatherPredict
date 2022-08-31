import urllib3
import os
import re
from bs4 import BeautifulSoup
import csv
import pandas as pd
import time
import datetime


# 配置请求
class GetData:
    url = ""
    headers = ""
    proxies = ""

    def __init__(self, url, header=""):
        """
        :param url: 获取的网址
        :param header: 请求头，默认已内置
        """
        self.url = url

        if header == "":

            self.headers = {
                'Connection': 'Keep-Alive',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,'
                          '*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.81 Safari/537.36 Edg/104.0.1293.54',
                'Host': 'www.meteomanz.com'
            }

        else:
            self.headers = header

    def Get(self):
        """
        :return: 网址对应的网页内容
        """

        http = urllib3.PoolManager()
        return http.request('GET', self.url, headers=self.headers).data


def getdata(ind, city, start, end):
    f = open('database.csv', 'a+', encoding='utf-8', newline='')  # 写CSV文件
    csv_writer = csv.writer(f)

    # 爬虫常规操作
    time.sleep(0.1)

    url = "http://www.meteomanz.com/sy2?l=1&cou=2250&ind=" + ind + "&d1=" + str(start.day).zfill(2) + "&m1=" + str(
        start.month).zfill(2) + "&y1=" + str(start.year) + "&d2=" + str(end.day).zfill(2) + "&m2=" + str(
        end.month).zfill(2) + "&y2=" + str(end.year)
    g = GetData(url).Get()

    # beautifulsoup解析网页
    soup = BeautifulSoup(g, "html5lib")

    # 取<tbody>内容
    tb = soup.find(name='tbody')

    # 取tr内容
    try:
        past_tr = tb.find_all(name="tr")
        for tr in past_tr:
            # 取tr内每个td的内容
            try:
                text = tr.find_all(name="td")
            except:
                break
            flag = False

            for i in range(0, len(text)):
                if i == 0:
                    text[i] = text[i].a.string
                    # 网站bug，会给每个月第0天，比如 00/11/2020,所以要drop掉
                    if "00/" in text[i]:
                        flag = True
                elif i == 8:
                    # 把/8去掉，网页显示的格式
                    text[i] = text[i].string.replace("/8", "")
                elif i == 5:
                    # 去掉单位
                    text[i] = text[i].string.replace(" Hpa", "")
                elif i == 6:
                    # 去掉风力里括号内的内容
                    text[i] = re.sub(u"[º(.*?|N|W|E|S)]", "", text[i].string)
                else:
                    # 取每个元素的内容
                    text[i] = text[i].string
                # 丢失数据都取2(简陋做法)
                # 这么做 MAE=3.6021
                text[i] = "2" if text[i] == "-" else text[i]
                text[i] = "2" if text[i] == "Tr" else text[i]
            text = text[0:9]
            # 4. 写入csv文件内容
            if not flag:
                csv_writer.writerow(text)
    except:
        pass

    f.close()
    data = pd.read_csv("database.csv")
    res = data.dropna(how="all")
    res.to_csv("{}.csv".format(city), index=False,
               header=['Date', 'Ave.T', 'Max.T', 'Min.T', 'Prec(mm)', 'Press(Hpa)', 'Wind dir', 'Wind sp (km/h)',
                       'Cloud.c'])
    os.remove('database.csv')


if __name__ == '__main__':
    # 设定爬取起止日期
    end = datetime.datetime.now()
    start = (end - datetime.timedelta(days=20)).date()
    ind = input('输入气象站编号')
    city = input('输入城市名称')
    # 执行程序
    getdata(ind, city, start, end)
