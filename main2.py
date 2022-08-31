import tkinter as tk
from tkinter import ttk  # 导入内部包
from PIL import ImageTk, Image
import handle

# 设置作业文件路径
fileDir = ""
# 设置临时文件路径
tempDir = ""

# 表头的标识
weatherDataItem = ['#0', 'aveT', 'maxT', 'minT',
                   'precmm', 'pressHpa', 'windDir', 'windSpKmH']
# 表头宽度
weatherDataWidth = [80, 50, 50, 50, 50, 50, 50, 50]
# 表头显示的文字
weatherDataHeader = ['date', 'Ave.T', 'Max.T', 'Min.T', 'Prec(mm)',
                     'Press(Hpa)', 'Wind dir', 'Wind sp(km/h)']


class basedesk:
    def __init__(self, master):
        self.root = master
        self.root.config()
        # 设置界面的题目
        self.root.title("天气预报系统")
        # 整个界面的大小
        self.root.geometry('480x500')
        interface1(self.root)


class interface1:
    def __init__(self, master):

        self.master = master
        # 定义一个canvas
        self.canvas1 = tk.Canvas(self.master, width=480, height=500)

        # 定义背景图片，用全局变量
        imgpath = 'picture/bg_1.jpg'
        global img
        img = Image.open(imgpath)
        global photo
        photo = ImageTk.PhotoImage(img)
        self.canvas1.create_image(240, 250, image=photo)

        self.label0 = tk.Label(
            self.master, text="欢迎使用天气预报系统", font=("仿宋", 25), fg="black")
        self.canvas1.create_window(
            240, 50, width=350, height=40, window=self.label0)

        self.label1 = tk.Label(self.master, text="请选择城市:", font=(
            "黑体", 22), fg="black", background="white")
        self.canvas1.create_window(
            150, 150, width=170, height=28, window=self.label1)

        self.optionsCity = ['北京', '重庆', '广州', '南京', '青岛', '三亚', '上海', '深圳', '武汉']
        self.stringVarCity = tk.StringVar()
        self.stringVarCity.set(self.optionsCity[0])
        self.optionMenuCity = tk.OptionMenu(
            self.master, self.stringVarCity, *self.optionsCity)
        self.canvas1.create_window(
            300, 150, width=100, height=28, window=self.optionMenuCity)

        self.label2 = tk.Label(self.master, text="请选择模型:", font=(
            "黑体", 22), fg="black", background="white")
        self.canvas1.create_window(
            150, 220, width=170, height=28, window=self.label2)

        self.optionsModel = ['LSTM', 'ARIMA']
        self.stringVarModel = tk.StringVar()
        self.stringVarModel.set(self.optionsModel[0])
        self.optionMenuModel = tk.OptionMenu(
            self.master, self.stringVarModel, *self.optionsModel)
        self.canvas1.create_window(
            300, 220, width=100, height=28, window=self.optionMenuModel)

        self.button1 = tk.Button(
            self.master, text='查询', command=self.search)
        # 前面两个是控件的中心位置，后面两个数是宽和高
        self.canvas1.create_window(
            120, 420, width=100, height=30, window=self.button1)

        self.button2 = tk.Button(
            self.master, text='退出', command=self.master.quit)
        # 前面两个是控件的中心位置，后面两个数是宽和高
        self.canvas1.create_window(
            360, 420, width=100, height=30, window=self.button2)

        self.canvas1.pack()

    def search(self):
        city = self.stringVarCity.get()
        model = self.stringVarModel.get()
        print("search方法获得的city:", city)
        print("search方法获得的model:", model)
        weatherDatas = handle.handleCityAndModel(city, model)
        print("得到weatherDatas：", weatherDatas)
        self.canvas1.destroy()
        interface2(self.master, weatherDatas, city, model)


class interface2:
    def __init__(self, master, weatherDatas, city, model):
        self.master = master
        self.label1 = tk.Label(master, text="您选择的城市：" +
                                            city + ",您选择的算法：" + model, width=200)
        self.label2 = tk.Label(
            master, text="今天的温度范围：" + weatherDatas[0][3] + "~" + weatherDatas[0][2], width=200)
        print("显示weatherDatas：", weatherDatas)
        self.tree = ttk.Treeview(self.master)  # 表格
        self.tree["columns"] = weatherDataItem[1:]
        for i in range(len(weatherDataItem)):
            self.tree.column(weatherDataItem[i], width=weatherDataWidth[i])
            self.tree.heading(weatherDataItem[i], text=weatherDataHeader[i])
        for i in range(len(weatherDatas)):
            self.tree.insert(
                "", i, text=weatherDatas[i][0], values=weatherDatas[i][1:])
        self.button1 = tk.Button(master, text='返回', command=self.previous)
        self.label1.pack()
        self.label2.pack()
        self.tree.pack()
        self.button1.pack()

    def previous(self):
        self.label1.destroy()
        self.label2.destroy()
        self.tree.destroy()
        self.button1.destroy()
        interface1(self.master)


root = tk.Tk()
basedesk(root)
root.mainloop()
