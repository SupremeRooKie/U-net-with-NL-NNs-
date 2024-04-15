import csv
import matplotlib.pyplot as plt

file_path = "D:\BaiduNetdiskDownload\unet++\models\dsb2018_96_NestedUNet_woDS\log.csv"

with open(file_path, "r") as f:         # 打开csv文件
    reader = csv.reader(f)              # 读取csv文件
    list_data = list(reader)            # csv数据转换为列表
    f.close()

rows = len(list_data)                   # 行数
cols = len(list_data[0])                # 列数
print("rows =", rows)
print("cols =", cols)

col_0 = []
col_1 = []
col_2 = []

if (cols > 2):
    for i in range(0, rows):
        col_2.append(list_data[i][2])
        col_1.append(list_data[i][1])
        col_0.append(list_data[i][0])
elif (cols > 1):
    for i in range(0, rows):
        col_1.append(list_data[i][1])
        col_0.append(list_data[i][0])
else:
    for i in range(0, rows):
        col_0.append(list_data[i][0])

data_col_0=[int(x) for x in col_0]      # 0列数据
data_col_1=[int(x) for x in col_1]      # 1列数据
data_col_2=[int(x) for x in col_2]      # 2列数据