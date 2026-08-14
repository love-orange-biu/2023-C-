import pandas as pd

# 读取处理后的数据
file_path = "C题_正确处理后建模数据.xlsx"

df = pd.read_excel(
    file_path,
    sheet_name="品类日数据"
)

# 按品类统计净销量
result = df.groupby("分类名称")["净销量(千克)"].agg(
    样本数="count",
    总销量="sum",
    平均值="mean",
    中位数="median",
    标准差="std",
    最小值="min",
    最大值="max"
)

# 计算变异系数
result["变异系数CV"] = result["标准差"] / result["平均值"]

# 保留4位小数
result = result.round(4)

print(result)

# 保存
result.to_excel("问题1_六大品类描述性统计.xlsx")