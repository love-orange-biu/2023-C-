import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


# ==========================================================
# 1. 全局绘图设置
# ==========================================================
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

sns.set_theme(
    style="whitegrid",
    font="Microsoft YaHei",
    rc={
        "axes.unicode_minus": False,
        "axes.facecolor": "#F7F8FB",
        "figure.facecolor": "white",
        "grid.color": "#D9D9D9",
        "grid.linestyle": "--",
        "grid.alpha": 0.35
    }
)


# ==========================================================
# 2. 读取数据
# ==========================================================
file_path = "C题_正确处理后建模数据.xlsx"

df = pd.read_excel(
    file_path,
    sheet_name="品类日数据"
)

df["日期"] = pd.to_datetime(df["日期"])

print("数据读取成功！")
print("数据量：", len(df))
print(
    "日期范围：",
    df["日期"].min(),
    "至",
    df["日期"].max()
)


# ==========================================================
# 3. 六大蔬菜品类
# ==========================================================
categories = [
    "花叶类",
    "花菜类",
    "水生根茎类",
    "茄类",
    "辣椒类",
    "食用菌"
]


# ==========================================================
# 4. 统一低饱和度配色
# ==========================================================
colors = [
    "#7FA6D8",   # 花叶类
    "#E7AE7A",   # 花菜类
    "#82BE98",   # 水生根茎类
    "#DB8D88",   # 茄类
    "#9C8BC4",   # 辣椒类
    "#AA8D72"    # 食用菌
]


# ==========================================================
# 5. 构造时间变量
# ==========================================================
df["年份"] = df["日期"].dt.year
df["月份"] = df["日期"].dt.month
df["星期"] = df["日期"].dt.dayofweek

weekday_map = {
    0: "星期一",
    1: "星期二",
    2: "星期三",
    3: "星期四",
    4: "星期五",
    5: "星期六",
    6: "星期日"
}

df["星期名称"] = df["星期"].map(weekday_map)

weekday_order = [
    "星期一",
    "星期二",
    "星期三",
    "星期四",
    "星期五",
    "星期六",
    "星期日"
]


# ==========================================================
# 6. 图3：六大品类三年销量时间趋势
# ==========================================================

fig = plt.figure(
    figsize=(15, 13.5),
    dpi=150,
    facecolor="white"
)

# ----------------------------------------------------------
# 单独建立标题区域
# ----------------------------------------------------------
title_ax = fig.add_axes([
    0.06,
    0.915,
    0.90,
    0.07
])

title_ax.axis("off")

# 主标题
title_ax.text(
    0.5,
    0.70,
    "六大蔬菜品类日销售量时间变化趋势",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    color="#222222"
)

# 副标题
title_ax.text(
    0.5,
    0.15,
    "浅色曲线表示每日销量，实线表示30日移动平均趋势",
    ha="center",
    va="center",
    fontsize=10.5,
    color="#666666"
)


# ----------------------------------------------------------
# 创建3×2子图区域
# ----------------------------------------------------------
gs = fig.add_gridspec(
    3,
    2,
    left=0.075,
    right=0.975,
    bottom=0.07,
    top=0.865,
    hspace=0.62,
    wspace=0.23
)

axes = []

for row in range(3):
    for col in range(2):
        axes.append(
            fig.add_subplot(
                gs[row, col]
            )
        )


# ----------------------------------------------------------
# 绘制六个品类
# ----------------------------------------------------------
for i, category in enumerate(categories):

    ax = axes[i]

    temp = df[
        df["分类名称"] == category
    ].copy()

    temp = temp.sort_values("日期")


    # 30日移动平均
    temp["30日移动平均"] = (
        temp["净销量(千克)"]
        .rolling(
            window=30,
            center=True,
            min_periods=1
        )
        .mean()
    )


    # 每日销量
    ax.plot(
        temp["日期"],
        temp["净销量(千克)"],
        color=colors[i],
        linewidth=0.6,
        alpha=0.20,
        label="日销量"
    )


    # 30日移动平均
    ax.plot(
        temp["日期"],
        temp["30日移动平均"],
        color=colors[i],
        linewidth=2.2,
        alpha=0.98,
        label="30日移动平均"
    )


    # 子图标题
    ax.set_title(
        category,
        fontsize=13,
        fontweight="bold",
        pad=10
    )


    # Y轴
    ax.set_ylabel(
        "销量（千克）",
        fontsize=10,
        labelpad=7
    )


    # 日期刻度，每6个月一个
    ax.xaxis.set_major_locator(
        mdates.MonthLocator(
            interval=6
        )
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%Y-%m"
        )
    )


    ax.tick_params(
        axis="x",
        labelrotation=0,
        labelsize=8.5,
        pad=5
    )

    ax.tick_params(
        axis="y",
        labelsize=9
    )


    # 只有最后一排显示“日期”
    if i >= 4:
        ax.set_xlabel(
            "日期",
            fontsize=10,
            labelpad=8
        )
    else:
        ax.set_xlabel("")


    # 网格
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.28
    )

    ax.grid(
        axis="x",
        visible=False
    )


    # 背景
    ax.set_facecolor(
        "#F7F8FB"
    )


    # 边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAAAAA")
    ax.spines["bottom"].set_color("#AAAAAA")


    # 图例
    ax.legend(
        fontsize=8,
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#DDDDDD"
    )


# ----------------------------------------------------------
# 保存图3
# ----------------------------------------------------------
plt.savefig(
    "问题1_图3_六大品类销量时间趋势.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()


# ==========================================================
# 7. 计算月份规律
# ==========================================================
monthly = (
    df
    .groupby(
        ["分类名称", "月份"]
    )["净销量(千克)"]
    .mean()
    .reset_index()
)


# ==========================================================
# 8. 图4：六大品类月平均销量
# ==========================================================
fig, ax = plt.subplots(
    figsize=(13.5, 6.5),
    dpi=150
)

for i, category in enumerate(categories):

    temp = monthly[
        monthly["分类名称"] == category
    ]

    ax.plot(
        temp["月份"],
        temp["净销量(千克)"],
        marker="o",
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=0.8,
        linewidth=2.2,
        color=colors[i],
        label=category
    )


ax.set_title(
    "六大蔬菜品类月平均销售量变化",
    fontsize=17,
    fontweight="bold",
    pad=18
)

ax.set_xlabel(
    "月份",
    fontsize=12,
    labelpad=10
)

ax.set_ylabel(
    "平均日销量（千克）",
    fontsize=12,
    labelpad=10
)

ax.set_xticks(
    range(1, 13)
)

ax.set_xticklabels(
    [
        "1月", "2月", "3月", "4月",
        "5月", "6月", "7月", "8月",
        "9月", "10月", "11月", "12月"
    ]
)

ax.legend(
    ncol=3,
    fontsize=9,
    frameon=True,
    framealpha=0.95,
    facecolor="white",
    edgecolor="#DDDDDD"
)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.65,
    alpha=0.30
)

ax.grid(
    axis="x",
    visible=False
)

ax.set_facecolor("#F7F8FB")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#AAAAAA")
ax.spines["bottom"].set_color("#AAAAAA")

plt.tight_layout()

plt.savefig(
    "问题1_图4_六大品类月平均销量.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()


# ==========================================================
# 9. 计算星期规律
# ==========================================================
weekly = (
    df
    .groupby(
        ["分类名称", "星期名称"]
    )["净销量(千克)"]
    .mean()
    .reset_index()
)

weekly["星期名称"] = pd.Categorical(
    weekly["星期名称"],
    categories=weekday_order,
    ordered=True
)

weekly = weekly.sort_values(
    [
        "分类名称",
        "星期名称"
    ]
)


# ==========================================================
# 10. 星期热力图数据
# ==========================================================
weekly_pivot = weekly.pivot(
    index="分类名称",
    columns="星期名称",
    values="净销量(千克)"
)

weekly_pivot = weekly_pivot.reindex(
    index=categories,
    columns=weekday_order
)


# ==========================================================
# 11. 自定义低饱和度热力图颜色
# ==========================================================
soft_heatmap_colors = [
    "#FAF3E8",
    "#F2EBC8",
    "#DDE7C7",
    "#BDDCCB",
    "#9CCDC7",
    "#82B8C8",
    "#6D9EC0",
    "#587EAA"
]

soft_cmap = LinearSegmentedColormap.from_list(
    "soft_pastel_heatmap",
    soft_heatmap_colors,
    N=256
)


# ==========================================================
# 12. 图5：星期销量热力图
# ==========================================================
fig, ax = plt.subplots(
    figsize=(12.5, 6),
    dpi=150
)

heatmap = sns.heatmap(
    weekly_pivot,
    cmap=soft_cmap,
    annot=True,
    fmt=".1f",
    annot_kws={
        "fontsize": 10.5
    },
    linewidths=1.2,
    linecolor="white",
    cbar_kws={
        "label": "平均日销量（千克）",
        "shrink": 0.88,
        "pad": 0.035
    },
    ax=ax
)


# ==========================================================
# 13. 自动调整热力图数字颜色
# ==========================================================
values = weekly_pivot.values

vmin = np.nanmin(values)
vmax = np.nanmax(values)

for text, value in zip(
    ax.texts,
    values.flatten()
):

    normalized = (
        (value - vmin)
        /
        (vmax - vmin)
    )

    if normalized > 0.62:

        text.set_color("white")
        text.set_fontweight("medium")

    else:

        text.set_color("#333333")


# ==========================================================
# 14. 热力图标题
# ==========================================================
ax.set_title(
    "六大蔬菜品类星期销售规律",
    fontsize=17,
    fontweight="bold",
    pad=22
)

ax.text(
    0.5,
    1.015,
    "颜色深浅表示各品类在不同星期的平均日销售水平",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=10,
    color="#666666"
)


# ==========================================================
# 15. 热力图坐标轴
# ==========================================================
ax.set_xlabel(
    "星期",
    fontsize=12,
    labelpad=10
)

ax.set_ylabel(
    "蔬菜品类",
    fontsize=12,
    labelpad=12
)

ax.set_xticklabels(
    weekday_order,
    rotation=0,
    fontsize=10.5
)

ax.set_yticklabels(
    categories,
    rotation=0,
    fontsize=10.5
)

ax.tick_params(
    axis="both",
    length=0
)


# ==========================================================
# 16. 热力图颜色条
# ==========================================================
cbar = heatmap.collections[0].colorbar

cbar.ax.tick_params(
    labelsize=9
)

cbar.ax.set_ylabel(
    "平均日销量（千克）",
    fontsize=10,
    labelpad=12
)

fig.patch.set_facecolor("white")

plt.tight_layout()

plt.savefig(
    "问题1_图5_六大品类星期销量热力图_优化版.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()


# ==========================================================
# 17. 月份统计表
# ==========================================================
monthly_table = monthly.pivot(
    index="月份",
    columns="分类名称",
    values="净销量(千克)"
)

monthly_table = monthly_table.reindex(
    columns=categories
)

monthly_table = monthly_table.round(4)


# ==========================================================
# 18. 星期统计表
# ==========================================================
weekly_table = weekly_pivot.round(4)


# ==========================================================
# 19. 工作日与周末比较
# ==========================================================
df["日期类型"] = np.where(
    df["星期"].isin([5, 6]),
    "周末",
    "工作日"
)

weekend_result = (
    df
    .groupby(
        [
            "分类名称",
            "日期类型"
        ]
    )["净销量(千克)"]
    .mean()
    .unstack()
)

weekend_result = weekend_result.reindex(
    categories
)

weekend_result["周末提升比例(%)"] = (
    (
        weekend_result["周末"]
        /
        weekend_result["工作日"]
        - 1
    )
    * 100
)

weekend_result = weekend_result.round(4)


# ==========================================================
# 20. 保存统计结果
# ==========================================================
with pd.ExcelWriter(
    "问题1_时间规律统计结果.xlsx",
    engine="openpyxl"
) as writer:

    monthly_table.to_excel(
        writer,
        sheet_name="月平均销量"
    )

    weekly_table.to_excel(
        writer,
        sheet_name="星期平均销量"
    )

    weekend_result.to_excel(
        writer,
        sheet_name="工作日周末比较"
    )


# ==========================================================
# 21. 输出月平均销量
# ==========================================================
print("\n========================================")
print("月平均销量")
print("========================================")
print(monthly_table)


# ==========================================================
# 22. 输出星期平均销量
# ==========================================================
print("\n========================================")
print("星期平均销量")
print("========================================")
print(weekly_table)


# ==========================================================
# 23. 各品类最高销量月份
# ==========================================================
print("\n========================================")
print("各品类平均销量最高月份")
print("========================================")

for category in categories:

    temp = monthly[
        monthly["分类名称"] == category
    ]

    max_row = temp.loc[
        temp["净销量(千克)"].idxmax()
    ]

    print(
        f"{category}："
        f"{int(max_row['月份'])}月，"
        f"平均销量 "
        f"{max_row['净销量(千克)']:.2f} 千克"
    )


# ==========================================================
# 24. 各品类最低销量月份
# ==========================================================
print("\n========================================")
print("各品类平均销量最低月份")
print("========================================")

for category in categories:

    temp = monthly[
        monthly["分类名称"] == category
    ]

    min_row = temp.loc[
        temp["净销量(千克)"].idxmin()
    ]

    print(
        f"{category}："
        f"{int(min_row['月份'])}月，"
        f"平均销量 "
        f"{min_row['净销量(千克)']:.2f} 千克"
    )


# ==========================================================
# 25. 各品类最高销量星期
# ==========================================================
print("\n========================================")
print("各品类平均销量最高星期")
print("========================================")

for category in categories:

    temp = weekly[
        weekly["分类名称"] == category
    ]

    max_row = temp.loc[
        temp["净销量(千克)"].idxmax()
    ]

    print(
        f"{category}："
        f"{max_row['星期名称']}，"
        f"平均销量 "
        f"{max_row['净销量(千克)']:.2f} 千克"
    )


# ==========================================================
# 26. 各品类最低销量星期
# ==========================================================
print("\n========================================")
print("各品类平均销量最低星期")
print("========================================")

for category in categories:

    temp = weekly[
        weekly["分类名称"] == category
    ]

    min_row = temp.loc[
        temp["净销量(千克)"].idxmin()
    ]

    print(
        f"{category}："
        f"{min_row['星期名称']}，"
        f"平均销量 "
        f"{min_row['净销量(千克)']:.2f} 千克"
    )


# ==========================================================
# 27. 工作日与周末比较
# ==========================================================
print("\n========================================")
print("工作日与周末销量比较")
print("========================================")

print(
    weekend_result[
        [
            "工作日",
            "周末",
            "周末提升比例(%)"
        ]
    ]
)


# ==========================================================
# 28. 完成提示
# ==========================================================
print("\n========================================")
print("第三步运行完成！")
print("========================================")

print("\n已生成以下文件：")
print("1. 问题1_图3_六大品类销量时间趋势.png")
print("2. 问题1_图4_六大品类月平均销量.png")
print("3. 问题1_图5_六大品类星期销量热力图_优化版.png")
print("4. 问题1_时间规律统计结果.xlsx")