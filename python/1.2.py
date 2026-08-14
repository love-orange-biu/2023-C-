import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ==========================================================
# 1. 全局绘图风格
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
        "grid.alpha": 0.4
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

print("数据读取成功！")
print("数据量：", len(df))


# ==========================================================
# 3. 六大品类
# ==========================================================
categories = [
    "花叶类",
    "花菜类",
    "水生根茎类",
    "茄类",
    "辣椒类",
    "食用菌"
]


# KDE曲线颜色
colors = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#937860"
]


# 小提琴图柔和配色
violin_colors = [
    "#A9C4EB",
    "#F2C49B",
    "#A8D8B9",
    "#EFA7A0",
    "#C7B3E5",
    "#D7C3B2"
]


# ==========================================================
# 4. 数据筛选
# ==========================================================
plot_df = df[
    df["分类名称"].isin(categories)
][
    ["分类名称", "净销量(千克)"]
].dropna()


# ==========================================================
# 5. 图1：
# 优化版小提琴图 + 箱线图 + 均值点
# ==========================================================

fig, ax = plt.subplots(
    figsize=(13.5, 6.2),
    dpi=150
)


# ----------------------------------------------------------
# 5.1 计算99%分位数
# 仅用于控制图片显示范围
# 原始数据不删除
# ----------------------------------------------------------
upper_limit = (
    plot_df["净销量(千克)"]
    .quantile(0.99)
)

upper_limit = upper_limit * 1.12


# ----------------------------------------------------------
# 5.2 绘制小提琴图
# ----------------------------------------------------------
sns.violinplot(
    data=plot_df,

    x="分类名称",
    y="净销量(千克)",

    hue="分类名称",

    order=categories,
    hue_order=categories,

    palette=violin_colors,

    inner=None,

    # 不超出原始数据范围
    cut=0,

    # 统一宽度便于比较
    density_norm="width",

    # KDE平滑程度
    bw_adjust=0.75,

    linewidth=1.0,

    saturation=0.85,

    legend=False,

    ax=ax
)


# ----------------------------------------------------------
# 5.3 设置透明度
# ----------------------------------------------------------
for collection in ax.collections:

    try:
        collection.set_alpha(0.58)

    except Exception:
        pass


# ----------------------------------------------------------
# 5.4 内嵌箱线图
# ----------------------------------------------------------
sns.boxplot(
    data=plot_df,

    x="分类名称",
    y="净销量(千克)",

    order=categories,

    width=0.105,

    showfliers=False,

    boxprops={
        "facecolor": "#FFFFFF",
        "edgecolor": "#555555",
        "linewidth": 1.25,
        "alpha": 0.90
    },

    medianprops={
        "color": "#333333",
        "linewidth": 2.0
    },

    whiskerprops={
        "color": "#666666",
        "linewidth": 1.2
    },

    capprops={
        "color": "#666666",
        "linewidth": 1.2
    },

    ax=ax
)


# ----------------------------------------------------------
# 5.5 计算均值
# ----------------------------------------------------------
means = (
    plot_df
    .groupby("分类名称")["净销量(千克)"]
    .mean()
    .reindex(categories)
)


# ----------------------------------------------------------
# 5.6 添加均值点
# ----------------------------------------------------------
for i, category in enumerate(categories):

    ax.scatter(
        i,

        means[category],

        s=65,

        color="#D64B40",

        edgecolor="white",

        linewidth=1.1,

        zorder=10
    )


# ----------------------------------------------------------
# 5.7 标注均值
# ----------------------------------------------------------
for i, category in enumerate(categories):

    ax.annotate(
        f"{means[category]:.1f}",

        xy=(
            i,
            means[category]
        ),

        xytext=(
            8,
            5
        ),

        textcoords="offset points",

        fontsize=9,

        color="#A33A32",

        fontweight="bold"
    )


# ----------------------------------------------------------
# 5.8 设置纵轴显示范围
# ----------------------------------------------------------
ax.set_ylim(
    0,
    upper_limit
)


# ----------------------------------------------------------
# 5.9 标题
# ----------------------------------------------------------
ax.set_title(
    "六大蔬菜品类日销售量分布特征",

    fontsize=17,

    fontweight="bold",

    pad=20
)


# 副标题
ax.text(
    0.5,
    1.015,

    "小提琴宽度表示数据密度，箱体表示四分位区间，红点表示均值",

    transform=ax.transAxes,

    ha="center",

    va="bottom",

    fontsize=10.5,

    color="#666666"
)


# ----------------------------------------------------------
# 5.10 坐标轴
# ----------------------------------------------------------
ax.set_xlabel(
    "蔬菜品类",
    fontsize=12,
    labelpad=12
)

ax.set_ylabel(
    "日销售量（千克）",
    fontsize=12,
    labelpad=10
)

ax.tick_params(
    axis="x",
    labelsize=11,
    pad=6
)

ax.tick_params(
    axis="y",
    labelsize=10
)


# ----------------------------------------------------------
# 5.11 网格
# ----------------------------------------------------------
ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.65,
    color="#CFCFCF",
    alpha=0.45
)

ax.grid(
    axis="x",
    visible=False
)


# ----------------------------------------------------------
# 5.12 背景
# ----------------------------------------------------------
ax.set_facecolor("#F7F8FB")
fig.patch.set_facecolor("white")


# ----------------------------------------------------------
# 5.13 去除多余边框
# ----------------------------------------------------------
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_color("#999999")
ax.spines["bottom"].set_color("#999999")


# ----------------------------------------------------------
# 5.14 图例
# ----------------------------------------------------------
legend_elements = [

    Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor="#D64B40",
        markeredgecolor="white",
        markersize=8,
        label="平均销量"
    ),

    Patch(
        facecolor="white",
        edgecolor="#555555",
        label="四分位区间"
    ),

    Patch(
        facecolor="#A9C4EB",
        edgecolor="#999999",
        alpha=0.58,
        label="销量概率密度"
    )
]

ax.legend(
    handles=legend_elements,
    loc="upper right",
    frameon=True,
    framealpha=0.95,
    facecolor="white",
    edgecolor="#DDDDDD",
    fontsize=9.5
)


# ==========================================================
# 关键修改：手动给底部留出足够空间
# ==========================================================

# 不再使用 tight_layout()
# 直接手动控制上下左右留白
fig.subplots_adjust(
    left=0.08,
    right=0.97,
    top=0.86,
    bottom=0.25
)


# ----------------------------------------------------------
# 5.15 图下注释
# ----------------------------------------------------------
fig.text(
    0.08,
    0.055,

    "注：为提高主体分布的可读性，纵轴展示至全部日销量的99%分位数；"
    "极端销售记录仍保留在全部统计计算中。",

    fontsize=9.5,
    color="#777777",

    ha="left",
    va="bottom"
)


# ----------------------------------------------------------
# 5.16 保存
# ----------------------------------------------------------
plt.savefig(
    "问题1_图1_六大品类日销量小提琴图_优化版.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

# ==========================================================
# 6. 图2：
# 六大品类直方图 + KDE
# ==========================================================

fig, axes = plt.subplots(
    2,
    3,

    figsize=(15, 8.5)
)

axes = axes.flatten()


for i, category in enumerate(categories):

    ax = axes[i]


    data = df.loc[
        df["分类名称"] == category,
        "净销量(千克)"
    ].dropna()


    color = colors[i]


    # ------------------------------------------------------
    # 6.1 直方图
    # ------------------------------------------------------
    ax.hist(
        data,

        bins=30,

        density=True,

        color=color,

        alpha=0.38,

        edgecolor="white",

        linewidth=0.8
    )


    # ------------------------------------------------------
    # 6.2 KDE核密度曲线
    # ------------------------------------------------------
    kde = gaussian_kde(
        data
    )


    x = np.linspace(
        data.min(),

        data.max(),

        400
    )


    ax.plot(
        x,

        kde(x),

        color=color,

        linewidth=2.2
    )


    # ------------------------------------------------------
    # 6.3 均值与中位数
    # ------------------------------------------------------
    mean_value = data.mean()

    median_value = data.median()


    ax.axvline(
        mean_value,

        color=color,

        linestyle="--",

        linewidth=1.6,

        label=f"均值={mean_value:.2f}"
    )


    ax.axvline(
        median_value,

        color=color,

        linestyle=":",

        linewidth=1.8,

        label=f"中位数={median_value:.2f}"
    )


    # ------------------------------------------------------
    # 6.4 子图标题
    # ------------------------------------------------------
    ax.set_title(
        category,

        fontsize=13,

        fontweight="bold",

        pad=8
    )


    # ------------------------------------------------------
    # 6.5 坐标轴
    # ------------------------------------------------------
    ax.set_xlabel(
        "日销售量（千克）",

        fontsize=10
    )


    ax.set_ylabel(
        "概率密度",

        fontsize=10
    )


    # ------------------------------------------------------
    # 6.6 图例
    # ------------------------------------------------------
    ax.legend(
        fontsize=8,

        frameon=True,

        facecolor="white",

        edgecolor="#DDDDDD"
    )


    # ------------------------------------------------------
    # 6.7 网格
    # ------------------------------------------------------
    ax.grid(
        axis="y",

        linestyle="--",

        alpha=0.35
    )


    ax.grid(
        axis="x",

        visible=False
    )


    # ------------------------------------------------------
    # 6.8 去掉右上边框
    # ------------------------------------------------------
    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )


    ax.set_facecolor(
        "#F7F8FB"
    )


# ----------------------------------------------------------
# 6.9 总标题
# ----------------------------------------------------------
fig.suptitle(
    "六大蔬菜品类日销售量分布",

    fontsize=17,

    fontweight="bold",

    y=0.99
)


plt.tight_layout(
    rect=[0, 0, 1, 0.96]
)


plt.savefig(
    "问题1_图2_六大品类日销量分布图.png",

    dpi=300,

    bbox_inches="tight",

    facecolor="white"
)


plt.show()


# ==========================================================
# 7. 计算销量分布统计指标
# ==========================================================

distribution_result = []


for category in categories:

    data = df.loc[
        df["分类名称"] == category,
        "净销量(千克)"
    ].dropna()


    distribution_result.append(
        {
            "分类名称":
                category,

            "样本数":
                len(data),

            "均值":
                data.mean(),

            "中位数":
                data.median(),

            "标准差":
                data.std(),

            "最小值":
                data.min(),

            "最大值":
                data.max(),

            "变异系数CV":
                data.std() / data.mean(),

            "偏度":
                data.skew(),

            "峰度":
                data.kurt()
        }
    )


distribution_result = pd.DataFrame(
    distribution_result
)


distribution_result = (
    distribution_result
    .round(4)
)


# ==========================================================
# 8. 输出统计结果
# ==========================================================

print(
    "\n各品类销量分布统计结果："
)

print(
    distribution_result
)


# ==========================================================
# 9. 保存Excel
# ==========================================================

distribution_result.to_excel(
    "问题1_六大品类销量分布统计.xlsx",

    index=False
)


# ==========================================================
# 10. 控制台输出初步结论
# ==========================================================

print(
    "\n------------------------------"
)

print(
    "各品类偏度："
)

for _, row in distribution_result.iterrows():

    print(
        f"{row['分类名称']}："
        f"{row['偏度']}"
    )


print(
    "\n各品类变异系数CV："
)

for _, row in distribution_result.iterrows():

    print(
        f"{row['分类名称']}："
        f"{row['变异系数CV']}"
    )


# ==========================================================
# 11. 完成提示
# ==========================================================

print(
    "\n第二步运行完成！"
)

print(
    "\n已生成："
)

print(
    "1. 问题1_图1_六大品类日销量小提琴图_优化版.png"
)

print(
    "2. 问题1_图2_六大品类日销量分布图.png"
)

print(
    "3. 问题1_六大品类销量分布统计.xlsx"
)