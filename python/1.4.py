import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


# ==========================================================
# 1. 全局设置
# ==========================================================

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


sns.set_theme(
    style="white",
    font="Microsoft YaHei"
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


# ==========================================================
# 3. 六大品类顺序
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
# 4. 构造“日期 × 品类”的销量矩阵
#
# 每一行：某一天
# 每一列：一个蔬菜品类
# 单元格：当天该品类净销量
# ==========================================================

sales_matrix = df.pivot_table(
    index="日期",
    columns="分类名称",
    values="净销量(千克)",
    aggfunc="sum"
)


# 调整列顺序
sales_matrix = sales_matrix.reindex(
    columns=categories
)


# ==========================================================
# 5. 检查缺失值
# ==========================================================

print("\n==============================")
print("各品类缺失值数量")
print("==============================")

print(
    sales_matrix.isnull().sum()
)


# ==========================================================
# 6. 缺失值处理
#
# 某天某品类没有记录时，
# 不直接视为0，而是保留NaN。
#
# pandas在计算相关系数时会自动使用
# 两个品类同时存在数据的日期。
# ==========================================================

print("\n销量矩阵大小：")
print(sales_matrix.shape)

print("\n销量矩阵前5行：")
print(sales_matrix.head())


# ==========================================================
# 7. Pearson相关系数
# ==========================================================

pearson_corr = sales_matrix.corr(
    method="pearson"
)


print("\n==============================")
print("Pearson相关系数矩阵")
print("==============================")

print(
    pearson_corr.round(4)
)


# ==========================================================
# 8. Spearman相关系数
# ==========================================================

spearman_corr = sales_matrix.corr(
    method="spearman"
)


print("\n==============================")
print("Spearman相关系数矩阵")
print("==============================")

print(
    spearman_corr.round(4)
)


# ==========================================================
# 9. 创建柔和低饱和度相关性配色
#
# 负相关：柔和蓝
# 0附近：浅灰白
# 正相关：柔和红
# ==========================================================

corr_colors = [
    "#7696B8",
    "#AFC4D6",
    "#D8E2E8",
    "#F5F3EF",
    "#EAD8D3",
    "#D9AAA2",
    "#C77C73"
]

corr_cmap = LinearSegmentedColormap.from_list(
    "soft_corr",
    corr_colors,
    N=256
)


# ==========================================================
# 10. 图6：Pearson相关系数热力图
# ==========================================================

fig, ax = plt.subplots(
    figsize=(9.5, 7.5),
    dpi=150
)


mask = np.triu(
    np.ones_like(
        pearson_corr,
        dtype=bool
    ),
    k=1
)


heatmap = sns.heatmap(
    pearson_corr,

    mask=mask,

    cmap=corr_cmap,

    vmin=-1,
    vmax=1,
    center=0,

    annot=True,
    fmt=".3f",

    annot_kws={
        "fontsize": 11
    },

    square=True,

    linewidths=1.5,
    linecolor="white",

    cbar_kws={
        "label": "Pearson相关系数",
        "shrink": 0.82,
        "pad": 0.04
    },

    ax=ax
)


ax.set_title(
    "六大蔬菜品类日销量 Pearson 相关性",
    fontsize=17,
    fontweight="bold",
    pad=22
)


ax.text(
    0.5,
    1.015,
    "相关系数越接近1表示正相关越强，越接近-1表示负相关越强",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=9.5,
    color="#666666"
)


ax.set_xlabel("")
ax.set_ylabel("")


ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=0,
    fontsize=10
)

ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=0,
    fontsize=10
)


plt.tight_layout()


plt.savefig(
    "问题1_图6_Pearson相关系数热力图.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)


plt.show()


# ==========================================================
# 11. 图7：Spearman相关系数热力图
# ==========================================================

fig, ax = plt.subplots(
    figsize=(9.5, 7.5),
    dpi=150
)


mask = np.triu(
    np.ones_like(
        spearman_corr,
        dtype=bool
    ),
    k=1
)


heatmap = sns.heatmap(
    spearman_corr,

    mask=mask,

    cmap=corr_cmap,

    vmin=-1,
    vmax=1,
    center=0,

    annot=True,
    fmt=".3f",

    annot_kws={
        "fontsize": 11
    },

    square=True,

    linewidths=1.5,
    linecolor="white",

    cbar_kws={
        "label": "Spearman相关系数",
        "shrink": 0.82,
        "pad": 0.04
    },

    ax=ax
)


ax.set_title(
    "六大蔬菜品类日销量 Spearman 相关性",
    fontsize=17,
    fontweight="bold",
    pad=22
)


ax.text(
    0.5,
    1.015,
    "Spearman秩相关对异常值和非正态分布具有更好的稳健性",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=9.5,
    color="#666666"
)


ax.set_xlabel("")
ax.set_ylabel("")


ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=0,
    fontsize=10
)

ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=0,
    fontsize=10
)


plt.tight_layout()


plt.savefig(
    "问题1_图7_Spearman相关系数热力图.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)


plt.show()


# ==========================================================
# 12. 图8：六大品类销量散点矩阵
#
# 为避免极端销量使主体区域被严重压缩，
# 绘图时对每个品类仅截取99%分位数范围。
#
# 注意：
# 这里只影响图形展示，
# 不影响上面的相关系数计算。
# ==========================================================

plot_data = sales_matrix.copy()


for category in categories:

    upper = plot_data[
        category
    ].quantile(0.99)

    plot_data.loc[
        plot_data[category] > upper,
        category
    ] = np.nan


# 删除全为空的行
plot_data = plot_data.dropna(
    how="all"
)


# ----------------------------------------------------------
# 为散点矩阵设置颜色
# ----------------------------------------------------------

pair_colors = {
    "花叶类": "#7FA6D8",
    "花菜类": "#E7AE7A",
    "水生根茎类": "#82BE98",
    "茄类": "#DB8D88",
    "辣椒类": "#9C8BC4",
    "食用菌": "#AA8D72"
}


# ----------------------------------------------------------
# 创建PairGrid
# ----------------------------------------------------------

g = sns.PairGrid(
    plot_data[categories],
    diag_sharey=False,
    height=2.1
)


# ----------------------------------------------------------
# 下三角：散点图
# ----------------------------------------------------------

g.map_lower(
    sns.scatterplot,
    s=13,
    alpha=0.28,
    color="#718096",
    edgecolor=None
)


# ----------------------------------------------------------
# 对角线：直方图 + KDE
# ----------------------------------------------------------

g.map_diag(
    sns.histplot,
    kde=True,
    bins=25,
    color="#9DB7D5",
    alpha=0.55,
    edgecolor="white"
)


# ----------------------------------------------------------
# 上三角：隐藏
# ----------------------------------------------------------

for i in range(len(categories)):

    for j in range(len(categories)):

        if j > i:

            g.axes[i, j].set_visible(
                False
            )


# ----------------------------------------------------------
# 美化坐标轴
# ----------------------------------------------------------

for ax_row in g.axes:

    for ax in ax_row:

        if ax is None:
            continue

        ax.set_facecolor(
            "#F7F8FB"
        )

        ax.grid(
            linestyle="--",
            linewidth=0.5,
            alpha=0.20
        )


# ----------------------------------------------------------
# 总标题
# ----------------------------------------------------------

g.fig.subplots_adjust(
    top=0.94,
    left=0.08,
    right=0.98,
    bottom=0.08,
    hspace=0.08,
    wspace=0.08
)


g.fig.suptitle(
    "六大蔬菜品类日销量两两关系散点矩阵",
    fontsize=18,
    fontweight="bold",
    y=0.985
)


g.fig.text(
    0.5,
    0.955,
    "散点分布用于辅助判断不同蔬菜品类销量之间的关联关系",
    ha="center",
    va="center",
    fontsize=10,
    color="#666666"
)


g.fig.savefig(
    "问题1_图8_六大品类销量散点矩阵.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)


plt.show()


# ==========================================================
# 13. 提取所有品类组合
# ==========================================================

relationship_list = []


for i in range(
    len(categories)
):

    for j in range(
        i + 1,
        len(categories)
    ):

        category1 = categories[i]
        category2 = categories[j]

        pearson_value = pearson_corr.loc[
            category1,
            category2
        ]

        spearman_value = spearman_corr.loc[
            category1,
            category2
        ]


        relationship_list.append(
            {
                "品类1": category1,
                "品类2": category2,
                "Pearson相关系数": pearson_value,
                "Spearman相关系数": spearman_value
            }
        )


relationship_df = pd.DataFrame(
    relationship_list
)


# ==========================================================
# 14. 根据Spearman绝对值划分相关程度
# ==========================================================

def classify_correlation(r):

    abs_r = abs(r)

    if abs_r >= 0.8:
        strength = "强相关"

    elif abs_r >= 0.5:
        strength = "中等相关"

    elif abs_r >= 0.3:
        strength = "弱相关"

    else:
        strength = "相关性很弱"

    if r > 0:
        direction = "正"

    elif r < 0:
        direction = "负"

    else:
        direction = ""

    return direction + strength


relationship_df["相关程度"] = (
    relationship_df[
        "Spearman相关系数"
    ]
    .apply(
        classify_correlation
    )
)


# ==========================================================
# 15. 按Spearman绝对值排序
# ==========================================================

relationship_df[
    "Spearman绝对值"
] = (
    relationship_df[
        "Spearman相关系数"
    ]
    .abs()
)


relationship_df = relationship_df.sort_values(
    "Spearman绝对值",
    ascending=False
)


relationship_df[
    "Pearson相关系数"
] = (
    relationship_df[
        "Pearson相关系数"
    ]
    .round(4)
)


relationship_df[
    "Spearman相关系数"
] = (
    relationship_df[
        "Spearman相关系数"
    ]
    .round(4)
)


relationship_df[
    "Spearman绝对值"
] = (
    relationship_df[
        "Spearman绝对值"
    ]
    .round(4)
)


# ==========================================================
# 16. 保存Excel结果
# ==========================================================

with pd.ExcelWriter(
    "问题1_品类相关性分析结果.xlsx",
    engine="openpyxl"
) as writer:

    sales_matrix.to_excel(
        writer,
        sheet_name="日销量矩阵"
    )

    pearson_corr.round(4).to_excel(
        writer,
        sheet_name="Pearson相关系数"
    )

    spearman_corr.round(4).to_excel(
        writer,
        sheet_name="Spearman相关系数"
    )

    relationship_df.to_excel(
        writer,
        sheet_name="品类两两相关性",
        index=False
    )


# ==========================================================
# 17. 输出相关性排序
# ==========================================================

print(
    "\n========================================"
)

print(
    "六大品类两两相关性排序"
)

print(
    "========================================"
)


print(
    relationship_df[
        [
            "品类1",
            "品类2",
            "Pearson相关系数",
            "Spearman相关系数",
            "相关程度"
        ]
    ].to_string(
        index=False
    )
)


# ==========================================================
# 18. 输出最强的5组关系
# ==========================================================

print(
    "\n========================================"
)

print(
    "相关性最强的5组品类"
)

print(
    "========================================"
)


top5 = relationship_df.head(5)


for _, row in top5.iterrows():

    print(
        f"{row['品类1']} - "
        f"{row['品类2']}："
        f"Pearson={row['Pearson相关系数']:.4f}，"
        f"Spearman={row['Spearman相关系数']:.4f}，"
        f"{row['相关程度']}"
    )


# ==========================================================
# 19. 完成提示
# ==========================================================

print(
    "\n========================================"
)

print(
    "第四步运行完成！"
)

print(
    "========================================"
)


print(
    "\n已生成："
)

print(
    "1. 问题1_图6_Pearson相关系数热力图.png"
)

print(
    "2. 问题1_图7_Spearman相关系数热力图.png"
)

print(
    "3. 问题1_图8_六大品类销量散点矩阵.png"
)

print(
    "4. 问题1_品类相关性分析结果.xlsx"
)