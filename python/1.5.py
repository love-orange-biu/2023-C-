import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
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
        "axes.facecolor": "#F8F9FB",
        "figure.facecolor": "white",
        "grid.color": "#D9D9D9",
        "grid.linestyle": "--",
        "grid.alpha": 0.25
    }
)


# ==========================================================
# 2. 参数设置
# ==========================================================

file_path = "C题_正确处理后建模数据.xlsx"

# 有效单品至少销售多少天
MIN_SALE_DAYS = 60

# 每个品类用于相关性分析的代表性单品数
TOP_N_PER_CATEGORY = 5

# 两个单品至少共同销售多少天才计算相关性
MIN_COMMON_DAYS = 30

# 累计销量图每个品类显示多少个单品
TOP_SHOW = 3


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
# 4. 读取单品日数据
# ==========================================================

df = pd.read_excel(
    file_path,
    sheet_name="单品日数据"
)

df["日期"] = pd.to_datetime(
    df["日期"]
)

df["单品编码"] = (
    df["单品编码"]
    .astype(str)
)

print("数据读取成功！")
print("数据量：", len(df))
print(
    "日期范围：",
    df["日期"].min(),
    "至",
    df["日期"].max()
)


# ==========================================================
# 5. 单品描述性统计
# ==========================================================

item_stats = (
    df
    .groupby(
        [
            "单品编码",
            "单品名称",
            "分类名称"
        ]
    )
    .agg(
        销售天数=(
            "日期",
            "nunique"
        ),

        总销量=(
            "净销量(千克)",
            "sum"
        ),

        日均销量=(
            "净销量(千克)",
            "mean"
        ),

        销量中位数=(
            "净销量(千克)",
            "median"
        ),

        销量标准差=(
            "净销量(千克)",
            "std"
        ),

        最大日销量=(
            "净销量(千克)",
            "max"
        )
    )
    .reset_index()
)


# ==========================================================
# 6. 计算变异系数
# ==========================================================

item_stats["变异系数CV"] = (
    item_stats["销量标准差"]
    /
    item_stats["日均销量"]
)

item_stats = item_stats.round(4)


# ==========================================================
# 7. 有效单品筛选
# ==========================================================

valid_items = item_stats[
    item_stats["销售天数"]
    >= MIN_SALE_DAYS
].copy()

valid_items = valid_items.sort_values(
    "总销量",
    ascending=False
)


print("\n====================================")
print("全部单品数量")
print("====================================")

print(
    item_stats[
        "单品编码"
    ].nunique()
)


print("\n====================================")
print(
    f"销售天数不少于 "
    f"{MIN_SALE_DAYS} 天的有效单品数量"
)
print("====================================")

print(
    valid_items[
        "单品编码"
    ].nunique()
)


# ==========================================================
# 8. 每个品类选累计销量最高的5个代表性单品
# ==========================================================

representative_items = (
    valid_items
    .sort_values(
        [
            "分类名称",
            "总销量"
        ],
        ascending=[
            True,
            False
        ]
    )
    .groupby(
        "分类名称",
        group_keys=False
    )
    .head(
        TOP_N_PER_CATEGORY
    )
    .copy()
)


print("\n====================================")
print("各品类代表性单品")
print("====================================")

print(
    representative_items[
        [
            "单品编码",
            "单品名称",
            "分类名称",
            "销售天数",
            "总销量",
            "日均销量",
            "变异系数CV"
        ]
    ]
    .to_string(
        index=False
    )
)


# ==========================================================
# 9. 提取代表性单品
# ==========================================================

representative_codes = (
    representative_items[
        "单品编码"
    ]
    .astype(str)
    .tolist()
)

rep_df = df[
    df["单品编码"]
    .isin(
        representative_codes
    )
].copy()


# ==========================================================
# 10. 构造 日期 × 单品 销量矩阵
#
# 注意：
# 缺失值保留 NaN
# 不再填 0
# ==========================================================

item_matrix = rep_df.pivot_table(
    index="日期",
    columns="单品名称",
    values="净销量(千克)",
    aggfunc="sum"
)


print("\n====================================")
print("代表单品日销量矩阵大小")
print("====================================")

print(
    item_matrix.shape
)


# ==========================================================
# 11. 初始化 Spearman相关系数矩阵
# ==========================================================

items = item_matrix.columns.tolist()

spearman_corr = pd.DataFrame(
    np.nan,
    index=items,
    columns=items
)

common_days_matrix = pd.DataFrame(
    0,
    index=items,
    columns=items,
    dtype=int
)


# ==========================================================
# 12. 两两计算 Spearman 相关系数
#
# 只使用两个单品同时有销售记录的日期
# ==========================================================

for i, item1 in enumerate(items):

    for j, item2 in enumerate(items):

        # ----------------------------------------------
        # 对角线：自己与自己
        # ----------------------------------------------
        if item1 == item2:

            spearman_corr.loc[
                item1,
                item2
            ] = 1.0

            common_days_matrix.loc[
                item1,
                item2
            ] = (
                item_matrix[
                    item1
                ]
                .notna()
                .sum()
            )

            continue


        # ----------------------------------------------
        # 两个单品共同有销售记录的日期
        # ----------------------------------------------
        pair_data = (
            item_matrix[
                [
                    item1,
                    item2
                ]
            ]
            .dropna()
        )

        common_days = len(
            pair_data
        )

        common_days_matrix.loc[
            item1,
            item2
        ] = common_days


        # ----------------------------------------------
        # 共同销售天数不足，不计算
        # ----------------------------------------------
        if (
            common_days
            <
            MIN_COMMON_DAYS
        ):

            spearman_corr.loc[
                item1,
                item2
            ] = np.nan

            continue


        # ----------------------------------------------
        # Spearman相关性
        # ----------------------------------------------
        r, p = spearmanr(
            pair_data[
                item1
            ],
            pair_data[
                item2
            ]
        )

        spearman_corr.loc[
            item1,
            item2
        ] = r


print(
    "\nSpearman相关系数计算完成！"
)


# ==========================================================
# 13. 为聚类准备矩阵
#
# 聚类算法不能有 NaN
#
# 只在聚类排序阶段：
# 将无有效结果的组合临时按0相关处理
#
# Excel中仍保留NaN
# ==========================================================

corr_for_cluster = (
    spearman_corr
    .fillna(0)
    .copy()
)


# ==========================================================
# 14. 基于相关系数进行层次聚类
#
# 距离 = 1 - Spearman相关系数
# ==========================================================

distance_matrix = (
    1
    -
    corr_for_cluster
)

np.fill_diagonal(
    distance_matrix.values,
    0
)

condensed_distance = squareform(
    distance_matrix.values,
    checks=False
)

Z = linkage(
    condensed_distance,
    method="average"
)

order = leaves_list(
    Z
)

ordered_items = [
    corr_for_cluster.index[i]
    for i in order
]

ordered_corr = (
    spearman_corr
    .loc[
        ordered_items,
        ordered_items
    ]
)


# ==========================================================
# 15. 简洁版相关性热力图
#
# 只画下三角
# ==========================================================

mask = np.triu(
    np.ones_like(
        ordered_corr,
        dtype=bool
    ),
    k=1
)


# ==========================================================
# 16. 简洁红白蓝配色
# ==========================================================

corr_cmap = (
    LinearSegmentedColormap
    .from_list(
        "clean_corr",
        [
            "#4F81BD",   # 负相关
            "#D9E4F0",
            "#F7F7F7",   # 0附近
            "#F2D1CB",
            "#C95A4A"    # 正相关
        ],
        N=256
    )
)


# ==========================================================
# 17. 图9：
# 代表性单品 Spearman 相关性
# ==========================================================

fig, ax = plt.subplots(
    figsize=(13, 11),
    dpi=150
)


sns.heatmap(
    ordered_corr,

    mask=mask,

    cmap=corr_cmap,

    vmin=-1,
    vmax=1,
    center=0,

    square=True,

    linewidths=0.65,
    linecolor="white",

    cbar_kws={
        "label": "Spearman 相关系数",
        "shrink": 0.72,
        "pad": 0.025
    },

    ax=ax
)


# ==========================================================
# 18. 图9标题
# ==========================================================

ax.set_title(
    "代表性蔬菜单品销量相关性",
    fontsize=20,
    fontweight="bold",
    pad=24
)

ax.text(
    0.5,
    1.015,

    "Spearman秩相关：红色表示正相关，蓝色表示负相关",

    transform=ax.transAxes,

    ha="center",
    va="bottom",

    fontsize=11,

    color="#666666"
)


# ==========================================================
# 19. 图9坐标轴
# ==========================================================

ax.set_xlabel("")
ax.set_ylabel("")

plt.xticks(
    rotation=90,
    fontsize=8
)

plt.yticks(
    rotation=0,
    fontsize=8
)

plt.tight_layout()


plt.savefig(
    "问题1_图9_简洁版_代表性单品Spearman相关性.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()


# ==========================================================
# 20. 提取所有有效单品组合
# ==========================================================

relationship_list = []


for i in range(
    len(items)
):

    for j in range(
        i + 1,
        len(items)
    ):

        item1 = items[i]
        item2 = items[j]

        r = spearman_corr.loc[
            item1,
            item2
        ]

        common_days = (
            common_days_matrix.loc[
                item1,
                item2
            ]
        )


        # ----------------------------------------------
        # 无有效相关性则跳过
        # ----------------------------------------------
        if pd.isna(r):
            continue


        relationship_list.append(
            {
                "单品1": item1,
                "单品2": item2,
                "共同销售天数": common_days,
                "Spearman相关系数": r,
                "相关系数绝对值": abs(r)
            }
        )


relationship_df = pd.DataFrame(
    relationship_list
)


# ==========================================================
# 21. 定义相关程度
# ==========================================================

def classify_corr(r):

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


    return (
        direction
        +
        strength
    )


relationship_df[
    "相关程度"
] = (
    relationship_df[
        "Spearman相关系数"
    ]
    .apply(
        classify_corr
    )
)


# ==========================================================
# 22. 正相关排序
# ==========================================================

positive_corr = (
    relationship_df[
        relationship_df[
            "Spearman相关系数"
        ] > 0
    ]
    .sort_values(
        [
            "Spearman相关系数",
            "共同销售天数"
        ],
        ascending=[
            False,
            False
        ]
    )
)


# ==========================================================
# 23. 负相关排序
# ==========================================================

negative_corr = (
    relationship_df[
        relationship_df[
            "Spearman相关系数"
        ] < 0
    ]
    .sort_values(
        [
            "Spearman相关系数",
            "共同销售天数"
        ],
        ascending=[
            True,
            False
        ]
    )
)


# ==========================================================
# 24. 输出正相关最强15组
# ==========================================================

print("\n====================================")
print("修正后 Spearman 正相关最强的15组单品")
print("====================================")

print(
    positive_corr[
        [
            "单品1",
            "单品2",
            "共同销售天数",
            "Spearman相关系数",
            "相关程度"
        ]
    ]
    .head(15)
    .round(4)
    .to_string(
        index=False
    )
)


# ==========================================================
# 25. 输出负相关最明显15组
# ==========================================================

print("\n====================================")
print("修正后 Spearman 负相关最明显的15组单品")
print("====================================")

print(
    negative_corr[
        [
            "单品1",
            "单品2",
            "共同销售天数",
            "Spearman相关系数",
            "相关程度"
        ]
    ]
    .head(15)
    .round(4)
    .to_string(
        index=False
    )
)


# ==========================================================
# 26. 简洁版累计销量图
#
# 每个品类只显示Top 3
# ==========================================================

plot_items = (
    valid_items
    .sort_values(
        [
            "分类名称",
            "总销量"
        ],
        ascending=[
            True,
            False
        ]
    )
    .groupby(
        "分类名称",
        group_keys=False
    )
    .head(
        TOP_SHOW
    )
    .copy()
)


# ==========================================================
# 27. 累计销量排序
# ==========================================================

plot_items = (
    plot_items
    .sort_values(
        "总销量",
        ascending=True
    )
)


# ==========================================================
# 28. 构造显示名称
# ==========================================================

plot_items[
    "显示名称"
] = (
    plot_items[
        "单品名称"
    ]
    +
    "  |  "
    +
    plot_items[
        "分类名称"
    ]
)


# ==========================================================
# 29. 图10：
# 各品类主要单品累计销量
# ==========================================================

fig, ax = plt.subplots(
    figsize=(11, 8),
    dpi=150
)


bars = ax.barh(
    plot_items[
        "显示名称"
    ],

    plot_items[
        "总销量"
    ],

    color="#789FD0",

    height=0.68,

    alpha=0.92
)


# ==========================================================
# 30. 柱尾添加累计销量
# ==========================================================

max_value = (
    plot_items[
        "总销量"
    ]
    .max()
)


for bar, value in zip(
    bars,
    plot_items[
        "总销量"
    ]
):

    ax.text(
        value
        +
        max_value
        * 0.012,

        bar.get_y()
        +
        bar.get_height()
        / 2,

        f"{value:,.0f}",

        va="center",
        ha="left",

        fontsize=9,

        color="#444444"
    )


# ==========================================================
# 31. 图10标题
# ==========================================================

ax.set_title(
    "各蔬菜品类主要单品累计销量",
    fontsize=20,
    fontweight="bold",
    pad=22
)

ax.text(
    0.5,
    1.015,

    f"各品类选取累计销量最高的{TOP_SHOW}个有效单品",

    transform=ax.transAxes,

    ha="center",
    va="bottom",

    fontsize=10.5,

    color="#666666"
)


# ==========================================================
# 32. 图10坐标轴
# ==========================================================

ax.set_xlabel(
    "累计销量（千克）",
    fontsize=11
)

ax.set_ylabel(
    "",
    fontsize=11
)

ax.tick_params(
    axis="y",
    labelsize=9.5
)

ax.tick_params(
    axis="x",
    labelsize=9
)


# ==========================================================
# 33. 给数值标签留出空间
# ==========================================================

ax.set_xlim(
    0,
    max_value
    * 1.12
)


# ==========================================================
# 34. 网格
# ==========================================================

ax.grid(
    axis="x",
    linestyle="--",
    linewidth=0.7,
    alpha=0.25
)

ax.grid(
    axis="y",
    visible=False
)


# ==========================================================
# 35. 边框和背景
# ==========================================================

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

ax.spines[
    "left"
].set_color(
    "#BBBBBB"
)

ax.spines[
    "bottom"
].set_color(
    "#BBBBBB"
)

ax.set_facecolor(
    "#F8F9FB"
)


plt.tight_layout()


plt.savefig(
    "问题1_图10_简洁版_主要单品累计销量.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()


# ==========================================================
# 36. 保存Excel
# ==========================================================

with pd.ExcelWriter(
    "问题1_最终版_单品销售关联分析结果.xlsx",
    engine="openpyxl"
) as writer:


    # ----------------------------------------------
    # 全部单品统计
    # ----------------------------------------------
    item_stats.to_excel(
        writer,
        sheet_name="全部单品统计",
        index=False
    )


    # ----------------------------------------------
    # 有效单品
    # ----------------------------------------------
    valid_items.to_excel(
        writer,
        sheet_name="有效单品",
        index=False
    )


    # ----------------------------------------------
    # 代表性单品
    # ----------------------------------------------
    representative_items.to_excel(
        writer,
        sheet_name="代表性单品",
        index=False
    )


    # ----------------------------------------------
    # 原始代表单品日销量
    # ----------------------------------------------
    item_matrix.to_excel(
        writer,
        sheet_name="代表单品原始日销量"
    )


    # ----------------------------------------------
    # Spearman相关矩阵
    # ----------------------------------------------
    spearman_corr.round(
        4
    ).to_excel(
        writer,
        sheet_name="Spearman相关矩阵"
    )


    # ----------------------------------------------
    # 共同销售天数矩阵
    # ----------------------------------------------
    common_days_matrix.to_excel(
        writer,
        sheet_name="共同销售天数矩阵"
    )


    # ----------------------------------------------
    # 全部有效组合
    # ----------------------------------------------
    relationship_df.round(
        4
    ).to_excel(
        writer,
        sheet_name="有效单品组合",
        index=False
    )


    # ----------------------------------------------
    # 正相关排序
    # ----------------------------------------------
    positive_corr.round(
        4
    ).to_excel(
        writer,
        sheet_name="正相关排序",
        index=False
    )


    # ----------------------------------------------
    # 负相关排序
    # ----------------------------------------------
    negative_corr.round(
        4
    ).to_excel(
        writer,
        sheet_name="负相关排序",
        index=False
    )


    # ----------------------------------------------
    # 累计销量图展示单品
    # ----------------------------------------------
    plot_items.to_excel(
        writer,
        sheet_name="主要单品累计销量",
        index=False
    )


# ==========================================================
# 37. 完成提示
# ==========================================================

print("\n====================================")
print("第五步最终版运行完成！")
print("====================================")

print("\n已生成：")

print(
    "1. 问题1_图9_简洁版_代表性单品Spearman相关性.png"
)

print(
    "2. 问题1_图10_简洁版_主要单品累计销量.png"
)

print(
    "3. 问题1_最终版_单品销售关联分析结果.xlsx"
)