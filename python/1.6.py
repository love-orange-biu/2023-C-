import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================================
# 1. 基础设置
# ==========================================================

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

sns.set_theme(
    style="white",
    font="Microsoft YaHei"
)

file_path = "C题_正确处理后建模数据.xlsx"

categories = [
    "花叶类",
    "花菜类",
    "水生根茎类",
    "茄类",
    "辣椒类",
    "食用菌"
]


# ==========================================================
# 2. 读取品类日数据
# ==========================================================

df = pd.read_excel(
    file_path,
    sheet_name="品类日数据"
)

df["日期"] = pd.to_datetime(
    df["日期"]
)

# 星期：
# Monday=0 ... Sunday=6
df["星期"] = df["日期"].dt.dayofweek

df["月份"] = df["日期"].dt.month


print("数据读取成功！")
print("数据量：", len(df))

print(
    "日期范围：",
    df["日期"].min(),
    "至",
    df["日期"].max()
)


# ==========================================================
# 3. 基础统计指标
# ==========================================================

basic_stats = (
    df
    .groupby(
        "分类名称"
    )["净销量(千克)"]
    .agg(
        日均销量="mean",
        标准差="std",
        中位数="median"
    )
)


# ==========================================================
# 4. 变异系数 CV
# ==========================================================

basic_stats["变异系数CV"] = (
    basic_stats["标准差"]
    /
    basic_stats["日均销量"]
)


# ==========================================================
# 5. 偏度
# ==========================================================

skewness = (
    df
    .groupby(
        "分类名称"
    )["净销量(千克)"]
    .skew()
)

basic_stats["偏度"] = skewness


# ==========================================================
# 6. 周末效应
#
# 周末：星期六、星期日
# 工作日：星期一至星期五
#
# 周末效应 =
# 周末平均销量 / 工作日平均销量 - 1
# ==========================================================

weekday_sales = (
    df[
        df["星期"].isin(
            [0, 1, 2, 3, 4]
        )
    ]
    .groupby(
        "分类名称"
    )["净销量(千克)"]
    .mean()
)


weekend_sales = (
    df[
        df["星期"].isin(
            [5, 6]
        )
    ]
    .groupby(
        "分类名称"
    )["净销量(千克)"]
    .mean()
)


weekend_effect = (
    weekend_sales
    /
    weekday_sales
    -
    1
)


basic_stats["周末效应"] = weekend_effect


# ==========================================================
# 7. 月平均销量
# ==========================================================

monthly_sales = (
    df
    .groupby(
        [
            "分类名称",
            "月份"
        ]
    )["净销量(千克)"]
    .mean()
    .unstack()
)


# ==========================================================
# 8. 月度极差率
#
# (最高月份平均销量 - 最低月份平均销量)
# /
# 12个月平均销量
# ==========================================================

monthly_range_rate = (
    (
        monthly_sales.max(
            axis=1
        )
        -
        monthly_sales.min(
            axis=1
        )
    )
    /
    monthly_sales.mean(
        axis=1
    )
)


basic_stats["月度极差率"] = (
    monthly_range_rate
)


# ==========================================================
# 9. 销量最高月份、最低月份
# ==========================================================

basic_stats["销量最高月份"] = (
    monthly_sales.idxmax(
        axis=1
    )
)

basic_stats["销量最低月份"] = (
    monthly_sales.idxmin(
        axis=1
    )
)


# ==========================================================
# 10. 构造 日期 × 品类 销量矩阵
# ==========================================================

category_matrix = (
    df
    .pivot_table(
        index="日期",
        columns="分类名称",
        values="净销量(千克)",
        aggfunc="sum"
    )
)


category_matrix = (
    category_matrix
    .reindex(
        columns=categories
    )
)


# ==========================================================
# 11. Spearman 品类相关系数
# ==========================================================

spearman_corr = (
    category_matrix
    .corr(
        method="spearman"
    )
)


print(
    "\nSpearman相关系数矩阵："
)

print(
    spearman_corr.round(4)
)


# ==========================================================
# 12. 每个品类与其他品类的平均相关系数
# ==========================================================

mean_corr = {}


for category in categories:

    corr_values = (
        spearman_corr
        .loc[
            category
        ]
        .drop(
            category
        )
    )

    mean_corr[
        category
    ] = (
        corr_values.mean()
    )


basic_stats[
    "平均品类相关性"
] = pd.Series(
    mean_corr
)


# ==========================================================
# 13. 固定品类顺序
# ==========================================================

basic_stats = (
    basic_stats
    .reindex(
        categories
    )
)


# ==========================================================
# 14. 综合指标结果
# ==========================================================

result = basic_stats[
    [
        "日均销量",
        "变异系数CV",
        "偏度",
        "周末效应",
        "月度极差率",
        "平均品类相关性",
        "销量最高月份",
        "销量最低月份"
    ]
].copy()


# 百分比形式
result["周末效应(%)"] = (
    result["周末效应"]
    *
    100
)

result["月度极差率(%)"] = (
    result["月度极差率"]
    *
    100
)


print(
    "\n=========================================="
)

print(
    "六大蔬菜品类综合销售特征"
)

print(
    "=========================================="
)

print(
    result.round(
        4
    ).to_string()
)


# ==========================================================
# 15. 用于综合热力图的数据
# ==========================================================

heatmap_data = (
    basic_stats[
        [
            "日均销量",
            "变异系数CV",
            "偏度",
            "周末效应",
            "月度极差率",
            "平均品类相关性"
        ]
    ]
    .copy()
)


# ==========================================================
# 16. Min-Max 标准化
#
# 不使用 sklearn
#
# x' = (x - min) / (max - min)
# ==========================================================

heatmap_scaled = pd.DataFrame(
    index=heatmap_data.index
)


for column in heatmap_data.columns:

    col_min = (
        heatmap_data[
            column
        ]
        .min()
    )

    col_max = (
        heatmap_data[
            column
        ]
        .max()
    )


    # 防止某列最大值=最小值导致除0
    if col_max == col_min:

        heatmap_scaled[
            column
        ] = 0

    else:

        heatmap_scaled[
            column
        ] = (
            (
                heatmap_data[
                    column
                ]
                -
                col_min
            )
            /
            (
                col_max
                -
                col_min
            )
        )


# ==========================================================
# 17. 修改热力图指标名称
# ==========================================================

heatmap_scaled.columns = [
    "销售规模",
    "销量波动",
    "右偏程度",
    "周末效应",
    "月度波动",
    "品类关联"
]


# ==========================================================
# 18. 柔和热力图配色
# ==========================================================

cmap = sns.blend_palette(
    [
        "#F5F0E8",
        "#DDE8D5",
        "#B8D8D0",
        "#89B9C5",
        "#7296B8"
    ],
    as_cmap=True
)


# ==========================================================
# 19. 绘制综合销售特征热力图
# ==========================================================

fig, ax = plt.subplots(
    figsize=(11, 6.5),
    dpi=150
)


heatmap = sns.heatmap(
    heatmap_scaled,

    cmap=cmap,

    vmin=0,
    vmax=1,

    annot=True,

    fmt=".2f",

    linewidths=2,

    linecolor="white",

    annot_kws={
        "fontsize": 11
    },

    cbar_kws={
        "label": "标准化特征值",
        "shrink": 0.80,
        "pad": 0.03
    },

    ax=ax
)


# ==========================================================
# 20. 标题
# ==========================================================

ax.set_title(
    "六大蔬菜品类销售特征综合比较",
    fontsize=20,
    fontweight="bold",
    pad=26
)


ax.text(
    0.5,
    1.025,

    "颜色越深表示该项特征在六个品类中的相对水平越高",

    transform=ax.transAxes,

    ha="center",
    va="bottom",

    fontsize=10.5,

    color="#666666"
)


# ==========================================================
# 21. 坐标轴
# ==========================================================

ax.set_xlabel(
    "销售特征",
    fontsize=12,
    labelpad=12
)

ax.set_ylabel(
    "蔬菜品类",
    fontsize=12,
    labelpad=12
)


ax.set_xticklabels(
    heatmap_scaled.columns,
    rotation=0,
    fontsize=11
)

ax.set_yticklabels(
    heatmap_scaled.index,
    rotation=0,
    fontsize=11
)


ax.tick_params(
    axis="both",
    length=0
)


# ==========================================================
# 22. 颜色条
# ==========================================================

cbar = (
    heatmap
    .collections[0]
    .colorbar
)

cbar.ax.tick_params(
    labelsize=9
)

cbar.ax.set_ylabel(
    "标准化特征值",
    fontsize=10,
    labelpad=12
)


plt.tight_layout()


# ==========================================================
# 23. 保存图片
# ==========================================================

plt.savefig(
    "问题1_图11_六大品类销售特征综合比较.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()


# ==========================================================
# 24. 构造最终综合结果表
# ==========================================================

summary_list = []


for category in categories:

    row = basic_stats.loc[
        category
    ]

    summary_list.append(
        {
            "分类名称":
                category,

            "日均销量":
                row[
                    "日均销量"
                ],

            "变异系数CV":
                row[
                    "变异系数CV"
                ],

            "偏度":
                row[
                    "偏度"
                ],

            "周末效应(%)":
                row[
                    "周末效应"
                ]
                *
                100,

            "月度极差率(%)":
                row[
                    "月度极差率"
                ]
                *
                100,

            "平均品类相关性":
                row[
                    "平均品类相关性"
                ],

            "销量最高月份":
                int(
                    row[
                        "销量最高月份"
                    ]
                ),

            "销量最低月份":
                int(
                    row[
                        "销量最低月份"
                    ]
                )
        }
    )


summary_df = pd.DataFrame(
    summary_list
)


# ==========================================================
# 25. 保存Excel
# ==========================================================

with pd.ExcelWriter(
    "问题1_第六步_综合销售特征.xlsx",
    engine="openpyxl"
) as writer:


    summary_df.round(
        4
    ).to_excel(
        writer,
        sheet_name="综合销售特征",
        index=False
    )


    heatmap_data.round(
        4
    ).to_excel(
        writer,
        sheet_name="原始综合指标"
    )


    heatmap_scaled.round(
        4
    ).to_excel(
        writer,
        sheet_name="标准化综合指标"
    )


    monthly_sales.round(
        4
    ).to_excel(
        writer,
        sheet_name="月度平均销量"
    )


    spearman_corr.round(
        4
    ).to_excel(
        writer,
        sheet_name="品类Spearman相关性"
    )


# ==========================================================
# 26. 自动提取关键结论
# ==========================================================

print(
    "\n=========================================="
)

print(
    "问题一关键结论自动提取"
)

print(
    "=========================================="
)


# ==========================================================
# 27. 日均销量最高
# ==========================================================

max_sales_category = (
    basic_stats[
        "日均销量"
    ]
    .idxmax()
)

max_sales_value = (
    basic_stats.loc[
        max_sales_category,
        "日均销量"
    ]
)

print(
    f"\n日均销量最高："
    f"{max_sales_category}，"
    f"{max_sales_value:.2f} 千克"
)


# ==========================================================
# 28. 日均销量最低
# ==========================================================

min_sales_category = (
    basic_stats[
        "日均销量"
    ]
    .idxmin()
)

min_sales_value = (
    basic_stats.loc[
        min_sales_category,
        "日均销量"
    ]
)

print(
    f"日均销量最低："
    f"{min_sales_category}，"
    f"{min_sales_value:.2f} 千克"
)


# ==========================================================
# 29. 波动最大
# ==========================================================

max_cv_category = (
    basic_stats[
        "变异系数CV"
    ]
    .idxmax()
)

max_cv_value = (
    basic_stats.loc[
        max_cv_category,
        "变异系数CV"
    ]
)

print(
    f"相对波动最大："
    f"{max_cv_category}，"
    f"CV={max_cv_value:.3f}"
)


# ==========================================================
# 30. 波动最小
# ==========================================================

min_cv_category = (
    basic_stats[
        "变异系数CV"
    ]
    .idxmin()
)

min_cv_value = (
    basic_stats.loc[
        min_cv_category,
        "变异系数CV"
    ]
)

print(
    f"相对波动最小："
    f"{min_cv_category}，"
    f"CV={min_cv_value:.3f}"
)


# ==========================================================
# 31. 偏度最大
# ==========================================================

max_skew_category = (
    basic_stats[
        "偏度"
    ]
    .idxmax()
)

max_skew_value = (
    basic_stats.loc[
        max_skew_category,
        "偏度"
    ]
)

print(
    f"右偏程度最高："
    f"{max_skew_category}，"
    f"偏度={max_skew_value:.3f}"
)


# ==========================================================
# 32. 周末效应最明显
# ==========================================================

max_weekend_category = (
    basic_stats[
        "周末效应"
    ]
    .idxmax()
)

max_weekend_value = (
    basic_stats.loc[
        max_weekend_category,
        "周末效应"
    ]
    *
    100
)

print(
    f"周末销量提升最明显："
    f"{max_weekend_category}，"
    f"{max_weekend_value:.2f}%"
)


# ==========================================================
# 33. 月度波动最大
# ==========================================================

max_month_category = (
    basic_stats[
        "月度极差率"
    ]
    .idxmax()
)

max_month_value = (
    basic_stats.loc[
        max_month_category,
        "月度极差率"
    ]
    *
    100
)

print(
    f"月度变化最明显："
    f"{max_month_category}，"
    f"{max_month_value:.2f}%"
)


# ==========================================================
# 34. 与其他品类平均关联最强
# ==========================================================

max_corr_category = (
    basic_stats[
        "平均品类相关性"
    ]
    .idxmax()
)

max_corr_value = (
    basic_stats.loc[
        max_corr_category,
        "平均品类相关性"
    ]
)

print(
    f"与其他品类整体关联最强："
    f"{max_corr_category}，"
    f"平均Spearman="
    f"{max_corr_value:.3f}"
)


# ==========================================================
# 35. 各品类最高、最低月份
# ==========================================================

print(
    "\n各品类销量最高/最低月份："
)


for category in categories:

    max_month = int(
        basic_stats.loc[
            category,
            "销量最高月份"
        ]
    )

    min_month = int(
        basic_stats.loc[
            category,
            "销量最低月份"
        ]
    )

    print(
        f"{category}："
        f"最高 {max_month} 月，"
        f"最低 {min_month} 月"
    )


# ==========================================================
# 36. 完成提示
# ==========================================================

print(
    "\n=========================================="
)

print(
    "第六步完成！"
)

print(
    "=========================================="
)

print(
    "\n已生成："
)

print(
    "1. 问题1_图11_六大品类销售特征综合比较.png"
)

print(
    "2. 问题1_第六步_综合销售特征.xlsx"
)