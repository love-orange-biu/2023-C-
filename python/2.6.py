# ============================================================
# 问题二 Step 6
# 稳健性与敏感性分析 —— 最终低饱和配色版
#
# 内容：
# 1. 市场需求 ±5% 扰动
# 2. 批发价格 ±5% 扰动
# 3. 构造3×3共9种组合情景
# 4. 安全库存率3%、5%、8%敏感性分析
# 5. 输出Excel
# 6. 绘制低饱和、高透明度论文图
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# 1. 文件路径
# ============================================================

step5_file = (
    "问题二_Step5_最终版/"
    "问题二_Step5_稳健补货定价优化结果.xlsx"
)

output_dir = Path("问题二_Step6")
output_dir.mkdir(exist_ok=True)


# ============================================================
# 2. 中文字体
# ============================================================

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS"
]

plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 3. 统一低饱和度配色
# ============================================================

# 蓝
SOFT_BLUE = "#7FA9C9"

# 浅蓝
SOFT_LIGHT_BLUE = "#A9BED8"

# 橙
SOFT_ORANGE = "#DDA66D"

# 绿
SOFT_GREEN = "#8DB79E"

# 红
SOFT_RED = "#D89591"

# 紫
SOFT_PURPLE = "#A99BC5"

# 棕
SOFT_BROWN = "#B29A80"

# 灰
SOFT_GRAY = "#CECECE"

# 浅黄
SOFT_YELLOW = "#E5D69B"

# 背景
BACKGROUND_COLOR = "#FAFAFA"

# 网格
GRID_COLOR = "#D8D8D8"

# 透明度
LINE_ALPHA = 0.78
BAR_ALPHA = 0.58
HEAT_ALPHA = 0.78
GRID_ALPHA = 0.35


# ============================================================
# 4. 六大品类
# ============================================================

categories = [
    "花叶类",
    "花菜类",
    "水生根茎类",
    "茄类",
    "辣椒类",
    "食用菌"
]


category_colors = {
    "花叶类": "#8FAED3",
    "花菜类": "#E3AD7A",
    "水生根茎类": "#8DBCA4",
    "茄类": "#D99593",
    "辣椒类": "#A99BC8",
    "食用菌": "#B8A087"
}


# ============================================================
# 5. Step5模型参数
# ============================================================

# 最终价格稳定性惩罚系数
PENALTY_SHARE = 0.05


# ============================================================
# 6. 检查Step5文件
# ============================================================

if not os.path.exists(step5_file):

    raise FileNotFoundError(
        "\n未找到Step5结果文件：\n"
        f"{step5_file}\n\n"
        "请确认Step5最终版已经运行完成。"
    )


print("=" * 90)
print("问题二 Step 6：稳健性与敏感性分析")
print("=" * 90)


# ============================================================
# 7. 读取Step5每日策略
# ============================================================

data = pd.read_excel(
    step5_file,
    sheet_name="每日最优策略"
)


data["日期"] = pd.to_datetime(
    data["日期"]
)


data["分类名称"] = pd.Categorical(
    data["分类名称"],
    categories=categories,
    ordered=True
)


data = (
    data
    .sort_values(
        [
            "日期",
            "分类名称"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 8. 数据完整性检查
# ============================================================

required_cols = [
    "日期",
    "分类名称",
    "预测批发价(元/千克)",
    "参考售价(元/千克)",
    "预测基准销量(千克)",
    "价格弹性β",
    "损耗率",
    "近期售价标准差",
    "价格下限(元/千克)",
    "价格上限(元/千克)",
    "最优售价(元/千克)",
    "最优补货量(千克)",
    "优化后实际预测利润(元)"
]


missing_cols = [
    col
    for col in required_cols
    if col not in data.columns
]


if missing_cols:

    raise ValueError(
        "\nStep5结果文件缺少字段：\n"
        + "、".join(missing_cols)
    )


if len(data) != 42:

    raise ValueError(
        f"\nStep5每日策略正常应该有42行，"
        f"当前为{len(data)}行。"
    )


print("\nStep5结果读取成功。")
print("每日决策数量：", len(data))


# ============================================================
# 9. 重建需求函数参数A
# ============================================================
#
# D(P) = A * P^β
#
# 已知参考售价Pref、基准需求D0：
#
# A = D0 / Pref^β
#
# ============================================================

data["需求函数参数A"] = (
    data[
        "预测基准销量(千克)"
    ]
    /
    (
        data[
            "参考售价(元/千克)"
        ]
        **
        data[
            "价格弹性β"
        ]
    )
)


# ============================================================
# 10. 定义需求函数
# ============================================================

def demand_function(
    price,
    A,
    beta
):

    if price <= 0:

        return 0.0

    demand = (
        A
        * price ** beta
    )

    return max(
        float(demand),
        0.0
    )


# ============================================================
# 11. 实际利润函数
# ============================================================
#
# D(P) = A P^β
#
# 补货量：
#
# R = (1+s)D/(1-loss)
#
# s = 安全库存率
#
# 利润：
#
# π = P×D - C×R
#
# ============================================================

def actual_profit(
    price,
    cost,
    A,
    beta,
    loss_rate,
    safety_rate
):

    demand = demand_function(
        price,
        A,
        beta
    )

    replenishment = (
        (1 + safety_rate)
        * demand
        /
        max(
            1 - loss_rate,
            1e-6
        )
    )

    profit = (
        price * demand
        -
        cost * replenishment
    )

    return profit


# ============================================================
# 12. 价格稳定性惩罚
# ============================================================

def price_penalty(
    price,
    reference_price,
    baseline_revenue,
    price_std,
    penalty_share
):

    sigma_eff = max(
        price_std,
        0.05 * reference_price
    )

    deviation = (
        price
        - reference_price
    ) / sigma_eff

    penalty = (
        penalty_share
        * baseline_revenue
        * deviation ** 2
    )

    return penalty


# ============================================================
# 13. 综合经营目标
# ============================================================

def robust_objective(
    price,
    cost,
    A,
    beta,
    loss_rate,
    safety_rate,
    reference_price,
    baseline_revenue,
    price_std,
    penalty_share
):

    profit = actual_profit(
        price,
        cost,
        A,
        beta,
        loss_rate,
        safety_rate
    )

    penalty = price_penalty(
        price,
        reference_price,
        baseline_revenue,
        price_std,
        penalty_share
    )

    return (
        profit
        - penalty
    )


# ============================================================
# 14. 单一情景重新优化
# ============================================================

def optimize_scenario(
    source_data,
    demand_factor=1.0,
    cost_factor=1.0,
    safety_rate=0.05
):

    result_rows = []

    for _, row in source_data.iterrows():

        # ----------------------------------------------------
        # 基本参数
        # ----------------------------------------------------

        beta = float(
            row[
                "价格弹性β"
            ]
        )

        # ----------------------------------------------------
        # 市场需求扰动
        # ----------------------------------------------------

        A = (
            float(
                row[
                    "需求函数参数A"
                ]
            )
            * demand_factor
        )

        # ----------------------------------------------------
        # 批发价格扰动
        # ----------------------------------------------------

        cost = (
            float(
                row[
                    "预测批发价(元/千克)"
                ]
            )
            * cost_factor
        )

        # ----------------------------------------------------
        # 其他模型参数
        # ----------------------------------------------------

        reference_price = float(
            row[
                "参考售价(元/千克)"
            ]
        )

        loss_rate = float(
            row[
                "损耗率"
            ]
        )

        price_std = float(
            row[
                "近期售价标准差"
            ]
        )

        price_low = float(
            row[
                "价格下限(元/千克)"
            ]
        )

        price_high = float(
            row[
                "价格上限(元/千克)"
            ]
        )

        # ----------------------------------------------------
        # 基准需求同步扰动
        # ----------------------------------------------------

        baseline_demand = (
            float(
                row[
                    "预测基准销量(千克)"
                ]
            )
            * demand_factor
        )

        baseline_revenue = (
            reference_price
            * baseline_demand
        )

        # ----------------------------------------------------
        # 成本变化后调整最低售价
        # ----------------------------------------------------

        price_low_adjusted = max(
            price_low,
            cost * 1.01
        )

        # ----------------------------------------------------
        # 防止价格区间失效
        # ----------------------------------------------------

        if (
            price_low_adjusted
            >= price_high
        ):

            price_high_adjusted = (
                price_low_adjusted
                * 1.03
            )

        else:

            price_high_adjusted = (
                price_high
            )

        # ----------------------------------------------------
        # 优化
        # ----------------------------------------------------

        opt = minimize_scalar(

            lambda p:
            -robust_objective(

                p,

                cost,

                A,

                beta,

                loss_rate,

                safety_rate,

                reference_price,

                baseline_revenue,

                price_std,

                PENALTY_SHARE
            ),

            bounds=(
                price_low_adjusted,
                price_high_adjusted
            ),

            method="bounded",

            options={
                "xatol": 1e-8
            }
        )

        # ----------------------------------------------------
        # 最优售价
        # ----------------------------------------------------

        optimal_price = float(
            opt.x
        )

        # ----------------------------------------------------
        # 最优销量
        # ----------------------------------------------------

        optimal_demand = (
            demand_function(
                optimal_price,
                A,
                beta
            )
        )

        # ----------------------------------------------------
        # 最优补货量
        # ----------------------------------------------------

        optimal_replenishment = (
            (1 + safety_rate)
            * optimal_demand
            /
            max(
                1 - loss_rate,
                1e-6
            )
        )

        # ----------------------------------------------------
        # 实际利润
        # ----------------------------------------------------

        optimal_profit = (
            actual_profit(
                optimal_price,
                cost,
                A,
                beta,
                loss_rate,
                safety_rate
            )
        )

        # ----------------------------------------------------
        # 保存
        # ----------------------------------------------------

        result_rows.append(
            {
                "日期":
                    row["日期"],

                "分类名称":
                    row["分类名称"],

                "最优售价":
                    optimal_price,

                "最优补货量":
                    optimal_replenishment,

                "最优预测销量":
                    optimal_demand,

                "实际预测利润":
                    optimal_profit
            }
        )

    return pd.DataFrame(
        result_rows
    )


# ============================================================
# 15. 基准情景
# ============================================================
#
# 需求不变
# 成本不变
# 安全库存率5%
#
# ============================================================

baseline_result = optimize_scenario(
    data,
    demand_factor=1.00,
    cost_factor=1.00,
    safety_rate=0.05
)


baseline_total_profit = (
    baseline_result[
        "实际预测利润"
    ].sum()
)


baseline_total_replenishment = (
    baseline_result[
        "最优补货量"
    ].sum()
)


baseline_mean_price = (
    baseline_result[
        "最优售价"
    ].mean()
)


print("\n" + "=" * 90)
print("基准情景")
print("=" * 90)

print(
    f"七天总预测利润："
    f"{baseline_total_profit:.2f} 元"
)

print(
    f"七天总补货量："
    f"{baseline_total_replenishment:.2f} 千克"
)

print(
    f"平均最优售价："
    f"{baseline_mean_price:.3f} 元/千克"
)


# ============================================================
# 16. 构造±5%需求、成本扰动情景
# ============================================================

demand_factors = [
    0.95,
    1.00,
    1.05
]


cost_factors = [
    0.95,
    1.00,
    1.05
]


scenario_rows = []

scenario_detail_list = []


for demand_factor in demand_factors:

    for cost_factor in cost_factors:

        scenario_result = optimize_scenario(
            data,
            demand_factor=demand_factor,
            cost_factor=cost_factor,
            safety_rate=0.05
        )

        # ----------------------------------------------------
        # 总利润
        # ----------------------------------------------------

        total_profit = (
            scenario_result[
                "实际预测利润"
            ].sum()
        )

        # ----------------------------------------------------
        # 总补货量
        # ----------------------------------------------------

        total_replenishment = (
            scenario_result[
                "最优补货量"
            ].sum()
        )

        # ----------------------------------------------------
        # 平均售价
        # ----------------------------------------------------

        mean_price = (
            scenario_result[
                "最优售价"
            ].mean()
        )

        # ----------------------------------------------------
        # 相对基准变化
        # ----------------------------------------------------

        profit_change = (
            total_profit
            / baseline_total_profit
            - 1
        )

        replenishment_change = (
            total_replenishment
            / baseline_total_replenishment
            - 1
        )

        price_change = (
            mean_price
            / baseline_mean_price
            - 1
        )

        scenario_name = (

            f"需求"
            f"{(demand_factor - 1) * 100:+.0f}%"

            f"_成本"

            f"{(cost_factor - 1) * 100:+.0f}%"
        )

        scenario_rows.append(
            {
                "情景":
                    scenario_name,

                "需求扰动":
                    demand_factor - 1,

                "批发价扰动":
                    cost_factor - 1,

                "七天总预测利润":
                    total_profit,

                "七天总补货量":
                    total_replenishment,

                "平均最优售价":
                    mean_price,

                "利润变化率":
                    profit_change,

                "补货量变化率":
                    replenishment_change,

                "平均售价变化率":
                    price_change
            }
        )

        temp_detail = (
            scenario_result
            .copy()
        )

        temp_detail[
            "情景"
        ] = scenario_name

        scenario_detail_list.append(
            temp_detail
        )


scenario_df = pd.DataFrame(
    scenario_rows
)


scenario_detail_df = pd.concat(
    scenario_detail_list,
    ignore_index=True
)


# ============================================================
# 17. 输出9种情景
# ============================================================

print("\n" + "=" * 100)
print("需求与批发价格 ±5% 情景分析")
print("=" * 100)


display_scenario = (
    scenario_df.copy()
)


display_scenario[
    "需求扰动(%)"
] = (
    display_scenario[
        "需求扰动"
    ] * 100
)


display_scenario[
    "批发价扰动(%)"
] = (
    display_scenario[
        "批发价扰动"
    ] * 100
)


display_scenario[
    "利润变化率(%)"
] = (
    display_scenario[
        "利润变化率"
    ] * 100
)


display_scenario[
    "补货量变化率(%)"
] = (
    display_scenario[
        "补货量变化率"
    ] * 100
)


display_scenario[
    "平均售价变化率(%)"
] = (
    display_scenario[
        "平均售价变化率"
    ] * 100
)


print(
    display_scenario[
        [
            "情景",
            "七天总预测利润",
            "七天总补货量",
            "平均最优售价",
            "利润变化率(%)",
            "补货量变化率(%)",
            "平均售价变化率(%)"
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)


# ============================================================
# 18. 安全库存率敏感性
# ============================================================

safety_rates = [
    0.03,
    0.05,
    0.08
]


safety_rows = []


for safety_rate in safety_rates:

    safety_result = optimize_scenario(
        data,
        demand_factor=1.00,
        cost_factor=1.00,
        safety_rate=safety_rate
    )

    total_profit = (
        safety_result[
            "实际预测利润"
        ].sum()
    )

    total_replenishment = (
        safety_result[
            "最优补货量"
        ].sum()
    )

    mean_price = (
        safety_result[
            "最优售价"
        ].mean()
    )

    safety_rows.append(
        {
            "安全库存率":
                safety_rate,

            "七天总预测利润":
                total_profit,

            "七天总补货量":
                total_replenishment,

            "平均最优售价":
                mean_price,

            "相对5%方案利润变化率":
                (
                    total_profit
                    / baseline_total_profit
                    - 1
                ),

            "相对5%方案补货变化率":
                (
                    total_replenishment
                    / baseline_total_replenishment
                    - 1
                )
        }
    )


safety_df = pd.DataFrame(
    safety_rows
)


# ============================================================
# 19. 打印安全库存敏感性
# ============================================================

print("\n" + "=" * 100)
print("安全库存率敏感性分析")
print("=" * 100)


display_safety = (
    safety_df.copy()
)


display_safety[
    "安全库存率(%)"
] = (
    display_safety[
        "安全库存率"
    ] * 100
)


display_safety[
    "利润变化率(%)"
] = (
    display_safety[
        "相对5%方案利润变化率"
    ] * 100
)


display_safety[
    "补货变化率(%)"
] = (
    display_safety[
        "相对5%方案补货变化率"
    ] * 100
)


print(
    display_safety[
        [
            "安全库存率(%)",
            "七天总预测利润",
            "七天总补货量",
            "平均最优售价",
            "利润变化率(%)",
            "补货变化率(%)"
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)


# ============================================================
# 20. 稳健性总体指标
# ============================================================

max_profit_change = (
    scenario_df[
        "利润变化率"
    ]
    .abs()
    .max()
)


max_replenishment_change = (
    scenario_df[
        "补货量变化率"
    ]
    .abs()
    .max()
)


max_price_change = (
    scenario_df[
        "平均售价变化率"
    ]
    .abs()
    .max()
)


print("\n" + "=" * 100)
print("稳健性总体指标")
print("=" * 100)


print(
    f"9种情景下最大利润变化幅度："
    f"{max_profit_change:.2%}"
)


print(
    f"9种情景下最大补货量变化幅度："
    f"{max_replenishment_change:.2%}"
)


print(
    f"9种情景下平均售价最大变化幅度："
    f"{max_price_change:.2%}"
)


# ============================================================
# 21. 利润情景矩阵
# ============================================================

profit_matrix = (
    scenario_df
    .pivot(
        index="需求扰动",
        columns="批发价扰动",
        values="七天总预测利润"
    )
    .sort_index(
        ascending=False
    )
)


# ============================================================
# 22. 图1
# 安全库存率变化对利润影响
# 低饱和蓝色 + 高透明度
# ============================================================

fig, ax = plt.subplots(
    figsize=(10.5, 6.5),
    facecolor="white"
)


ax.set_facecolor(
    BACKGROUND_COLOR
)


x_safety = (
    safety_df[
        "安全库存率"
    ] * 100
)


y_profit = (
    safety_df[
        "七天总预测利润"
    ]
)


ax.plot(
    x_safety,
    y_profit,

    color=SOFT_BLUE,

    linewidth=2.6,

    marker="o",

    markersize=8,

    markerfacecolor=SOFT_BLUE,

    markeredgecolor="#6F94B1",

    alpha=LINE_ALPHA
)


# ------------------------------------------------------------
# 数值标签
# ------------------------------------------------------------

for x_value, y_value in zip(
    x_safety,
    y_profit
):

    ax.text(
        x_value + 0.03,
        y_value + 7,

        f"{y_value:.0f}",

        fontsize=11,

        color="#333333"
    )


ax.set_title(
    "安全库存率变化对七天预测利润的影响",
    fontsize=18,
    fontweight="bold",
    pad=17
)


ax.set_xlabel(
    "安全库存率（%）",
    fontsize=12
)


ax.set_ylabel(
    "七天预测利润（元）",
    fontsize=12
)


ax.grid(
    True,
    linestyle="--",
    linewidth=0.8,
    color=GRID_COLOR,
    alpha=GRID_ALPHA
)


ax.set_axisbelow(
    True
)


ax.spines[
    "top"
].set_visible(False)


ax.spines[
    "right"
].set_visible(False)


ax.spines[
    "left"
].set_color(
    "#777777"
)


ax.spines[
    "bottom"
].set_color(
    "#777777"
)


plt.tight_layout()


plt.savefig(
    output_dir
    /
    "Step6_安全库存率敏感性_低饱和版.png",

    dpi=300,

    bbox_inches="tight",

    facecolor="white"
)


plt.show()


# ============================================================
# 23. 图2
# 不同需求—成本扰动情景利润变化
#
# 正向：低饱和蓝
# 负向：低饱和红
# 基准：浅灰
# ============================================================

scenario_plot = (
    scenario_df
    .sort_values(
        [
            "需求扰动",
            "批发价扰动"
        ]
    )
    .reset_index(
        drop=True
    )
)


x_pos = np.arange(
    len(
        scenario_plot
    )
)


profit_changes = (
    scenario_plot[
        "利润变化率"
    ] * 100
)


# ------------------------------------------------------------
# 根据正负自动赋低饱和颜色
# ------------------------------------------------------------

bar_colors = []


for value in profit_changes:

    if value > 0.001:

        bar_colors.append(
            SOFT_BLUE
        )

    elif value < -0.001:

        bar_colors.append(
            SOFT_RED
        )

    else:

        bar_colors.append(
            SOFT_GRAY
        )


fig, ax = plt.subplots(
    figsize=(14, 7),
    facecolor="white"
)


ax.set_facecolor(
    BACKGROUND_COLOR
)


bars = ax.bar(
    x_pos,
    profit_changes,

    width=0.75,

    color=bar_colors,

    alpha=BAR_ALPHA,

    edgecolor="none"
)


# ------------------------------------------------------------
# 0基准线
# ------------------------------------------------------------

ax.axhline(
    0,

    color="#777777",

    linewidth=1.2,

    alpha=0.75
)


# ------------------------------------------------------------
# 添加柱上数值
# ------------------------------------------------------------

for bar, value in zip(
    bars,
    profit_changes
):

    x_center = (
        bar.get_x()
        +
        bar.get_width()
        / 2
    )

    if value >= 0:

        ax.text(
            x_center,
            value + 0.5,

            f"{value:.1f}%",

            ha="center",

            va="bottom",

            fontsize=10,

            color="#333333"
        )

    else:

        ax.text(
            x_center,
            value - 0.7,

            f"{value:.1f}%",

            ha="center",

            va="top",

            fontsize=10,

            color="#333333"
        )


ax.set_xticks(
    x_pos
)


ax.set_xticklabels(
    scenario_plot[
        "情景"
    ],

    rotation=28,

    ha="right",

    fontsize=10
)


ax.set_title(
    "不同需求—成本扰动情景下预测利润变化",
    fontsize=18,
    fontweight="bold",
    pad=17
)


ax.set_xlabel(
    "扰动情景",
    fontsize=12
)


ax.set_ylabel(
    "相对基准方案利润变化率（%）",
    fontsize=12
)


ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.8,
    color=GRID_COLOR,
    alpha=GRID_ALPHA
)


ax.set_axisbelow(
    True
)


ax.spines[
    "top"
].set_visible(False)


ax.spines[
    "right"
].set_visible(False)


ax.spines[
    "left"
].set_color(
    "#777777"
)


ax.spines[
    "bottom"
].set_color(
    "#777777"
)


plt.tight_layout()


plt.savefig(
    output_dir
    /
    "Step6_不同情景利润变化_低饱和版.png",

    dpi=300,

    bbox_inches="tight",

    facecolor="white"
)


plt.show()


# ============================================================
# 24. 图3
# 需求—批发价格扰动利润热力图
#
# 自定义低饱和：
# 浅红 → 浅米黄 → 浅绿
# ============================================================

soft_heatmap_colors = [
    "#D99A96",   # 低饱和红
    "#E8C7A0",   # 浅橙
    "#E8DCAB",   # 米黄
    "#B9CE9B",   # 浅绿
    "#83AF94"    # 低饱和绿
]


soft_cmap = (
    LinearSegmentedColormap
    .from_list(
        "soft_profit",
        soft_heatmap_colors
    )
)


fig, ax = plt.subplots(
    figsize=(9.5, 7),
    facecolor="white"
)


ax.set_facecolor(
    BACKGROUND_COLOR
)


matrix_values = (
    profit_matrix.values
)


im = ax.imshow(
    matrix_values,

    cmap=soft_cmap,

    aspect="auto",

    alpha=HEAT_ALPHA
)


# ------------------------------------------------------------
# 单元格文字
# ------------------------------------------------------------

for i in range(
    matrix_values.shape[0]
):

    for j in range(
        matrix_values.shape[1]
    ):

        ax.text(
            j,
            i,

            f"{matrix_values[i, j]:.0f}",

            ha="center",

            va="center",

            fontsize=14,

            color="#222222"
        )


# ------------------------------------------------------------
# 坐标
# ------------------------------------------------------------

ax.set_xticks(
    range(
        len(
            profit_matrix.columns
        )
    )
)


ax.set_xticklabels(
    [
        f"{x * 100:+.0f}%"
        for x
        in profit_matrix.columns
    ],

    fontsize=12
)


ax.set_yticks(
    range(
        len(
            profit_matrix.index
        )
    )
)


ax.set_yticklabels(
    [
        f"{x * 100:+.0f}%"
        for x
        in profit_matrix.index
    ],

    fontsize=12
)


ax.set_xlabel(
    "批发价格扰动",
    fontsize=13
)


ax.set_ylabel(
    "市场需求扰动",
    fontsize=13
)


ax.set_title(
    "需求与批发价格扰动下七天预测利润",
    fontsize=18,
    fontweight="bold",
    pad=17
)


# ------------------------------------------------------------
# 颜色条
# ------------------------------------------------------------

cbar = plt.colorbar(
    im,
    ax=ax,
    fraction=0.045,
    pad=0.04
)


cbar.set_label(
    "七天预测利润（元）",
    fontsize=12
)


cbar.ax.tick_params(
    labelsize=10
)


# ------------------------------------------------------------
# 清除边框
# ------------------------------------------------------------

for spine in [
    "top",
    "right",
    "left",
    "bottom"
]:

    ax.spines[
        spine
    ].set_visible(False)


plt.tight_layout()


plt.savefig(
    output_dir
    /
    "Step6_需求成本扰动利润热力图_低饱和版.png",

    dpi=300,

    bbox_inches="tight",

    facecolor="white"
)


plt.show()


# ============================================================
# 25. 保存Excel
# ============================================================

excel_path = (
    output_dir
    /
    "问题二_Step6_稳健性与敏感性分析.xlsx"
)


with pd.ExcelWriter(
    excel_path,
    engine="openpyxl"
) as writer:

    # --------------------------------------------------------
    # 9种情景
    # --------------------------------------------------------

    scenario_df.to_excel(
        writer,

        sheet_name="需求成本情景分析",

        index=False
    )

    # --------------------------------------------------------
    # 9种情景的每日详细结果
    # --------------------------------------------------------

    scenario_detail_df.to_excel(
        writer,

        sheet_name="各情景详细结果",

        index=False
    )

    # --------------------------------------------------------
    # 安全库存率
    # --------------------------------------------------------

    safety_df.to_excel(
        writer,

        sheet_name="安全库存敏感性",

        index=False
    )

    # --------------------------------------------------------
    # 利润矩阵
    # --------------------------------------------------------

    profit_matrix.to_excel(
        writer,

        sheet_name="利润情景矩阵"
    )


# ============================================================
# 26. 最终输出
# ============================================================

print("\n" + "=" * 90)
print("Step 6 运行完成")
print("=" * 90)


print(
    "\nExcel结果：\n"
    f"{excel_path}"
)


print(
    "\n生成图片："
)


print(
    "1. Step6_安全库存率敏感性_低饱和版.png"
)


print(
    "2. Step6_不同情景利润变化_低饱和版.png"
)


print(
    "3. Step6_需求成本扰动利润热力图_低饱和版.png"
)


print("\n" + "=" * 90)
print("稳健性分析最终结论数据")
print("=" * 90)


print(
    f"\n基准七天预测利润："
    f"{baseline_total_profit:.2f} 元"
)


print(
    f"需求—成本扰动下最大利润变化幅度："
    f"{max_profit_change:.2%}"
)


print(
    f"需求—成本扰动下最大补货量变化幅度："
    f"{max_replenishment_change:.2%}"
)


print(
    f"需求—成本扰动下平均售价最大变化幅度："
    f"{max_price_change:.2%}"
)