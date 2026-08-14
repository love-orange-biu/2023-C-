import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from pathlib import Path

# ============================================================
# 1. 基本设置
# ============================================================

main_file = "C题_正确处理后建模数据.xlsx"

step4_file = (
    "问题二_Step4/"
    "未来7天需求函数与成本参数.xlsx"
)

output_dir = Path("问题二_Step5_最终版")
output_dir.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei"
]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 2. 参数
# ============================================================

# 补货安全系数
SAFETY_RATE = 0.05

# 价格稳定惩罚主参数
PENALTY_SHARE = 0.05

# 用最近90天数据确定正常经营区间
LOOKBACK_DAYS = 90

# 最近价格正常波动区间：
# 至少允许 ±8%，最多允许 ±20%
MIN_PRICE_BAND = 0.08
MAX_PRICE_BAND = 0.20

# 历史价格标准差倍数
STD_MULTIPLIER = 2.0

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

grid_color = "#D8D8D8"
background_color = "#FAFAFA"

# ============================================================
# 3. 检查文件
# ============================================================

if not os.path.exists(main_file):
    raise FileNotFoundError(
        f"未找到：{main_file}"
    )

if not os.path.exists(step4_file):
    raise FileNotFoundError(
        f"未找到：{step4_file}"
    )

# ============================================================
# 4. 读取历史数据
# ============================================================

history = pd.read_excel(
    main_file,
    sheet_name="品类日数据"
)

history["日期"] = pd.to_datetime(
    history["日期"]
)

numeric_cols = [
    "净销量(千克)",
    "加权平均售价(元/千克)",
    "加权平均批发价(元/千克)",
    "成本加成率",
    "损耗率"
]

for col in numeric_cols:

    if col in history.columns:

        history[col] = pd.to_numeric(
            history[col],
            errors="coerce"
        )

# ============================================================
# 5. 直接读取Step4最终输入
# ============================================================

future = pd.read_excel(
    step4_file
)

future["日期"] = pd.to_datetime(
    future["日期"]
)

required_cols = [
    "日期",
    "分类名称",
    "价格弹性β",
    "需求函数参数A",
    "参考售价(元/千克)",
    "预测基准销量(千克)",
    "预测批发价(元/千克)"
]

missing = [
    x
    for x in required_cols
    if x not in future.columns
]

if missing:

    raise ValueError(
        "Step4缺少字段："
        + "、".join(missing)
    )

for col in required_cols[2:]:

    future[col] = pd.to_numeric(
        future[col],
        errors="coerce"
    )

future = future[
    future["日期"].between(
        "2023-07-01",
        "2023-07-07"
    )
].copy()

future = future[
    future["分类名称"].isin(
        categories
    )
].copy()

# ============================================================
# 6. 数据完整性检查
# ============================================================

if len(future) != 42:

    raise ValueError(
        f"未来数据应有42行，"
        f"当前为{len(future)}行"
    )

if not (
    future.groupby("日期")[
        "分类名称"
    ].nunique() == 6
).all():

    raise ValueError(
        "部分日期缺少品类"
    )

if not (
    future.groupby("分类名称")[
        "日期"
    ].nunique() == 7
).all():

    raise ValueError(
        "部分品类不足7天"
    )

# ============================================================
# 7. 打印7月1日输入，防止再次串数据
# ============================================================

print("\n" + "=" * 90)
print("2023-07-01 输入检查")
print("=" * 90)

print(
    future[
        future["日期"]
        == pd.Timestamp(
            "2023-07-01"
        )
    ][
        [
            "分类名称",
            "预测基准销量(千克)",
            "预测批发价(元/千克)",
            "参考售价(元/千克)",
            "价格弹性β"
        ]
    ]
    .round(4)
    .to_string(index=False)
)

# ============================================================
# 8. 检查Step4需求函数
# ============================================================
#
# 理论上：
#
# D(Pref) = A * Pref^β
#
# 应基本等于 Step3 的预测基准销量
#
# ============================================================

future["模型参考销量"] = (
    future["需求函数参数A"]
    *
    future["参考售价(元/千克)"]
    ** future["价格弹性β"]
)

future["需求对应误差"] = (
    np.abs(
        future["模型参考销量"]
        -
        future["预测基准销量(千克)"]
    )
    /
    future["预测基准销量(千克)"]
)

print(
    "\n需求函数与Step3预测最大相对误差："
    f"{future['需求对应误差'].max():.6%}"
)

# ============================================================
# 9. 计算品类损耗率
# ============================================================

loss_rows = []

for category in categories:

    values = history.loc[
        history["分类名称"]
        == category,
        "损耗率"
    ].dropna()

    if len(values) == 0:

        raise ValueError(
            f"{category}缺少损耗率"
        )

    loss = values.median()

    if loss > 1:
        loss /= 100

    loss = np.clip(
        loss,
        0,
        0.50
    )

    loss_rows.append(
        {
            "分类名称": category,
            "损耗率": loss
        }
    )

loss_df = pd.DataFrame(
    loss_rows
)

# ============================================================
# 10. 最近90天经营特征
# ============================================================
#
# 不再采用三年全部数据的加成率分位数，
# 而采用未来预测前最近90天，更符合短期经营状态。
#
# ============================================================

history_end = pd.Timestamp(
    "2023-06-30"
)

history_start = (
    history_end
    - pd.Timedelta(
        days=LOOKBACK_DAYS - 1
    )
)

recent = history[
    history["日期"].between(
        history_start,
        history_end
    )
].copy()

recent_rows = []

for category in categories:

    temp = recent[
        recent["分类名称"]
        == category
    ].copy()

    # ------------------------------------------
    # 最近售价
    # ------------------------------------------

    price_values = temp[
        "加权平均售价(元/千克)"
    ].dropna()

    # 若90天数据太少，退回完整历史
    if len(price_values) < 20:

        price_values = history.loc[
            history["分类名称"]
            == category,
            "加权平均售价(元/千克)"
        ].dropna()

    price_mean = price_values.mean()

    price_std = price_values.std()

    # ------------------------------------------
    # 最近成本加成率
    # ------------------------------------------

    markup_values = temp[
        "成本加成率"
    ].dropna().copy()

    if len(markup_values) < 20:

        markup_values = history.loc[
            history["分类名称"]
            == category,
            "成本加成率"
        ].dropna().copy()

    if markup_values.median() > 3:

        markup_values = (
            markup_values
            / 100
        )

    markup_values = markup_values[
        (markup_values >= 0)
        &
        (markup_values <= 3)
    ]

    # 使用10%~90%分位数
    markup_low = markup_values.quantile(
        0.10
    )

    markup_high = markup_values.quantile(
        0.90
    )

    recent_rows.append(
        {
            "分类名称":
                category,

            "近期平均售价":
                price_mean,

            "近期售价标准差":
                price_std,

            "近期加成率10%分位":
                markup_low,

            "近期加成率90%分位":
                markup_high
        }
    )

recent_df = pd.DataFrame(
    recent_rows
)

# ============================================================
# 11. 合并模型参数
# ============================================================

data = (
    future
    .merge(
        loss_df,
        on="分类名称",
        how="left"
    )
    .merge(
        recent_df,
        on="分类名称",
        how="left"
    )
)

# ============================================================
# 12. 需求函数
# ============================================================

def demand_function(
    price,
    A,
    beta
):

    if price <= 0:
        return 0.0

    return max(
        float(
            A * price ** beta
        ),
        0.0
    )

# ============================================================
# 13. 实际经营利润
# ============================================================
#
# D(P) = A P^β
#
# 补货量：
#
# R = (1+s)D/(1-loss)
#
# 利润：
#
# π = P D - C R
#
# ============================================================

def actual_profit(
    price,
    cost,
    A,
    beta,
    loss_rate
):

    demand = demand_function(
        price,
        A,
        beta
    )

    replenish = (
        (1 + SAFETY_RATE)
        * demand
        / max(
            1 - loss_rate,
            1e-6
        )
    )

    return (
        price * demand
        -
        cost * replenish
    )

# ============================================================
# 14. 价格稳定性惩罚
# ============================================================
#
# 为避免恒弹性需求 |β|<1 时，
# 利润持续随价格升高，
# 增加价格偏离正常售价的经营风险成本：
#
# R(P) =
# λ × Revenue0 ×
# [(P-Pref)/σeff]^2
#
# λ = 0.05
#
# σeff：
# 至少取参考售价5%，
# 防止历史价格波动极小时惩罚过强。
#
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

    return (
        penalty_share
        * baseline_revenue
        * deviation ** 2
    )

# ============================================================
# 15. 综合目标
# ============================================================

def robust_objective(
    price,
    cost,
    A,
    beta,
    loss_rate,
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
        loss_rate
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
# 16. 构造价格范围
# ============================================================
#
# 新版本：
#
# ① 最近90天加成率作为经营参考；
# ② 最近价格波动确定可接受价格范围；
# ③ 保证参考售价在可行范围内；
# ④ 售价不能低于采购成本的101%。
#
# ============================================================

def get_price_bounds(row):

    cost = float(
        row[
            "预测批发价(元/千克)"
        ]
    )

    pref = float(
        row[
            "参考售价(元/千克)"
        ]
    )

    std = float(
        row[
            "近期售价标准差"
        ]
    )

    markup_low = float(
        row[
            "近期加成率10%分位"
        ]
    )

    markup_high = float(
        row[
            "近期加成率90%分位"
        ]
    )

    # ------------------------------------------
    # 由历史价格标准差决定波动范围
    # ------------------------------------------

    band_abs = max(
        STD_MULTIPLIER * std,
        MIN_PRICE_BAND * pref
    )

    band_abs = min(
        band_abs,
        MAX_PRICE_BAND * pref
    )

    volatility_low = (
        pref
        - band_abs
    )

    volatility_high = (
        pref
        + band_abs
    )

    # ------------------------------------------
    # 最近加成率约束
    # ------------------------------------------

    markup_price_low = (
        cost
        * (
            1 + markup_low
        )
    )

    markup_price_high = (
        cost
        * (
            1 + markup_high
        )
    )

    # 保证当前参考售价可以进入合理经营区间
    markup_price_low = min(
        markup_price_low,
        pref
    )

    markup_price_high = max(
        markup_price_high,
        pref
    )

    # ------------------------------------------
    # 综合价格范围
    # ------------------------------------------

    price_low = max(
        cost * 1.01,
        volatility_low,
        markup_price_low
    )

    price_high = min(
        volatility_high,
        markup_price_high
    )

    # ------------------------------------------
    # 极少数无交集情况
    # ------------------------------------------

    if price_low >= price_high:

        price_low = max(
            cost * 1.01,
            pref * 0.92
        )

        price_high = max(
            price_low * 1.02,
            pref * 1.08
        )

    return (
        price_low,
        price_high
    )

# ============================================================
# 17. 给定惩罚系数运行一次优化
# ============================================================

def run_optimization(
    input_data,
    penalty_share
):

    result_rows = []

    for _, row in input_data.iterrows():

        date = row["日期"]

        category = row[
            "分类名称"
        ]

        beta = float(
            row[
                "价格弹性β"
            ]
        )

        A = float(
            row[
                "需求函数参数A"
            ]
        )

        cost = float(
            row[
                "预测批发价(元/千克)"
            ]
        )

        pref = float(
            row[
                "参考售价(元/千克)"
            ]
        )

        base_demand = float(
            row[
                "预测基准销量(千克)"
            ]
        )

        loss = float(
            row[
                "损耗率"
            ]
        )

        price_std = float(
            row[
                "近期售价标准差"
            ]
        )

        price_low, price_high = (
            get_price_bounds(
                row
            )
        )

        # --------------------------------------
        # 基准收入
        # --------------------------------------

        baseline_revenue = (
            pref
            * base_demand
        )

        # --------------------------------------
        # 优化综合目标
        # --------------------------------------

        opt = minimize_scalar(

            lambda p:
            -robust_objective(
                p,
                cost,
                A,
                beta,
                loss,
                pref,
                baseline_revenue,
                price_std,
                penalty_share
            ),

            bounds=(
                price_low,
                price_high
            ),

            method="bounded",

            options={
                "xatol": 1e-8
            }
        )

        optimal_price = float(
            opt.x
        )

        # --------------------------------------
        # 优化需求
        # --------------------------------------

        optimal_demand = (
            demand_function(
                optimal_price,
                A,
                beta
            )
        )

        # --------------------------------------
        # 补货量
        # --------------------------------------

        optimal_replenishment = (
            (1 + SAFETY_RATE)
            * optimal_demand
            / (
                1 - loss
            )
        )

        # --------------------------------------
        # 实际经营利润
        # --------------------------------------

        optimal_profit = (
            actual_profit(
                optimal_price,
                cost,
                A,
                beta,
                loss
            )
        )

        # --------------------------------------
        # 稳定性成本
        # --------------------------------------

        penalty = price_penalty(
            optimal_price,
            pref,
            baseline_revenue,
            price_std,
            penalty_share
        )

        robust_value = (
            optimal_profit
            - penalty
        )

        # --------------------------------------
        # 基准策略
        # --------------------------------------

        baseline_replenishment = (
            (1 + SAFETY_RATE)
            * base_demand
            / (
                1 - loss
            )
        )

        baseline_profit = (
            pref
            * base_demand
            -
            cost
            * baseline_replenishment
        )

        # --------------------------------------
        # 加成率
        # --------------------------------------

        optimal_markup = (
            optimal_price
            / cost
            - 1
        )

        # --------------------------------------
        # 价格偏离率
        # --------------------------------------

        price_deviation = (
            optimal_price
            / pref
            - 1
        )

        # --------------------------------------
        # 边界检查
        # --------------------------------------

        tolerance = 1e-4

        if (
            abs(
                optimal_price
                - price_high
            )
            < tolerance
        ):

            boundary = (
                "达到价格上界"
            )

        elif (
            abs(
                optimal_price
                - price_low
            )
            < tolerance
        ):

            boundary = (
                "达到价格下界"
            )

        else:

            boundary = (
                "内部最优"
            )

        result_rows.append(
            {
                "日期":
                    date,

                "分类名称":
                    category,

                "预测批发价(元/千克)":
                    cost,

                "参考售价(元/千克)":
                    pref,

                "预测基准销量(千克)":
                    base_demand,

                "价格弹性β":
                    beta,

                "损耗率":
                    loss,

                "安全库存率":
                    SAFETY_RATE,

                "近期售价标准差":
                    price_std,

                "价格下限(元/千克)":
                    price_low,

                "价格上限(元/千克)":
                    price_high,

                "最优售价(元/千克)":
                    optimal_price,

                "最优售价偏离率":
                    price_deviation,

                "最优成本加成率":
                    optimal_markup,

                "最优预测销量(千克)":
                    optimal_demand,

                "最优补货量(千克)":
                    optimal_replenishment,

                "基准预测利润(元)":
                    baseline_profit,

                "优化后实际预测利润(元)":
                    optimal_profit,

                "价格稳定性成本":
                    penalty,

                "综合经营目标值":
                    robust_value,

                "实际利润增加额(元)":
                    (
                        optimal_profit
                        - baseline_profit
                    ),

                "最优解位置":
                    boundary
            }
        )

    return pd.DataFrame(
        result_rows
    )

# ============================================================
# 18. 主方案：惩罚系数5%
# ============================================================

result_df = run_optimization(
    data,
    PENALTY_SHARE
)

result_df[
    "分类名称"
] = pd.Categorical(
    result_df[
        "分类名称"
    ],
    categories=categories,
    ordered=True
)

result_df = (
    result_df
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
# 19. 边界检查
# ============================================================

print("\n" + "=" * 80)
print("主方案边界情况")
print("=" * 80)

print(
    result_df[
        "最优解位置"
    ]
    .value_counts()
)

# ============================================================
# 20. 打印7月1日策略
# ============================================================

print("\n" + "=" * 100)
print("2023年7月1日最终策略")
print("=" * 100)

print(
    result_df[
        result_df["日期"]
        == pd.Timestamp(
            "2023-07-01"
        )
    ][
        [
            "分类名称",
            "预测批发价(元/千克)",
            "参考售价(元/千克)",
            "最优售价(元/千克)",
            "最优售价偏离率",
            "最优补货量(千克)",
            "优化后实际预测利润(元)",
            "最优解位置"
        ]
    ]
    .round(4)
    .to_string(index=False)
)

# ============================================================
# 21. 七天汇总
# ============================================================

summary_df = (
    result_df
    .groupby(
        "分类名称",
        observed=True
    )
    .agg(
        七天总补货量_千克=(
            "最优补货量(千克)",
            "sum"
        ),

        平均最优售价_元每千克=(
            "最优售价(元/千克)",
            "mean"
        ),

        平均价格偏离率=(
            "最优售价偏离率",
            "mean"
        ),

        基准七天利润_元=(
            "基准预测利润(元)",
            "sum"
        ),

        优化七天实际利润_元=(
            "优化后实际预测利润(元)",
            "sum"
        ),

        七天价格稳定成本=(
            "价格稳定性成本",
            "sum"
        )
    )
    .reset_index()
)

summary_df[
    "实际利润增加额_元"
] = (
    summary_df[
        "优化七天实际利润_元"
    ]
    -
    summary_df[
        "基准七天利润_元"
    ]
)

# ============================================================
# 22. 总利润
# ============================================================

total_base = (
    result_df[
        "基准预测利润(元)"
    ].sum()
)

total_opt = (
    result_df[
        "优化后实际预测利润(元)"
    ].sum()
)

total_increase = (
    total_opt
    - total_base
)

print("\n" + "=" * 80)
print("总体实际利润")
print("=" * 80)

print(
    f"基准七天总预测利润："
    f"{total_base:.2f} 元"
)

print(
    f"优化七天实际预测利润："
    f"{total_opt:.2f} 元"
)

print(
    f"实际利润增加："
    f"{total_increase:.2f} 元"
)

if total_base > 0:

    print(
        f"总体实际利润提升率："
        f"{total_increase / total_base:.2%}"
    )

# ============================================================
# 23. 惩罚系数敏感性分析
# ============================================================

sensitivity_rows = []

for penalty_share in [
    0.03,
    0.05,
    0.08
]:

    temp_result = run_optimization(
        data,
        penalty_share
    )

    boundary_counts = (
        temp_result[
            "最优解位置"
        ]
        .value_counts()
    )

    sensitivity_rows.append(
        {
            "价格稳定惩罚系数":
                penalty_share,

            "内部最优数量":
                int(
                    boundary_counts.get(
                        "内部最优",
                        0
                    )
                ),

            "达到上界数量":
                int(
                    boundary_counts.get(
                        "达到价格上界",
                        0
                    )
                ),

            "达到下界数量":
                int(
                    boundary_counts.get(
                        "达到价格下界",
                        0
                    )
                ),

            "平均售价偏离率":
                temp_result[
                    "最优售价偏离率"
                ].abs().mean(),

            "七天实际预测利润":
                temp_result[
                    "优化后实际预测利润(元)"
                ].sum()
        }
    )

sensitivity_df = pd.DataFrame(
    sensitivity_rows
)

print("\n" + "=" * 80)
print("价格稳定惩罚参数敏感性分析")
print("=" * 80)

print(
    sensitivity_df
    .round(4)
    .to_string(index=False)
)

# ============================================================
# 24. 保存Excel
# ============================================================

excel_path = (
    output_dir
    /
    "问题二_Step5_稳健补货定价优化结果.xlsx"
)

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl"
) as writer:

    result_df.to_excel(
        writer,
        sheet_name="每日最优策略",
        index=False
    )

    summary_df.to_excel(
        writer,
        sheet_name="七天品类汇总",
        index=False
    )

    sensitivity_df.to_excel(
        writer,
        sheet_name="惩罚系数敏感性",
        index=False
    )

    recent_df.to_excel(
        writer,
        sheet_name="近期经营特征",
        index=False
    )

    loss_df.to_excel(
        writer,
        sheet_name="损耗率",
        index=False
    )

# ============================================================
# 25. 绘制最优补货策略
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 7),
    facecolor="white"
)

ax.set_facecolor(
    background_color
)

for category in categories:

    temp = result_df[
        result_df[
            "分类名称"
        ] == category
    ]

    ax.plot(
        temp["日期"],
        temp[
            "最优补货量(千克)"
        ],
        color=category_colors[
            category
        ],
        linewidth=2.1,
        marker="o",
        markersize=5.5,
        alpha=0.80,
        label=category
    )

ax.set_title(
    "2023年7月1—7日六大蔬菜品类稳健最优补货策略",
    fontsize=18,
    fontweight="bold",
    pad=15
)

ax.set_xlabel(
    "日期"
)

ax.set_ylabel(
    "最优补货量（千克）"
)

ax.grid(
    linestyle="--",
    color=grid_color,
    alpha=0.40
)

ax.legend(
    ncol=3
)

ax.spines["top"].set_visible(
    False
)

ax.spines["right"].set_visible(
    False
)

plt.xticks(
    rotation=25
)

plt.tight_layout()

plt.savefig(
    output_dir
    /
    "稳健最优补货策略.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

# ============================================================
# 26. 绘制定价策略
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 7),
    facecolor="white"
)

ax.set_facecolor(
    background_color
)

for category in categories:

    temp = result_df[
        result_df[
            "分类名称"
        ] == category
    ]

    ax.plot(
        temp["日期"],
        temp[
            "最优售价(元/千克)"
        ],
        color=category_colors[
            category
        ],
        linewidth=2.1,
        marker="o",
        markersize=5.5,
        alpha=0.80,
        label=category
    )

ax.set_title(
    "2023年7月1—7日六大蔬菜品类稳健最优定价策略",
    fontsize=18,
    fontweight="bold",
    pad=15
)

ax.set_xlabel(
    "日期"
)

ax.set_ylabel(
    "最优售价（元/千克）"
)

ax.grid(
    linestyle="--",
    color=grid_color,
    alpha=0.40
)

ax.legend(
    ncol=3
)

ax.spines["top"].set_visible(
    False
)

ax.spines["right"].set_visible(
    False
)

plt.xticks(
    rotation=25
)

plt.tight_layout()

plt.savefig(
    output_dir
    /
    "稳健最优定价策略.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

# ============================================================
# 27. 优化前后实际利润比较
# ============================================================

profit_plot = (
    summary_df
    .set_index(
        "分类名称"
    )
    .reindex(
        categories
    )
)

x = np.arange(
    len(categories)
)

width = 0.34

fig, ax = plt.subplots(
    figsize=(13, 7),
    facecolor="white"
)

ax.set_facecolor(
    background_color
)

ax.bar(
    x - width / 2,
    profit_plot[
        "基准七天利润_元"
    ],
    width,
    color="#D5D5D5",
    alpha=0.70,
    label="基准策略"
)

for i, category in enumerate(
    categories
):

    ax.bar(
        x[i] + width / 2,
        profit_plot.loc[
            category,
            "优化七天实际利润_元"
        ],
        width,
        color=category_colors[
            category
        ],
        alpha=0.58
    )

from matplotlib.patches import Patch

ax.legend(
    handles=[
        Patch(
            facecolor="#D5D5D5",
            alpha=0.70,
            label="基准策略"
        ),
        Patch(
            facecolor="#8FAED3",
            alpha=0.58,
            label="稳健优化策略"
        )
    ]
)

ax.set_xticks(x)

ax.set_xticklabels(
    categories
)

ax.set_title(
    "六大蔬菜品类稳健优化前后七天预测利润比较",
    fontsize=18,
    fontweight="bold",
    pad=15
)

ax.set_xlabel(
    "蔬菜品类"
)

ax.set_ylabel(
    "七天实际预测利润（元）"
)

ax.grid(
    axis="y",
    linestyle="--",
    color=grid_color,
    alpha=0.4
)

ax.set_axisbelow(
    True
)

ax.spines["top"].set_visible(
    False
)

ax.spines["right"].set_visible(
    False
)

plt.tight_layout()

plt.savefig(
    output_dir
    /
    "稳健优化前后利润比较.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print("\n" + "=" * 80)
print("Step 5 最终稳健修正版运行完成")
print("=" * 80)

print(
    "\n重点检查："
    "\n1. “内部最优数量”是否明显增加；"
    "\n2. 是否不再出现42/42全部达到上界；"
    "\n3. 5%惩罚下平均售价偏离率是否合理；"
    "\n4. 3%、5%、8%三个方案下补货和利润结论是否基本稳定。"
)