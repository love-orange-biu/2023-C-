import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# ============================================================
# 1. 基本设置
# ============================================================

file_path = "C题_正确处理后建模数据.xlsx"

step3_path = (
    "问题二_Step3/"
    "2023年7月1-7日六大品类基础需求预测.xlsx"
)

output_dir = Path("问题二_Step4")
output_dir.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei"
]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 2. 六大品类低饱和度配色
# ============================================================

category_colors = {
    "花叶类": "#8FAED3",
    "花菜类": "#E3AD7A",
    "水生根茎类": "#8DBCA4",
    "茄类": "#D99593",
    "辣椒类": "#A99BC8",
    "食用菌": "#B8A087"
}

categories = [
    "花叶类",
    "花菜类",
    "水生根茎类",
    "茄类",
    "辣椒类",
    "食用菌"
]

grid_color = "#D8D8D8"
axis_color = "#666666"
background_color = "#FAFAFA"

# ============================================================
# 3. 读取历史品类日数据
# ============================================================

df = pd.read_excel(
    file_path,
    sheet_name="品类日数据"
)

df["日期"] = pd.to_datetime(df["日期"])

df["批发价"] = pd.to_numeric(
    df["加权平均批发价(元/千克)"],
    errors="coerce"
)

df = df.dropna(
    subset=["日期", "分类名称", "批发价"]
).copy()

# ============================================================
# 4. 读取Step 3基础需求预测
# ============================================================

forecast_df = pd.read_excel(step3_path)

forecast_df["日期"] = pd.to_datetime(
    forecast_df["日期"]
)

# ============================================================
# 5. 构造需求函数参数 A
# ============================================================
#
# ln(Q) = Z + β ln(P)
#
# 因此：
#
# Q = exp(Z) * P^β
#
# 令：
#
# A = exp(Z)
#
# 则：
#
# D(P) = A * P^β
# ============================================================

forecast_df["需求函数参数A"] = np.exp(
    forecast_df["预测基础需求参数Z"]
)

# ============================================================
# 6. 构造完整每日时间索引
# ============================================================

full_dates = pd.date_range(
    start="2020-07-01",
    end="2023-06-30",
    freq="D"
)

future_dates = pd.date_range(
    start="2023-07-01",
    periods=7,
    freq="D"
)

# ============================================================
# 7. SARIMA候选模型
# ============================================================

candidate_models = [
    ((1, 0, 0), (1, 0, 0, 7)),
    ((1, 0, 1), (1, 0, 0, 7)),
    ((1, 0, 1), (1, 0, 1, 7)),
    ((2, 0, 1), (1, 0, 0, 7)),
    ((1, 1, 1), (1, 0, 1, 7))
]

# ============================================================
# 8. 初始化结果容器
# ============================================================

cost_forecasts = []

cost_validation = []

cost_model_comparison = []

# ============================================================
# 9. 六大品类分别建模
# ============================================================

for category in categories:

    print("\n" + "=" * 70)
    print(category)
    print("=" * 70)

    # --------------------------------------------------------
    # 9.1 提取该品类批发价格
    # --------------------------------------------------------

    temp = df[
        df["分类名称"] == category
    ].copy()

    temp = (
        temp
        .sort_values("日期")
        .set_index("日期")
    )

    # --------------------------------------------------------
    # 9.2 补齐每日时间序列
    # --------------------------------------------------------

    temp = temp.reindex(full_dates)

    # --------------------------------------------------------
    # 9.3 缺失值时间插值
    # --------------------------------------------------------

    temp["批发价"] = (
        temp["批发价"]
        .interpolate(method="time")
        .ffill()
        .bfill()
    )

    # --------------------------------------------------------
    # 9.4 明确指定日频率
    # --------------------------------------------------------

    temp.index = pd.DatetimeIndex(
        temp.index,
        freq="D"
    )

    series = temp["批发价"].astype(float).copy()

    # ========================================================
    # 10. 划分训练集和验证集
    # ========================================================

    validation_days = 28

    train = series.iloc[:-validation_days]
    valid = series.iloc[-validation_days:]

    best_spec = None

    # ========================================================
    # 11. 候选SARIMA模型比较
    # ========================================================

    for order, seasonal_order in candidate_models:

        try:

            model = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            result = model.fit(
                disp=False,
                maxiter=200
            )

            pred = result.forecast(
                steps=validation_days
            )

            pred.index = valid.index

            # =================================================
            # RMSE
            # =================================================

            rmse = np.sqrt(
                mean_squared_error(
                    valid.values,
                    pred.values
                )
            )

            # =================================================
            # NRMSE
            #
            # NRMSE = RMSE / 验证集平均批发价格
            # =================================================

            mean_actual = np.mean(
                valid.values
            )

            nrmse = (
                rmse / mean_actual
                if mean_actual != 0
                else np.nan
            )

            # =================================================
            # MAPE
            #
            # MAPE = mean(|实际-预测|/实际)
            # =================================================

            valid_array = valid.values
            pred_array = pred.values

            nonzero_mask = (
                np.abs(valid_array) > 1e-8
            )

            if np.any(nonzero_mask):

                mape = np.mean(
                    np.abs(
                        (
                            valid_array[nonzero_mask]
                            - pred_array[nonzero_mask]
                        )
                        /
                        valid_array[nonzero_mask]
                    )
                )

            else:

                mape = np.nan

            # -------------------------------------------------
            # 保存所有候选模型
            # -------------------------------------------------

            cost_model_comparison.append(
                {
                    "分类名称": category,
                    "order": str(order),
                    "seasonal_order":
                        str(seasonal_order),
                    "AIC": result.aic,
                    "验证RMSE": rmse,
                    "验证NRMSE": nrmse,
                    "验证MAPE": mape
                }
            )

            print(
                f"order={order}, "
                f"seasonal={seasonal_order}, "
                f"AIC={result.aic:.2f}, "
                f"RMSE={rmse:.4f}, "
                f"NRMSE={nrmse:.2%}, "
                f"MAPE={mape:.2%}"
            )

            # =================================================
            # 最优模型仍按RMSE最小选择
            # =================================================

            if (
                best_spec is None
                or rmse < best_spec["RMSE"]
            ):

                best_spec = {
                    "order": order,
                    "seasonal_order":
                        seasonal_order,
                    "RMSE": rmse,
                    "NRMSE": nrmse,
                    "MAPE": mape,
                    "AIC": result.aic
                }

        except Exception as e:

            print(
                "模型失败：",
                order,
                seasonal_order,
                str(e)
            )

    # ========================================================
    # 12. 检查有效模型
    # ========================================================

    if best_spec is None:

        raise RuntimeError(
            f"{category} 未找到可用SARIMA模型"
        )

    best_order = best_spec["order"]

    best_seasonal_order = (
        best_spec["seasonal_order"]
    )

    print("\n最佳模型：")

    print(
        f"order = {best_order}"
    )

    print(
        "seasonal_order = "
        f"{best_seasonal_order}"
    )

    print(
        f"RMSE = {best_spec['RMSE']:.4f}"
    )

    print(
        f"NRMSE = {best_spec['NRMSE']:.2%}"
    )

    print(
        f"MAPE = {best_spec['MAPE']:.2%}"
    )

    # ========================================================
    # 13. 使用完整历史数据重新拟合最佳模型
    # ========================================================

    final_model = SARIMAX(
        series,
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    final_result = final_model.fit(
        disp=False,
        maxiter=300
    )

    # ========================================================
    # 14. 预测2023年7月1—7日批发价格
    # ========================================================

    forecast_obj = final_result.get_forecast(
        steps=7
    )

    pred_cost = forecast_obj.predicted_mean

    ci = forecast_obj.conf_int(
        alpha=0.05
    )

    pred_cost.index = future_dates
    ci.index = future_dates

    # --------------------------------------------------------
    # 保存未来7天预测
    # --------------------------------------------------------

    for i, date in enumerate(future_dates):

        cost_forecasts.append(
            {
                "日期": date,
                "分类名称": category,
                "预测批发价(元/千克)":
                    float(pred_cost.iloc[i]),
                "95%下限":
                    float(ci.iloc[i, 0]),
                "95%上限":
                    float(ci.iloc[i, 1])
            }
        )

    # --------------------------------------------------------
    # 保存最终模型评价
    # --------------------------------------------------------

    cost_validation.append(
        {
            "分类名称": category,
            "最佳order":
                str(best_order),
            "最佳seasonal_order":
                str(best_seasonal_order),
            "验证RMSE":
                best_spec["RMSE"],
            "验证NRMSE":
                best_spec["NRMSE"],
            "验证MAPE":
                best_spec["MAPE"],
            "最终模型AIC":
                final_result.aic
        }
    )

# ============================================================
# 15. 整理结果
# ============================================================

cost_df = pd.DataFrame(
    cost_forecasts
)

validation_df = pd.DataFrame(
    cost_validation
)

comparison_df = pd.DataFrame(
    cost_model_comparison
)

# ============================================================
# 16. 保存Excel
# ============================================================

cost_df.to_excel(
    output_dir /
    "2023年7月1-7日六大品类批发价格预测.xlsx",
    index=False
)

validation_df.to_excel(
    output_dir /
    "批发价格预测模型验证.xlsx",
    index=False
)

comparison_df.to_excel(
    output_dir /
    "批发价格候选SARIMA模型比较.xlsx",
    index=False
)

# ============================================================
# 17. 合并Step 3需求预测与Step 4成本预测
# ============================================================

decision_base = forecast_df[
    [
        "日期",
        "分类名称",
        "价格弹性β",
        "预测基础需求参数Z",
        "需求函数参数A",
        "参考售价(元/千克)",
        "预测基准销量(千克)"
    ]
].copy()

decision_base = decision_base.merge(
    cost_df[
        [
            "日期",
            "分类名称",
            "预测批发价(元/千克)"
        ]
    ],
    on=["日期", "分类名称"],
    how="left"
)

# ============================================================
# 18. 生成需求函数
# ============================================================

decision_base["需求函数"] = (
    decision_base.apply(
        lambda row:
        (
            f"D(P)="
            f"{row['需求函数参数A']:.4f}"
            f"*P^({row['价格弹性β']:.4f})"
        ),
        axis=1
    )
)

decision_base.to_excel(
    output_dir /
    "未来7天需求函数与成本参数.xlsx",
    index=False
)

# ============================================================
# 19. 输出模型验证结果
# ============================================================

print("\n" + "=" * 80)
print("六大品类SARIMA模型验证结果")
print("=" * 80)

display_validation = (
    validation_df.copy()
)

display_validation[
    "验证NRMSE(%)"
] = (
    display_validation[
        "验证NRMSE"
    ] * 100
)

display_validation[
    "验证MAPE(%)"
] = (
    display_validation[
        "验证MAPE"
    ] * 100
)

print(
    display_validation[
        [
            "分类名称",
            "最佳order",
            "最佳seasonal_order",
            "验证RMSE",
            "验证NRMSE(%)",
            "验证MAPE(%)"
        ]
    ]
    .round(3)
    .to_string(index=False)
)

# ============================================================
# 20. 绘制未来7天批发价格预测图
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(17, 10),
    facecolor="white"
)

axes = axes.flatten()

for ax, category in zip(
    axes,
    categories
):

    temp_plot = cost_df[
        cost_df["分类名称"] == category
    ].copy()

    color = category_colors[
        category
    ]

    ax.set_facecolor(
        background_color
    )

    # --------------------------------------------------------
    # 95%预测区间
    # --------------------------------------------------------

    ax.fill_between(
        temp_plot["日期"],
        temp_plot["95%下限"],
        temp_plot["95%上限"],
        color=color,
        alpha=0.10,
        linewidth=0,
        label="95%预测区间"
    )

    # --------------------------------------------------------
    # 预测批发价格
    # --------------------------------------------------------

    ax.plot(
        temp_plot["日期"],
        temp_plot[
            "预测批发价(元/千克)"
        ],
        color=color,
        linewidth=2.2,
        alpha=0.82,
        marker="o",
        markersize=5.5,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="预测批发价",
        zorder=3
    )

    ax.set_title(
        category,
        fontsize=13,
        fontweight="bold",
        pad=8
    )

    ax.set_ylabel(
        "批发价格（元/千克）",
        fontsize=10
    )

    ax.tick_params(
        axis="x",
        rotation=30,
        labelsize=9
    )

    ax.tick_params(
        axis="y",
        labelsize=9
    )

    ax.grid(
        True,
        linestyle="--",
        color=grid_color,
        linewidth=0.7,
        alpha=0.35
    )

    ax.set_axisbelow(True)

    ax.legend(
        fontsize=9,
        frameon=True,
        facecolor="white",
        edgecolor="#DDDDDD",
        framealpha=0.85
    )

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    ax.spines["left"].set_color(
        axis_color
    )

    ax.spines["bottom"].set_color(
        axis_color
    )

fig.suptitle(
    "2023年7月1—7日六大蔬菜品类批发价格预测",
    fontsize=18,
    fontweight="bold",
    y=0.98
)

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)

plt.savefig(
    output_dir /
    "六大品类未来7天批发价格预测.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

# ============================================================
# 21. 绘制NRMSE验证误差图
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 6),
    facecolor="white"
)

ax.set_facecolor(
    background_color
)

x = np.arange(
    len(validation_df)
)

for i, row in validation_df.iterrows():

    category = row[
        "分类名称"
    ]

    # 转换成百分数
    value = (
        row["验证NRMSE"] * 100
    )

    color = category_colors[
        category
    ]

    ax.bar(
        i,
        value,
        width=0.65,
        color=color,
        alpha=0.50,
        edgecolor="white",
        linewidth=0.8
    )

    ax.text(
        i,
        value + 0.5,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#444444"
    )

ax.set_xticks(x)

ax.set_xticklabels(
    validation_df[
        "分类名称"
    ],
    fontsize=10
)

ax.set_ylabel(
    "验证集 NRMSE（%）",
    fontsize=11
)

ax.set_xlabel(
    "蔬菜品类",
    fontsize=11
)

ax.set_title(
    "六大蔬菜品类批发价格预测模型相对误差",
    fontsize=16,
    fontweight="bold",
    pad=14
)

ax.grid(
    axis="y",
    color=grid_color,
    linestyle="--",
    linewidth=0.8,
    alpha=0.35
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(
    False
)

ax.spines["right"].set_visible(
    False
)

ax.spines["left"].set_color(
    axis_color
)

ax.spines["bottom"].set_color(
    axis_color
)

max_nrmse = (
    validation_df[
        "验证NRMSE"
    ].max() * 100
)

ax.set_ylim(
    0,
    max_nrmse * 1.20
)

plt.tight_layout()

plt.savefig(
    output_dir /
    "批发价格预测模型NRMSE验证误差.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

# ============================================================
# 22. 绘制MAPE验证误差图
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 6),
    facecolor="white"
)

ax.set_facecolor(
    background_color
)

for i, row in validation_df.iterrows():

    category = row[
        "分类名称"
    ]

    value = (
        row["验证MAPE"] * 100
    )

    color = category_colors[
        category
    ]

    ax.bar(
        i,
        value,
        width=0.65,
        color=color,
        alpha=0.50,
        edgecolor="white",
        linewidth=0.8
    )

    ax.text(
        i,
        value + 0.5,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#444444"
    )

ax.set_xticks(x)

ax.set_xticklabels(
    validation_df[
        "分类名称"
    ],
    fontsize=10
)

ax.set_ylabel(
    "验证集 MAPE（%）",
    fontsize=11
)

ax.set_xlabel(
    "蔬菜品类",
    fontsize=11
)

ax.set_title(
    "六大蔬菜品类批发价格预测模型MAPE",
    fontsize=16,
    fontweight="bold",
    pad=14
)

ax.grid(
    axis="y",
    color=grid_color,
    linestyle="--",
    linewidth=0.8,
    alpha=0.35
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(
    False
)

ax.spines["right"].set_visible(
    False
)

ax.spines["left"].set_color(
    axis_color
)

ax.spines["bottom"].set_color(
    axis_color
)

max_mape = (
    validation_df[
        "验证MAPE"
    ].max() * 100
)

ax.set_ylim(
    0,
    max_mape * 1.20
)

plt.tight_layout()

plt.savefig(
    output_dir /
    "批发价格预测模型MAPE验证误差.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

# ============================================================
# 23. 完成
# ============================================================

print("\n" + "=" * 70)
print("Step 4 修正版运行完成")
print("=" * 70)

print(
    """
已生成：

1. 2023年7月1-7日六大品类批发价格预测.xlsx
2. 批发价格预测模型验证.xlsx
3. 批发价格候选SARIMA模型比较.xlsx
4. 未来7天需求函数与成本参数.xlsx
5. 六大品类未来7天批发价格预测.png
6. 批发价格预测模型NRMSE验证误差.png
7. 批发价格预测模型MAPE验证误差.png
"""
)