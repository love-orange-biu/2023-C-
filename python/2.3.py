import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# 1. 基本设置
# ============================================================

file_path = "C题_正确处理后建模数据.xlsx"

output_dir = Path("问题二_Step3")
output_dir.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei"
]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 2. 论文统一低饱和度配色
# ============================================================

category_colors = {
    "花叶类": "#8FAED3",      # 低饱和蓝
    "花菜类": "#E3AD7A",      # 低饱和橙
    "水生根茎类": "#8DBCA4",  # 低饱和绿
    "茄类": "#D99593",        # 低饱和红
    "辣椒类": "#A99BC8",      # 低饱和紫
    "食用菌": "#B8A087"       # 低饱和棕
}

grid_color = "#D8D8D8"
axis_color = "#666666"
background_color = "#FAFAFA"
text_color = "#333333"

# ============================================================
# 3. 读取数据
# ============================================================

df = pd.read_excel(
    file_path,
    sheet_name="品类日数据"
)

df["日期"] = pd.to_datetime(
    df["日期"]
)

df["净销量"] = pd.to_numeric(
    df["净销量(千克)"],
    errors="coerce"
)

df["售价"] = pd.to_numeric(
    df["加权平均售价(元/千克)"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "日期",
        "分类名称",
        "净销量",
        "售价"
    ]
).copy()

df = df[
    (df["净销量"] > 0) &
    (df["售价"] > 0)
].copy()

# ============================================================
# 4. 第二步得到的价格弹性
# ============================================================

elasticity = {
    "花叶类": -0.8860,
    "花菜类": -0.9013,
    "水生根茎类": -0.3556,
    "茄类": -0.0727,
    "辣椒类": -0.5249,
    "食用菌": -0.7442
}

categories = [
    "花叶类",
    "花菜类",
    "水生根茎类",
    "茄类",
    "辣椒类",
    "食用菌"
]

# ============================================================
# 5. 构造完整日序列
# ============================================================

start_date = pd.Timestamp(
    "2020-07-01"
)

end_date = pd.Timestamp(
    "2023-06-30"
)

full_dates = pd.date_range(
    start_date,
    end_date,
    freq="D"
)

# ============================================================
# 6. 候选SARIMA模型
# ============================================================

candidate_models = [
    ((1, 0, 0), (1, 0, 0, 7)),
    ((1, 0, 1), (1, 0, 0, 7)),
    ((1, 0, 1), (1, 0, 1, 7)),
    ((2, 0, 1), (1, 0, 0, 7)),
    ((1, 1, 1), (1, 0, 1, 7))
]

# ============================================================
# 7. 初始化结果容器
# ============================================================

model_comparison = []
forecast_results = []
validation_results = []

best_models = {}

# ============================================================
# 8. 六大品类分别建立SARIMA模型
# ============================================================

for category in categories:

    print("\n" + "=" * 70)
    print(category)
    print("=" * 70)

    temp = df[
        df["分类名称"] == category
    ].copy()

    temp = (
        temp
        .sort_values("日期")
        .set_index("日期")
    )

    # --------------------------------------------------------
    # 8.1 补齐完整日期
    # --------------------------------------------------------

    temp = temp.reindex(
        full_dates
    )

    temp["分类名称"] = category

    temp["净销量"] = (
        temp["净销量"]
        .interpolate(method="time")
        .ffill()
        .bfill()
    )

    temp["售价"] = (
        temp["售价"]
        .interpolate(method="time")
        .ffill()
        .bfill()
    )

    # --------------------------------------------------------
    # 8.2 构造价格调整后的基础需求
    # --------------------------------------------------------

    beta = elasticity[
        category
    ]

    temp["lnQ"] = np.log(
        temp["净销量"]
    )

    temp["lnP"] = np.log(
        temp["售价"]
    )

    temp["基础需求Z"] = (
        temp["lnQ"]
        - beta * temp["lnP"]
    )

    z = temp[
        "基础需求Z"
    ].copy()

    # ========================================================
    # 9. 使用最后28天做验证
    # ========================================================

    validation_days = 28

    train = z.iloc[
        :-validation_days
    ]

    valid = z.iloc[
        -validation_days:
    ]

    best_spec = None

    # --------------------------------------------------------
    # 9.1 比较候选SARIMA模型
    # --------------------------------------------------------

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

            rmse = np.sqrt(
                mean_squared_error(
                    valid,
                    pred
                )
            )

            mae = mean_absolute_error(
                valid,
                pred
            )

            model_comparison.append(
                {
                    "分类名称": category,
                    "order": str(order),
                    "seasonal_order": str(
                        seasonal_order
                    ),
                    "AIC": result.aic,
                    "验证RMSE": rmse,
                    "验证MAE": mae
                }
            )

            print(
                "order =",
                order,
                " seasonal =",
                seasonal_order,
                " AIC =",
                round(
                    result.aic,
                    2
                ),
                " RMSE =",
                round(
                    rmse,
                    4
                )
            )

            if (
                best_spec is None
                or rmse < best_spec["RMSE"]
            ):

                best_spec = {
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "RMSE": rmse,
                    "MAE": mae,
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
    # 10. 最优模型重新拟合全部历史数据
    # ========================================================

    order = best_spec[
        "order"
    ]

    seasonal_order = best_spec[
        "seasonal_order"
    ]

    print("\n最佳模型：")
    print(
        "order =",
        order
    )

    print(
        "seasonal_order =",
        seasonal_order
    )

    print(
        "验证RMSE =",
        round(
            best_spec["RMSE"],
            4
        )
    )

    final_model = SARIMAX(
        z,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    final_result = final_model.fit(
        disp=False,
        maxiter=300
    )

    best_models[
        category
    ] = final_result

    # ========================================================
    # 11. 预测未来7天
    # ========================================================

    forecast_obj = (
        final_result
        .get_forecast(
            steps=7
        )
    )

    z_pred = (
        forecast_obj
        .predicted_mean
    )

    ci = (
        forecast_obj
        .conf_int(
            alpha=0.05
        )
    )

    future_dates = pd.date_range(
        "2023-07-01",
        periods=7,
        freq="D"
    )

    # --------------------------------------------------------
    # 11.1 转换回基础需求水平
    # --------------------------------------------------------

    base_index = np.exp(
        z_pred.values
    )

    lower_index = np.exp(
        ci.iloc[:, 0].values
    )

    upper_index = np.exp(
        ci.iloc[:, 1].values
    )

    # --------------------------------------------------------
    # 11.2 最近30日平均售价作为参考售价
    # --------------------------------------------------------

    recent_price = (
        temp["售价"]
        .iloc[-30:]
        .mean()
    )

    baseline_demand = (
        base_index
        * (
            recent_price ** beta
        )
    )

    lower_demand = (
        lower_index
        * (
            recent_price ** beta
        )
    )

    upper_demand = (
        upper_index
        * (
            recent_price ** beta
        )
    )

    # --------------------------------------------------------
    # 11.3 保存7天预测
    # --------------------------------------------------------

    for i in range(7):

        forecast_results.append(
            {
                "日期": future_dates[i],
                "分类名称": category,
                "参考售价(元/千克)": recent_price,
                "价格弹性β": beta,
                "预测基础需求参数Z":
                    z_pred.values[i],
                "预测基准销量(千克)":
                    baseline_demand[i],
                "95%下限(千克)":
                    lower_demand[i],
                "95%上限(千克)":
                    upper_demand[i]
            }
        )

    validation_results.append(
        {
            "分类名称": category,
            "最佳order": str(
                order
            ),
            "最佳seasonal_order": str(
                seasonal_order
            ),
            "验证RMSE":
                best_spec["RMSE"],
            "验证MAE":
                best_spec["MAE"],
            "AIC":
                final_result.aic
        }
    )

# ============================================================
# 12. 保存Excel结果
# ============================================================

forecast_df = pd.DataFrame(
    forecast_results
)

validation_df = pd.DataFrame(
    validation_results
)

comparison_df = pd.DataFrame(
    model_comparison
)

forecast_df.to_excel(
    output_dir
    / "2023年7月1-7日六大品类基础需求预测.xlsx",
    index=False
)

validation_df.to_excel(
    output_dir
    / "SARIMA最佳模型及验证结果.xlsx",
    index=False
)

comparison_df.to_excel(
    output_dir
    / "SARIMA候选模型比较.xlsx",
    index=False
)

# ============================================================
# 13. 打印预测结果
# ============================================================

print("\n")
print("=" * 80)
print("2023年7月1—7日基础需求预测")
print("=" * 80)

display_cols = [
    "日期",
    "分类名称",
    "参考售价(元/千克)",
    "预测基准销量(千克)"
]

print(
    forecast_df[
        display_cols
    ]
    .round(3)
    .to_string(
        index=False
    )
)

# ============================================================
# 14. 绘制未来7天基础需求预测图
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

    temp_plot = forecast_df[
        forecast_df[
            "分类名称"
        ] == category
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
        temp_plot[
            "95%下限(千克)"
        ],
        temp_plot[
            "95%上限(千克)"
        ],
        color=color,
        alpha=0.12,
        linewidth=0,
        label="95%预测区间"
    )

    # --------------------------------------------------------
    # 预测销量
    # --------------------------------------------------------

    ax.plot(
        temp_plot["日期"],
        temp_plot[
            "预测基准销量(千克)"
        ],
        color=color,
        linewidth=2.2,
        alpha=0.85,
        marker="o",
        markersize=6,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="预测销量",
        zorder=3
    )

    ax.set_title(
        category,
        fontsize=13,
        fontweight="bold",
        pad=8
    )

    ax.set_ylabel(
        "预测销量（千克）",
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
        color=grid_color,
        linestyle="--",
        linewidth=0.7,
        alpha=0.40
    )

    ax.set_axisbelow(
        True
    )

    ax.legend(
        fontsize=9,
        frameon=True,
        facecolor="white",
        edgecolor="#DDDDDD",
        framealpha=0.88
    )

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
        axis_color
    )

    ax.spines[
        "bottom"
    ].set_color(
        axis_color
    )

    ax.spines[
        "left"
    ].set_linewidth(
        0.8
    )

    ax.spines[
        "bottom"
    ].set_linewidth(
        0.8
    )

fig.suptitle(
    "2023年7月1—7日六大蔬菜品类基础需求预测",
    fontsize=18,
    fontweight="bold",
    y=0.98
)

plt.tight_layout(
    rect=[
        0,
        0,
        1,
        0.95
    ]
)

plt.savefig(
    output_dir
    / "六大品类未来7天基础需求预测_优化版.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

# ============================================================
# 15. 绘制模型验证误差图
#     低饱和度 + 更高透明度
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 6),
    facecolor="white"
)

ax.set_facecolor(
    background_color
)

x = np.arange(
    len(
        validation_df
    )
)

for i, row in validation_df.iterrows():

    category = row[
        "分类名称"
    ]

    value = row[
        "验证RMSE"
    ]

    color = category_colors[
        category
    ]

    # --------------------------------------------------------
    # 柱状图：
    # alpha调低，整体更加轻柔
    # --------------------------------------------------------

    ax.bar(
        i,
        value,
        width=0.65,
        color=color,
        alpha=0.55,
        edgecolor="white",
        linewidth=0.8
    )

    # --------------------------------------------------------
    # 数值标签
    # --------------------------------------------------------

    ax.text(
        i,
        value + 0.006,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#444444"
    )

# ------------------------------------------------------------
# 坐标轴设置
# ------------------------------------------------------------

ax.set_xticks(
    x
)

ax.set_xticklabels(
    validation_df[
        "分类名称"
    ],
    fontsize=10
)

ax.set_ylabel(
    "验证集 RMSE",
    fontsize=11
)

ax.set_xlabel(
    "蔬菜品类",
    fontsize=11
)

ax.set_title(
    "六大蔬菜品类基础需求预测模型验证误差",
    fontsize=16,
    fontweight="bold",
    pad=14
)

# ------------------------------------------------------------
# 网格
# ------------------------------------------------------------

ax.grid(
    axis="y",
    color=grid_color,
    linestyle="--",
    linewidth=0.8,
    alpha=0.40
)

ax.set_axisbelow(
    True
)

# ------------------------------------------------------------
# 边框
# ------------------------------------------------------------

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
    axis_color
)

ax.spines[
    "bottom"
].set_color(
    axis_color
)

ax.spines[
    "left"
].set_linewidth(
    0.8
)

ax.spines[
    "bottom"
].set_linewidth(
    0.8
)

# ------------------------------------------------------------
# 顶部留出空间给数字
# ------------------------------------------------------------

max_rmse = validation_df[
    "验证RMSE"
].max()

ax.set_ylim(
    0,
    max_rmse * 1.18
)

plt.tight_layout()

plt.savefig(
    output_dir
    / "需求预测模型验证误差_低饱和版.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

# ============================================================
# 16. 完成提示
# ============================================================

print("\n" + "=" * 70)
print("Step 3 完成")
print("=" * 70)

print(
    """
已生成：

1. 2023年7月1-7日六大品类基础需求预测.xlsx
2. SARIMA最佳模型及验证结果.xlsx
3. SARIMA候选模型比较.xlsx
4. 六大品类未来7天基础需求预测_优化版.png
5. 需求预测模型验证误差_低饱和版.png
"""
)