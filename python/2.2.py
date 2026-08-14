import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
from pathlib import Path

# ==========================================================
# 1. 基本设置
# ==========================================================

file_path = "C题_正确处理后建模数据.xlsx"

output_dir = Path("问题二_Step2")
output_dir.mkdir(exist_ok=True)

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ==========================================================
# 2. 低饱和度配色
# ==========================================================

category_colors = {
    "花叶类": "#7FA7D8",       # 低饱和蓝
    "花菜类": "#E6A26A",       # 低饱和橙
    "水生根茎类": "#79B894",   # 低饱和绿
    "茄类": "#D98582",         # 低饱和红
    "辣椒类": "#9B8AC4",       # 低饱和紫
    "食用菌": "#AE8F70"        # 低饱和棕
}

grid_color = "#D9D9D9"
axis_color = "#666666"
text_color = "#333333"

# ==========================================================
# 3. 读取数据
# ==========================================================

df = pd.read_excel(
    file_path,
    sheet_name="品类日数据"
)

df["日期"] = pd.to_datetime(df["日期"])

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
        "净销量",
        "售价",
        "分类名称",
        "日期"
    ]
).copy()

df = df[
    (df["净销量"] > 0) &
    (df["售价"] > 0)
].copy()

# ==========================================================
# 4. 构造变量
# ==========================================================

df["lnQ"] = np.log(df["净销量"])
df["lnP"] = np.log(df["售价"])

start_date = df["日期"].min()

df["时间趋势"] = (
    df["日期"] - start_date
).dt.days

df["星期"] = df["日期"].dt.dayofweek
df["月份"] = df["日期"].dt.month

# ==========================================================
# 5. 三种需求模型
# ==========================================================

formula_A = "lnQ ~ lnP"

formula_B = "lnQ ~ lnP + C(星期)"

formula_C = (
    "lnQ ~ lnP + 时间趋势 + "
    "C(星期) + C(月份)"
)

categories = [
    "花叶类",
    "花菜类",
    "水生根茎类",
    "茄类",
    "辣椒类",
    "食用菌"
]

comparison_results = []
elasticity_results = []

all_models = {}

# ==========================================================
# 6. 六个品类分别建模
# ==========================================================

for category in categories:

    temp = df[
        df["分类名称"] == category
    ].copy()

    print("\n" + "=" * 70)
    print(category)
    print("=" * 70)

    # 模型A
    model_A = smf.ols(
        formula_A,
        data=temp
    ).fit(cov_type="HC3")

    # 模型B
    model_B = smf.ols(
        formula_B,
        data=temp
    ).fit(cov_type="HC3")

    # 模型C
    model_C = smf.ols(
        formula_C,
        data=temp
    ).fit(cov_type="HC3")

    all_models[category] = {
        "A": model_A,
        "B": model_B,
        "C": model_C
    }

    # ------------------------------------------------------
    # 模型评价
    # ------------------------------------------------------

    for name, model in [
        ("模型A：仅价格", model_A),
        ("模型B：价格+星期", model_B),
        ("模型C：价格+星期+月份+趋势", model_C)
    ]:

        pred_log = model.predict(temp)

        rmse_log = np.sqrt(
            mean_squared_error(
                temp["lnQ"],
                pred_log
            )
        )

        comparison_results.append(
            {
                "分类名称": category,
                "模型": name,
                "R²": model.rsquared,
                "调整R²": model.rsquared_adj,
                "AIC": model.aic,
                "BIC": model.bic,
                "RMSE(log)": rmse_log,
                "价格弹性β": model.params.get(
                    "lnP",
                    np.nan
                ),
                "价格弹性P值": model.pvalues.get(
                    "lnP",
                    np.nan
                )
            }
        )

    # ------------------------------------------------------
    # 完整模型价格弹性
    # ------------------------------------------------------

    beta = model_C.params["lnP"]
    p_value = model_C.pvalues["lnP"]

    ci = model_C.conf_int().loc["lnP"]

    elasticity_results.append(
        {
            "分类名称": category,
            "价格弹性β": beta,
            "P值": p_value,
            "95%CI下限": ci.iloc[0],
            "95%CI上限": ci.iloc[1],
            "R²": model_C.rsquared,
            "调整R²": model_C.rsquared_adj,
            "AIC": model_C.aic
        }
    )

    print(
        f"完整模型价格弹性 β = {beta:.4f}"
    )

    print(
        f"P值 = {p_value:.6f}"
    )

    print(
        f"调整R² = {model_C.rsquared_adj:.4f}"
    )

# ==========================================================
# 7. 保存模型比较结果
# ==========================================================

comparison_df = pd.DataFrame(
    comparison_results
)

comparison_df.to_excel(
    output_dir / "三种需求模型比较.xlsx",
    index=False
)

# ==========================================================
# 8. 保存价格弹性结果
# ==========================================================

elasticity_df = pd.DataFrame(
    elasticity_results
)

elasticity_df.to_excel(
    output_dir / "六大品类价格弹性估计.xlsx",
    index=False
)

print("\n")
print("=" * 70)
print("六大品类完整模型价格弹性")
print("=" * 70)

print(
    elasticity_df.round(4).to_string(index=False)
)

# ==========================================================
# 9. 绘制价格弹性图
# ==========================================================

plot_df = elasticity_df.copy()

fig, ax = plt.subplots(
    figsize=(12, 6.5),
    facecolor="white"
)

ax.set_facecolor("#FAFAFA")

x = np.arange(
    len(plot_df)
)

for i, row in plot_df.iterrows():

    category = row["分类名称"]
    beta = row["价格弹性β"]

    lower = beta - row["95%CI下限"]
    upper = row["95%CI上限"] - beta

    color = category_colors[category]

    ax.errorbar(
        i,
        beta,
        yerr=np.array([[lower], [upper]]),
        fmt="o",
        color=color,
        ecolor=color,
        markersize=9,
        capsize=6,
        capthick=1.8,
        elinewidth=1.8,
        markeredgecolor="white",
        markeredgewidth=1.0,
        zorder=3
    )

    # 数值标签
    ax.text(
        i + 0.05,
        beta,
        f"{beta:.3f}",
        fontsize=10,
        color=color,
        va="center",
        fontweight="bold"
    )

# 0线
ax.axhline(
    0,
    color="#8A8A8A",
    linestyle="--",
    linewidth=1.2,
    alpha=0.8
)

# 单位弹性 -1
ax.axhline(
    -1,
    color="#A8A8A8",
    linestyle=":",
    linewidth=1.2,
    alpha=0.9
)

ax.set_xticks(x)

ax.set_xticklabels(
    plot_df["分类名称"],
    fontsize=11
)

ax.set_ylabel(
    "价格弹性系数 β",
    fontsize=12
)

ax.set_xlabel(
    "蔬菜品类",
    fontsize=12
)

ax.set_title(
    "六大蔬菜品类价格弹性估计及95%置信区间",
    fontsize=17,
    fontweight="bold",
    pad=18
)

ax.grid(
    axis="y",
    color=grid_color,
    linestyle="--",
    linewidth=0.8,
    alpha=0.55
)

# 边框
for spine in [
    "top",
    "right"
]:
    ax.spines[spine].set_visible(False)

for spine in [
    "left",
    "bottom"
]:
    ax.spines[spine].set_color(axis_color)

plt.tight_layout()

plt.savefig(
    output_dir / "六大品类价格弹性估计_优化版.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

# ==========================================================
# 10. 实际销量与模型拟合值比较
# ==========================================================

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

    temp = df[
        df["分类名称"] == category
    ].copy()

    temp = temp.sort_values(
        "日期"
    )

    model = all_models[category]["C"]

    pred_log = model.predict(temp)

    temp["预测销量"] = np.exp(pred_log)

    # ----------------------------------------------
    # 30日移动平均
    # ----------------------------------------------

    temp_indexed = temp.set_index(
        "日期"
    )

    actual_ma = (
        temp_indexed["净销量"]
        .rolling(
            30,
            min_periods=1
        )
        .mean()
    )

    pred_ma = (
        temp_indexed["预测销量"]
        .rolling(
            30,
            min_periods=1
        )
        .mean()
    )

    color = category_colors[
        category
    ]

    ax.set_facecolor(
        "#FAFAFA"
    )

    # 实际销量：浅色实线
    ax.plot(
        actual_ma.index,
        actual_ma.values,
        color=color,
        linewidth=1.7,
        alpha=0.65,
        label="实际销量"
    )

    # 模型拟合：同色深一点虚线
    ax.plot(
        pred_ma.index,
        pred_ma.values,
        color=color,
        linewidth=2.2,
        linestyle="--",
        alpha=1.0,
        label="模型拟合"
    )

    ax.set_title(
        category,
        fontsize=13,
        fontweight="bold",
        pad=8
    )

    ax.set_ylabel(
        "30日平均销量（千克）",
        fontsize=10
    )

    ax.grid(
        color=grid_color,
        linestyle="--",
        linewidth=0.7,
        alpha=0.5
    )

    ax.legend(
        fontsize=9,
        frameon=True,
        facecolor="white",
        edgecolor="#DDDDDD"
    )

    # 去掉上、右边框
    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.spines[
        "left"
    ].set_color(axis_color)

    ax.spines[
        "bottom"
    ].set_color(axis_color)

# 底部两行统一日期标签
for ax in axes[3:]:
    ax.set_xlabel(
        "日期",
        fontsize=10
    )

fig.suptitle(
    "六大蔬菜品类需求模型拟合效果（30日移动平均）",
    fontsize=18,
    fontweight="bold",
    y=0.98
)

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)

plt.savefig(
    output_dir / "六大品类需求模型拟合效果_优化版.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

# ==========================================================
# 11. 保存完整回归结果
# ==========================================================

with open(
    output_dir / "完整回归结果.txt",
    "w",
    encoding="utf-8"
) as f:

    for category in categories:

        f.write(
            "\n" + "=" * 80 + "\n"
        )

        f.write(
            category + "\n"
        )

        f.write(
            "=" * 80 + "\n"
        )

        f.write(
            all_models[
                category
            ]["C"]
            .summary()
            .as_text()
        )

        f.write("\n")

# ==========================================================
# 12. 运行完成提示
# ==========================================================

print("\n" + "=" * 70)
print("Step 2 完成")
print("=" * 70)

print(
    """
已生成：

1. 三种需求模型比较.xlsx
2. 六大品类价格弹性估计.xlsx
3. 六大品类价格弹性估计_优化版.png
4. 六大品类需求模型拟合效果_优化版.png
5. 完整回归结果.txt
"""
)