import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path

# ============================================================
# 1. 基本设置
# ============================================================

# Excel文件名
file_path = "C题_正确处理后建模数据.xlsx"

# 输出文件夹
output_dir = Path("问题二_Step1")
output_dir.mkdir(exist_ok=True)

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 2. 读取品类日数据
# ============================================================

df = pd.read_excel(
    file_path,
    sheet_name="品类日数据"
)

# 日期格式
df["日期"] = pd.to_datetime(df["日期"])

# 只保留分析所需字段
data = df[
    [
        "日期",
        "分类名称",
        "净销量(千克)",
        "加权平均售价(元/千克)",
        "加权平均批发价(元/千克)",
        "成本加成率",
        "损耗率"
    ]
].copy()

# 转换为数值
numeric_cols = [
    "净销量(千克)",
    "加权平均售价(元/千克)",
    "加权平均批发价(元/千克)",
    "成本加成率",
    "损耗率"
]

for col in numeric_cols:
    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )

# 去除分析变量缺失记录
data = data.dropna(
    subset=[
        "净销量(千克)",
        "成本加成率"
    ]
)

# ============================================================
# 3. 检查成本加成率计算是否正确
# ============================================================

data["重新计算成本加成率"] = (
    (
        data["加权平均售价(元/千克)"]
        - data["加权平均批发价(元/千克)"]
    )
    / data["加权平均批发价(元/千克)"]
)

data["加成率误差"] = (
    data["成本加成率"]
    - data["重新计算成本加成率"]
).abs()

print("=" * 60)
print("成本加成率检查")
print("=" * 60)

print(
    "最大计算误差：",
    data["加成率误差"].max()
)

# ============================================================
# 4. 成本加成率描述性统计
# ============================================================

stats = (
    data.groupby("分类名称")["成本加成率"]
    .agg(
        样本数="count",
        均值="mean",
        中位数="median",
        标准差="std",
        最小值="min",
        最大值="max"
    )
    .reset_index()
)

# 转换成百分数，便于阅读
for col in [
    "均值",
    "中位数",
    "标准差",
    "最小值",
    "最大值"
]:
    stats[col] = stats[col] * 100

stats = stats.rename(
    columns={
        "均值": "均值(%)",
        "中位数": "中位数(%)",
        "标准差": "标准差(%)",
        "最小值": "最小值(%)",
        "最大值": "最大值(%)"
    }
)

stats.to_excel(
    output_dir / "成本加成率描述性统计.xlsx",
    index=False
)

print("\n")
print("=" * 60)
print("六大品类成本加成率描述性统计")
print("=" * 60)
print(stats.to_string(index=False))

# ============================================================
# 5. Spearman相关性分析
# ============================================================

corr_results = []

categories = data["分类名称"].unique()

for category in categories:

    temp = data[
        data["分类名称"] == category
    ].copy()

    rho, p_value = spearmanr(
        temp["成本加成率"],
        temp["净销量(千克)"]
    )

    # 判断相关方向
    if rho > 0:
        direction = "正相关"
    elif rho < 0:
        direction = "负相关"
    else:
        direction = "无明显相关"

    # 判断相关程度
    abs_rho = abs(rho)

    if abs_rho < 0.2:
        strength = "极弱"
    elif abs_rho < 0.4:
        strength = "弱"
    elif abs_rho < 0.6:
        strength = "中等"
    elif abs_rho < 0.8:
        strength = "较强"
    else:
        strength = "强"

    # 显著性
    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"
    else:
        significance = "不显著"

    corr_results.append(
        {
            "分类名称": category,
            "Spearman相关系数": rho,
            "P值": p_value,
            "相关方向": direction,
            "相关程度": strength,
            "显著性": significance
        }
    )

corr_df = pd.DataFrame(corr_results)

corr_df = corr_df.sort_values(
    by="Spearman相关系数"
)

corr_df.to_excel(
    output_dir / "销量与成本加成率Spearman相关性.xlsx",
    index=False
)

print("\n")
print("=" * 60)
print("销量—成本加成率 Spearman 相关性")
print("=" * 60)
print(corr_df.to_string(index=False))

# ============================================================
# 6. 绘制六大品类散点图
# ============================================================

# 固定顺序
category_order = [
    "花叶类",
    "花菜类",
    "水生根茎类",
    "茄类",
    "辣椒类",
    "食用菌"
]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(16, 10)
)

axes = axes.flatten()

for ax, category in zip(
    axes,
    category_order
):

    temp = data[
        data["分类名称"] == category
    ].copy()

    x = temp["成本加成率"].values * 100
    y = temp["净销量(千克)"].values

    # ----------------------------
    # 散点
    # ----------------------------
    ax.scatter(
        x,
        y,
        s=18,
        alpha=0.35
    )

    # ----------------------------
    # 二次趋势拟合
    # ----------------------------
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x_valid = x[valid]
    y_valid = y[valid]

    if len(x_valid) >= 3:

        # 去掉极端加成率，仅用于画趋势线
        lower = np.percentile(
            x_valid,
            1
        )

        upper = np.percentile(
            x_valid,
            99
        )

        trend_mask = (
            (x_valid >= lower)
            & (x_valid <= upper)
        )

        x_fit = x_valid[trend_mask]
        y_fit = y_valid[trend_mask]

        if len(x_fit) >= 3:

            coeff = np.polyfit(
                x_fit,
                y_fit,
                2
            )

            poly = np.poly1d(coeff)

            x_line = np.linspace(
                x_fit.min(),
                x_fit.max(),
                200
            )

            y_line = poly(x_line)

            ax.plot(
                x_line,
                y_line,
                linewidth=2
            )

    # ----------------------------
    # Spearman系数
    # ----------------------------
    row = corr_df[
        corr_df["分类名称"] == category
    ].iloc[0]

    rho = row["Spearman相关系数"]
    p = row["P值"]

    if p < 0.001:
        p_text = "p < 0.001"
    else:
        p_text = f"p = {p:.3f}"

    ax.text(
        0.04,
        0.95,
        f"Spearman ρ = {rho:.3f}\n{p_text}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round",
            alpha=0.15
        )
    )

    # ----------------------------
    # 标题与坐标
    # ----------------------------
    ax.set_title(
        category,
        fontsize=13
    )

    ax.set_xlabel(
        "成本加成率（%）"
    )

    ax.set_ylabel(
        "日净销量（千克）"
    )

    ax.grid(
        alpha=0.2
    )

fig.suptitle(
    "六大蔬菜品类日销量与成本加成率关系",
    fontsize=17,
    y=0.98
)

plt.tight_layout(
    rect=[0, 0, 1, 0.96]
)

plt.savefig(
    output_dir / "六大品类销量与成本加成率关系.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 7. 再生成一张相关系数柱状图
# ============================================================

plot_corr = corr_df.copy()

fig, ax = plt.subplots(
    figsize=(10, 6)
)

bars = ax.bar(
    plot_corr["分类名称"],
    plot_corr["Spearman相关系数"]
)

ax.axhline(
    0,
    linewidth=1
)

ax.set_xlabel(
    "蔬菜品类"
)

ax.set_ylabel(
    "Spearman相关系数"
)

ax.set_title(
    "六大蔬菜品类销量与成本加成率相关程度"
)

# 数值标签
for bar, value in zip(
    bars,
    plot_corr["Spearman相关系数"]
):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:.3f}",
        ha="center",
        va="bottom" if value >= 0 else "top"
    )

ax.grid(
    axis="y",
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    output_dir / "销量与成本加成率相关系数.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 8. 保存本步骤使用的数据
# ============================================================

data.drop(
    columns=[
        "重新计算成本加成率",
        "加成率误差"
    ]
).to_excel(
    output_dir / "问题二_Step1分析数据.xlsx",
    index=False
)

print("\n")
print("=" * 60)
print("Step 1 完成")
print("=" * 60)

print(
    f"""
已生成以下文件：

1. {output_dir}/成本加成率描述性统计.xlsx
2. {output_dir}/销量与成本加成率Spearman相关性.xlsx
3. {output_dir}/六大品类销量与成本加成率关系.png
4. {output_dir}/销量与成本加成率相关系数.png
5. {output_dir}/问题二_Step1分析数据.xlsx
"""
)