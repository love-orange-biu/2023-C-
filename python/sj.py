import pandas as pd
import numpy as np

# ============================================================
# 1. 文件路径
# ============================================================
file1 = r"D:\python\附件1.xlsx"
file2 = r"D:\python\附件2.xlsx"
file3 = r"D:\python\附件3.xlsx"
file4 = r"D:\python\附件4.xlsx"

# ============================================================
# 2. 读取原始数据
# ============================================================
df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)
df3 = pd.read_excel(file3)
df4 = pd.read_excel(file4)

print("四个附件读取成功！")
print("附件1：", df1.shape)
print("附件2：", df2.shape)
print("附件3：", df3.shape)
print("附件4：", df4.shape)

# ============================================================
# 3. 统一列名：去除首尾空格
# ============================================================
for df in [df1, df2, df3, df4]:
    df.columns = df.columns.astype(str).str.strip()

print("\n附件1列名：", df1.columns.tolist())
print("附件2列名：", df2.columns.tolist())
print("附件3列名：", df3.columns.tolist())
print("附件4列名：", df4.columns.tolist())

# ============================================================
# 4. 通用检查函数
# ============================================================
def basic_check(df, name):
    result = []

    # 数据规模
    result.append({
        "数据表": name,
        "检查项目": "原始记录数",
        "检查结果": len(df),
        "备注": ""
    })

    result.append({
        "数据表": name,
        "检查项目": "字段数",
        "检查结果": df.shape[1],
        "备注": ""
    })

    # 完全重复行
    duplicate_count = df.duplicated().sum()

    result.append({
        "数据表": name,
        "检查项目": "完全重复记录数",
        "检查结果": int(duplicate_count),
        "备注": "完全相同的整行记录"
    })

    # 缺失值
    missing_total = df.isnull().sum().sum()

    result.append({
        "数据表": name,
        "检查项目": "缺失值总数",
        "检查结果": int(missing_total),
        "备注": ""
    })

    for col in df.columns:
        miss = df[col].isnull().sum()

        if miss > 0:
            result.append({
                "数据表": name,
                "检查项目": f"{col}缺失值",
                "检查结果": int(miss),
                "备注": f"缺失率={miss / len(df):.4%}"
            })

    return result


all_results = []

all_results.extend(basic_check(df1, "附件1"))
all_results.extend(basic_check(df2, "附件2"))
all_results.extend(basic_check(df3, "附件3"))
all_results.extend(basic_check(df4, "附件4"))

# ============================================================
# 5. 附件1：商品信息检查
# ============================================================

# 自动寻找编码相关列
code_cols1 = [c for c in df1.columns if "编码" in c]

if len(code_cols1) > 0:
    item_code_col1 = code_cols1[0]

    duplicated_code1 = df1[item_code_col1].duplicated().sum()

    all_results.append({
        "数据表": "附件1",
        "检查项目": "单品编码重复数",
        "检查结果": int(duplicated_code1),
        "备注": "用于检查单品编码唯一性"
    })

# ============================================================
# 6. 附件2：销售流水检查
# ============================================================

# 自动识别关键字段
sales_col = None
price_col = None
type_col = None
discount_col = None
date_col2 = None
item_code_col2 = None

for col in df2.columns:

    if "销量" in col:
        sales_col = col

    if "销售单价" in col or ("单价" in col and "批发" not in col):
        price_col = col

    if "销售类型" in col:
        type_col = col

    if "折扣" in col:
        discount_col = col

    if "销售日期" in col or col == "日期":
        date_col2 = col

    if "单品编码" in col:
        item_code_col2 = col

# 销量检查
if sales_col is not None:

    sales_numeric = pd.to_numeric(df2[sales_col], errors="coerce")

    negative_sales = (sales_numeric < 0).sum()
    zero_sales = (sales_numeric == 0).sum()

    all_results.append({
        "数据表": "附件2",
        "检查项目": "负销量记录数",
        "检查结果": int(negative_sales),
        "备注": "需要结合销售类型判断是否为退货"
    })

    all_results.append({
        "数据表": "附件2",
        "检查项目": "零销量记录数",
        "检查结果": int(zero_sales),
        "备注": ""
    })

    # IQR识别高销量离群值
    q1 = sales_numeric.quantile(0.25)
    q3 = sales_numeric.quantile(0.75)
    iqr = q3 - q1

    upper_limit = q3 + 1.5 * iqr

    high_outlier = (sales_numeric > upper_limit).sum()

    all_results.append({
        "数据表": "附件2",
        "检查项目": "IQR高销量离群记录数",
        "检查结果": int(high_outlier),
        "备注": f"IQR上界={upper_limit:.4f}，仅标记不直接删除"
    })

# 销售价格检查
if price_col is not None:

    price_numeric = pd.to_numeric(df2[price_col], errors="coerce")

    zero_price = (price_numeric == 0).sum()
    negative_price = (price_numeric < 0).sum()

    all_results.append({
        "数据表": "附件2",
        "检查项目": "销售单价为0记录数",
        "检查结果": int(zero_price),
        "备注": ""
    })

    all_results.append({
        "数据表": "附件2",
        "检查项目": "销售单价为负记录数",
        "检查结果": int(negative_price),
        "备注": ""
    })

# 退货类型一致性检查
if sales_col is not None and type_col is not None:

    negative_df = df2[
        pd.to_numeric(df2[sales_col], errors="coerce") < 0
    ]

    return_type_count = negative_df[type_col].astype(str).str.contains(
        "退货",
        na=False
    ).sum()

    all_results.append({
        "数据表": "附件2",
        "检查项目": "负销量中标记为退货的记录数",
        "检查结果": int(return_type_count),
        "备注": "用于检查负销量是否属于正常退货"
    })

# 折扣记录检查
if discount_col is not None:

    discount_values = df2[discount_col].astype(str)

    discount_count = discount_values.str.contains(
        "是|折扣",
        regex=True,
        na=False
    ).sum()

    all_results.append({
        "数据表": "附件2",
        "检查项目": "折扣销售记录数",
        "检查结果": int(discount_count),
        "备注": "折扣属于正常经营行为，一般保留"
    })

# 日期范围检查
if date_col2 is not None:

    dates = pd.to_datetime(df2[date_col2], errors="coerce")

    all_results.append({
        "数据表": "附件2",
        "检查项目": "最早销售日期",
        "检查结果": dates.min(),
        "备注": ""
    })

    all_results.append({
        "数据表": "附件2",
        "检查项目": "最晚销售日期",
        "检查结果": dates.max(),
        "备注": ""
    })

# ============================================================
# 7. 附件3：批发价格检查
# ============================================================

wholesale_col = None
date_col3 = None
item_code_col3 = None

for col in df3.columns:

    if "批发价格" in col or "批发价" in col:
        wholesale_col = col

    if "日期" in col:
        date_col3 = col

    if "单品编码" in col:
        item_code_col3 = col

if wholesale_col is not None:

    wholesale_numeric = pd.to_numeric(
        df3[wholesale_col],
        errors="coerce"
    )

    all_results.append({
        "数据表": "附件3",
        "检查项目": "批发价格为0记录数",
        "检查结果": int((wholesale_numeric == 0).sum()),
        "备注": ""
    })

    all_results.append({
        "数据表": "附件3",
        "检查项目": "批发价格为负记录数",
        "检查结果": int((wholesale_numeric < 0).sum()),
        "备注": ""
    })

# 日期+单品重复
if date_col3 is not None and item_code_col3 is not None:

    duplicated_wholesale = df3.duplicated(
        subset=[date_col3, item_code_col3]
    ).sum()

    all_results.append({
        "数据表": "附件3",
        "检查项目": "日期-单品重复记录数",
        "检查结果": int(duplicated_wholesale),
        "备注": "同一单品同一天原则上应只有一个批发价格"
    })

# ============================================================
# 8. 附件4：损耗率检查
# ============================================================

loss_col = None
item_code_col4 = None

for col in df4.columns:

    if "损耗率" in col:
        loss_col = col

    if "单品编码" in col:
        item_code_col4 = col

if loss_col is not None:

    # 去掉可能存在的%
    loss_series = (
        df4[loss_col]
        .astype(str)
        .str.replace("%", "", regex=False)
    )

    loss_numeric = pd.to_numeric(
        loss_series,
        errors="coerce"
    )

    all_results.append({
        "数据表": "附件4",
        "检查项目": "损耗率无法转为数值记录数",
        "检查结果": int(loss_numeric.isnull().sum()),
        "备注": ""
    })

    all_results.append({
        "数据表": "附件4",
        "检查项目": "负损耗率记录数",
        "检查结果": int((loss_numeric < 0).sum()),
        "备注": ""
    })

    all_results.append({
        "数据表": "附件4",
        "检查项目": "损耗率大于100%记录数",
        "检查结果": int((loss_numeric > 100).sum()),
        "备注": ""
    })

# ============================================================
# 9. 单品编码匹配检查
# ============================================================

if (
    len(code_cols1) > 0
    and item_code_col2 is not None
):

    code_set1 = set(
        df1[item_code_col1]
        .astype(str)
        .str.strip()
    )

    code2 = df2[item_code_col2].astype(str).str.strip()

    unmatched_sales = ~code2.isin(code_set1)

    all_results.append({
        "数据表": "附件1-附件2",
        "检查项目": "销售流水单品编码匹配失败记录数",
        "检查结果": int(unmatched_sales.sum()),
        "备注": "附件2中的单品编码无法在附件1找到"
    })

if (
    len(code_cols1) > 0
    and item_code_col3 is not None
):

    code_set1 = set(
        df1[item_code_col1]
        .astype(str)
        .str.strip()
    )

    code3 = df3[item_code_col3].astype(str).str.strip()

    unmatched_cost = ~code3.isin(code_set1)

    all_results.append({
        "数据表": "附件1-附件3",
        "检查项目": "批发价格单品编码匹配失败记录数",
        "检查结果": int(unmatched_cost.sum()),
        "备注": "附件3中的单品编码无法在附件1找到"
    })

# ============================================================
# 10. 销售数据与批发价格匹配检查
# ============================================================

if (
    date_col2 is not None
    and item_code_col2 is not None
    and date_col3 is not None
    and item_code_col3 is not None
):

    sales_key = df2[
        [date_col2, item_code_col2]
    ].copy()

    wholesale_key = df3[
        [date_col3, item_code_col3]
    ].copy()

    sales_key["日期统一"] = pd.to_datetime(
        sales_key[date_col2],
        errors="coerce"
    ).dt.date

    wholesale_key["日期统一"] = pd.to_datetime(
        wholesale_key[date_col3],
        errors="coerce"
    ).dt.date

    sales_key["编码统一"] = (
        sales_key[item_code_col2]
        .astype(str)
        .str.strip()
    )

    wholesale_key["编码统一"] = (
        wholesale_key[item_code_col3]
        .astype(str)
        .str.strip()
    )

    wholesale_pairs = set(
        zip(
            wholesale_key["日期统一"],
            wholesale_key["编码统一"]
        )
    )

    sales_pairs = list(
        zip(
            sales_key["日期统一"],
            sales_key["编码统一"]
        )
    )

    unmatched_price = sum(
        pair not in wholesale_pairs
        for pair in sales_pairs
    )

    all_results.append({
        "数据表": "附件2-附件3",
        "检查项目": "销售流水无法匹配当日批发价格记录数",
        "检查结果": int(unmatched_price),
        "备注": "按照日期+单品编码匹配"
    })

# ============================================================
# 11. 生成数据质量检查表
# ============================================================

quality_df = pd.DataFrame(all_results)

quality_df.to_excel(
    "数据质量检查表.xlsx",
    index=False
)

print("\n==============================")
print("数据质量检查完成！")
print("==============================")

print(quality_df.to_string(index=False))

print("\n已生成：数据质量检查表.xlsx")

# ============================================================
# 12. 同时保存异常明细
# ============================================================

with pd.ExcelWriter(
    "数据质量异常明细.xlsx",
    engine="openpyxl"
) as writer:

    # 附件2负销量
    if sales_col is not None:
        df2[
            pd.to_numeric(df2[sales_col], errors="coerce") < 0
        ].to_excel(
            writer,
            sheet_name="负销量记录",
            index=False
        )

    # 附件2零销量
    if sales_col is not None:
        df2[
            pd.to_numeric(df2[sales_col], errors="coerce") == 0
        ].to_excel(
            writer,
            sheet_name="零销量记录",
            index=False
        )

    # 附件2异常高销量
    if sales_col is not None:
        df2[
            pd.to_numeric(df2[sales_col], errors="coerce") > upper_limit
        ].to_excel(
            writer,
            sheet_name="高销量离群记录",
            index=False
        )

    # 附件2价格异常
    if price_col is not None:
        df2[
            pd.to_numeric(df2[price_col], errors="coerce") <= 0
        ].to_excel(
            writer,
            sheet_name="销售价格异常",
            index=False
        )

    # 附件3批发价格异常
    if wholesale_col is not None:
        df3[
            pd.to_numeric(
                df3[wholesale_col],
                errors="coerce"
            ) <= 0
        ].to_excel(
            writer,
            sheet_name="批发价格异常",
            index=False
        )

print("已生成：数据质量异常明细.xlsx")