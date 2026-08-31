
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Home Depot RTV Dashboard",
    page_icon="↩️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ORDERS_FILE = DATA_DIR / "orders_2026_ytd.csv"
MAPPING_FILE = DATA_DIR / "sku_mapping.xlsx"
RTV_FILE = DATA_DIR / "rtv_2026.xlsx"

# ----------------------------
# Basic helpers
# ----------------------------
def norm_text(series):
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
    )

def safe_divide(a, b):
    return np.where(b > 0, a / b, np.nan)

# ----------------------------
# Sales order loader
# Same logic as the existing Sales Dashboard V11
# ----------------------------
def read_orders_file(path):
    def _read(skiprows):
        return pd.read_csv(path, skiprows=skiprows, dtype=str)

    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        first_lines = [f.readline() for _ in range(6)]

    preview = "".join(first_lines)
    looks_like_hd_raw = (
        "Search Name:" in preview
        or "Exported at" in preview
        or (first_lines and "Line Status" not in first_lines[0])
    )

    if looks_like_hd_raw:
        df = _read(4)
    else:
        try:
            df = _read(0)
        except pd.errors.ParserError:
            df = _read(4)

    if "Order Date" not in df.columns or "Vendor SKU" not in df.columns:
        df = _read(4)

    required = ["Order Date", "Vendor SKU", "Quantity", "Unit Cost"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"订单文件缺少字段: {missing}")

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df["Unit Cost"] = pd.to_numeric(df["Unit Cost"], errors="coerce").fillna(0)

    df = df[
        df["Order Date"].notna()
        & (df["Order Date"] >= pd.Timestamp("2026-01-01"))
    ].copy()

    if "Line Status" in df.columns:
        line_status = norm_text(df["Line Status"])
        df = df[line_status.ne("CANCELLED") | line_status.isna()].copy()

    if "Unit Cost Currency" in df.columns:
        currency = norm_text(df["Unit Cost Currency"])
        df = df[~currency.str.contains("USD", na=False)].copy()

    return df

# ----------------------------
# Historical mapping
# Vendor SKU = MFG Model #
# ----------------------------
def map_orders_to_company_sku(orders, mapping):
    orders = orders.copy()
    mapping = mapping.copy()

    orders["_row_id"] = np.arange(len(orders))
    orders["_vendor_key"] = norm_text(orders["Vendor SKU"])

    required_mapping = ["公司SKU", "MFG Model #"]
    missing = [c for c in required_mapping if c not in mapping.columns]
    if missing:
        raise ValueError(f"Mapping文件缺少字段: {missing}")

    mapping["_vendor_key"] = norm_text(mapping["MFG Model #"])

    if "Valid From" not in mapping.columns:
        mapping["Valid From"] = pd.NaT
    if "Valid To" not in mapping.columns:
        mapping["Valid To"] = pd.NaT

    mapping["Valid From"] = pd.to_datetime(mapping["Valid From"], errors="coerce")
    mapping["Valid To"] = pd.to_datetime(mapping["Valid To"], errors="coerce")

    optional_cols = ["Brand", "产品名称", "OMSID"]
    for col in optional_cols:
        if col not in mapping.columns:
            mapping[col] = ""

    candidates = orders[["_row_id", "_vendor_key", "Order Date"]].merge(
        mapping[
            [
                "_vendor_key", "公司SKU", "Brand", "产品名称",
                "MFG Model #", "OMSID", "Valid From", "Valid To"
            ]
        ],
        on="_vendor_key",
        how="left",
    )

    valid = candidates[
        candidates["MFG Model #"].notna()
        & (candidates["Valid From"].isna() | (candidates["Order Date"] >= candidates["Valid From"]))
        & (candidates["Valid To"].isna() | (candidates["Order Date"] <= candidates["Valid To"]))
    ].copy()

    if len(valid):
        valid["_specificity"] = (
            valid["Valid From"].notna().astype(int)
            + valid["Valid To"].notna().astype(int)
        )
        valid["_from_rank"] = valid["Valid From"].fillna(pd.Timestamp("1900-01-01"))

        valid = (
            valid.sort_values(
                ["_row_id", "_specificity", "_from_rank"],
                ascending=[True, False, False],
            )
            .drop_duplicates("_row_id", keep="first")
        )

        chosen = valid[
            ["_row_id", "公司SKU", "Brand", "产品名称", "MFG Model #", "OMSID"]
        ]
    else:
        chosen = pd.DataFrame(
            columns=["_row_id", "公司SKU", "Brand", "产品名称", "MFG Model #", "OMSID"]
        )

    result = orders.merge(chosen, on="_row_id", how="left")
    result["公司SKU"] = result["公司SKU"].fillna("未匹配")
    result["Brand"] = result["Brand"].fillna("未匹配")
    result["产品名称"] = result["产品名称"].fillna("")
    return result

# ----------------------------
# RTV loader
# Main rule:
# Return month is based on Order Date, not RTV Date
# ----------------------------
def read_rtv_file(path):
    df = pd.read_excel(path, sheet_name="2026-RTV", dtype=str)

    # Compatible aliases in case the export changes slightly
    alias_groups = {
        "RTV Date": ["RTV Date", "RTV DATE", "RTV日期"],
        "Order Date": ["Order Date", "ORDER DATE", "订单日期"],
        "产品SKU": ["产品SKU", "公司SKU", "Product SKU"],
        "QTY": ["QTY", "Qty", "Quantity", "退货数量"],
    }

    rename_map = {}
    columns_upper = {str(c).strip().upper(): c for c in df.columns}

    for target, aliases in alias_groups.items():
        if target in df.columns:
            continue
        for alias in aliases:
            key = alias.strip().upper()
            if key in columns_upper:
                rename_map[columns_upper[key]] = target
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    required = ["Order Date", "产品SKU", "QTY"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"2026-RTV Sheet缺少字段: {missing}。当前字段为: {list(df.columns)}"
        )

    if "RTV Date" in df.columns:
        df["RTV Date"] = pd.to_datetime(df["RTV Date"], errors="coerce")

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["QTY"] = pd.to_numeric(df["QTY"], errors="coerce").fillna(0)

    for col in ["UNIT COST", "Unit Cost", "Total Cost", "10%运费", "总扣款"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["产品SKU"] = df["产品SKU"].astype("string").str.strip()

    df = df[
        df["Order Date"].notna()
        & (df["Order Date"] >= pd.Timestamp("2026-01-01"))
        & df["产品SKU"].notna()
        & (df["产品SKU"] != "")
        & (df["QTY"] != 0)
    ].copy()

    return df


# ----------------------------
# RTV Alias
# Used for products shipped as two cartons (-I / -O) but analyzed
# as one company SKU.
# ----------------------------
def read_rtv_alias(path):
    try:
        alias = pd.read_excel(path, sheet_name="RTV Alias", dtype=str)
    except Exception:
        return pd.DataFrame(columns=["RTV 产品SKU", "PART#", "公司SKU"])

    required = ["RTV 产品SKU", "公司SKU"]
    if any(c not in alias.columns for c in required):
        return pd.DataFrame(columns=["RTV 产品SKU", "PART#", "公司SKU"])

    if "PART#" not in alias.columns:
        alias["PART#"] = ""

    alias = alias[["RTV 产品SKU", "PART#", "公司SKU"]].copy()
    alias["RTV 产品SKU"] = norm_text(alias["RTV 产品SKU"])
    alias["PART#"] = norm_text(alias["PART#"])
    alias["公司SKU"] = alias["公司SKU"].astype("string").str.strip()

    alias = alias[
        alias["RTV 产品SKU"].notna()
        & (alias["RTV 产品SKU"] != "")
        & alias["公司SKU"].notna()
        & (alias["公司SKU"] != "")
    ].drop_duplicates(["RTV 产品SKU", "PART#"], keep="last")

    return alias


def apply_rtv_alias(rtv, alias):
    rtv = rtv.copy()
    rtv["原始产品SKU"] = rtv["产品SKU"]

    if alias is None or alias.empty:
        return rtv

    rtv["_rtv_sku_key"] = norm_text(rtv["产品SKU"])
    if "PART#" in rtv.columns:
        rtv["_part_key"] = norm_text(rtv["PART#"])
    else:
        rtv["_part_key"] = ""

    # First priority: exact Product SKU + PART# alias.
    exact_alias = alias[alias["PART#"].notna() & (alias["PART#"] != "")].copy()
    exact_alias = exact_alias.rename(
        columns={
            "RTV 产品SKU": "_rtv_sku_key",
            "PART#": "_part_key",
            "公司SKU": "_alias_company_sku_exact",
        }
    )
    if len(exact_alias):
        rtv = rtv.merge(
            exact_alias[["_rtv_sku_key", "_part_key", "_alias_company_sku_exact"]],
            on=["_rtv_sku_key", "_part_key"],
            how="left",
        )
    else:
        rtv["_alias_company_sku_exact"] = pd.NA

    # Second priority: Product SKU-only alias, useful if PART# is blank.
    sku_alias = (
        alias.sort_values("PART#")
        .drop_duplicates("RTV 产品SKU", keep="first")
        .rename(
            columns={
                "RTV 产品SKU": "_rtv_sku_key",
                "公司SKU": "_alias_company_sku_sku",
            }
        )
    )
    rtv = rtv.merge(
        sku_alias[["_rtv_sku_key", "_alias_company_sku_sku"]],
        on="_rtv_sku_key",
        how="left",
    )

    mapped = rtv["_alias_company_sku_exact"].fillna(
        rtv["_alias_company_sku_sku"]
    )
    rtv["产品SKU"] = mapped.fillna(rtv["产品SKU"])

    rtv = rtv.drop(
        columns=[
            "_rtv_sku_key",
            "_part_key",
            "_alias_company_sku_exact",
            "_alias_company_sku_sku",
        ],
        errors="ignore",
    )
    return rtv


# ----------------------------
# Load
# ----------------------------
def load_all():
    # Always read the latest GitHub data files on every rerun.
    orders = read_orders_file(ORDERS_FILE)
    mapping = pd.read_excel(MAPPING_FILE, sheet_name="Sheet1", dtype=str)
    sales = map_orders_to_company_sku(orders, mapping)

    rtv = read_rtv_file(RTV_FILE)
    rtv_alias = read_rtv_alias(MAPPING_FILE)
    rtv = apply_rtv_alias(rtv, rtv_alias)

    return sales, rtv, mapping

# ----------------------------
# File checks
# ----------------------------
st.title("↩️ Home Depot RTV Dashboard")
st.caption(
    "核心口径：所有退货均按 Order Date 归属月份，而不是按 RTV Date；"
    "销售分母与 Sales Dashboard 共用 orders_2026_ytd.csv；"
    "分体空调 I/O 两箱通过 sku_mapping.xlsx 的 RTV Alias 归并到同一公司SKU。"
)

missing_files = [
    p.name for p in [ORDERS_FILE, MAPPING_FILE, RTV_FILE]
    if not p.exists()
]
if missing_files:
    st.error("data 文件夹缺少以下文件：" + "、".join(missing_files))
    st.stop()

try:
    sales_raw, rtv_raw, mapping = load_all()
except Exception as e:
    st.error("数据读取失败")
    st.exception(e)
    st.stop()

# ----------------------------
# Period
# ----------------------------
latest_order_date = sales_raw["Order Date"].max()
latest_month = latest_order_date.to_period("M")
months = pd.period_range("2026-01", latest_month, freq="M")

sales_raw["Order Month"] = sales_raw["Order Date"].dt.to_period("M")
rtv_raw["Order Month"] = rtv_raw["Order Date"].dt.to_period("M")

# Sales denominator only uses mapped company SKU
sales = sales_raw[sales_raw["公司SKU"] != "未匹配"].copy()
rtv = rtv_raw.copy()

# Mapping metadata
meta_cols = [c for c in ["公司SKU", "Brand", "产品名称"] if c in mapping.columns]
meta = mapping[meta_cols].copy()
if "公司SKU" in meta.columns:
    meta["公司SKU"] = meta["公司SKU"].astype("string").str.strip()
    meta = meta.dropna(subset=["公司SKU"]).drop_duplicates("公司SKU", keep="first")

def normalize_company_sku(series):
    return (
        series.astype("string")
        .str.replace("\u00a0", " ", regex=False)
        .str.replace("\u3000", " ", regex=False)
        .str.strip()
        .str.upper()
    )

# 核心关系：RTV 产品SKU = Mapping 公司SKU
known_skus = (
    set(normalize_company_sku(meta["公司SKU"]).dropna())
    if "公司SKU" in meta.columns
    else set()
)

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.header("筛选")

if "Brand" in meta.columns:
    brands = sorted(meta["Brand"].dropna().astype(str).unique().tolist())
else:
    brands = []

selected_brand = st.sidebar.selectbox("Brand", ["全部"] + brands)

all_skus = sorted(
    set(sales["公司SKU"].astype(str))
    | set(rtv["产品SKU"].astype(str))
)

if selected_brand != "全部" and "Brand" in meta.columns:
    brand_skus = set(
        meta.loc[meta["Brand"].astype(str) == selected_brand, "公司SKU"].astype(str)
    )
    sku_options = sorted(set(all_skus) & brand_skus)
else:
    brand_skus = set(all_skus)
    sku_options = all_skus

selected_skus = st.sidebar.multiselect("公司 SKU", sku_options)

if selected_brand != "全部":
    sales = sales[sales["公司SKU"].astype(str).isin(brand_skus)]
    rtv = rtv[rtv["产品SKU"].astype(str).isin(brand_skus)]

if selected_skus:
    sales = sales[sales["公司SKU"].astype(str).isin(selected_skus)]
    rtv = rtv[rtv["产品SKU"].astype(str).isin(selected_skus)]

# ----------------------------
# KPI
# ----------------------------
sales_units = sales["Quantity"].sum()
return_units = rtv["QTY"].sum()
overall_rate = return_units / sales_units if sales_units else np.nan

if "总扣款" in rtv.columns:
    total_deduction = rtv["总扣款"].sum()
elif "Total Cost" in rtv.columns:
    total_deduction = rtv["Total Cost"].sum()
else:
    total_deduction = np.nan

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("2026 YTD 销量", f"{sales_units:,.0f}")
c2.metric("2026 YTD 退货量", f"{return_units:,.0f}")
c3.metric("2026 YTD 总体退货率", "—" if pd.isna(overall_rate) else f"{overall_rate:.2%}")
c4.metric("有退货的公司 SKU", f"{rtv['产品SKU'].nunique():,}")
c5.metric(
    "累计退货扣款",
    "—" if pd.isna(total_deduction) else f"${total_deduction:,.2f}",
)

st.caption(
    f"销售统计截止 {latest_order_date.strftime('%Y-%m-%d')}。"
    " 月退货率 = 某月 Order Date 对应的退货 QTY ÷ 同月销售 Units。"
)

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["月度退货率", "SKU总体退货率", "SKU详情", "退货原因", "数据质量"]
)

with tab1:
    st.subheader("公司整体 · 每月退货率")

    sales_m = (
        sales.groupby("Order Month")["Quantity"]
        .sum()
        .reindex(months, fill_value=0)
    )
    rtv_m = (
        rtv.groupby("Order Month")["QTY"]
        .sum()
        .reindex(months, fill_value=0)
    )

    monthly = pd.DataFrame({
        "月份": [str(m) for m in months],
        "销售Units": sales_m.values,
        "退货Units": rtv_m.values,
    })
    monthly["退货率"] = safe_divide(monthly["退货Units"], monthly["销售Units"])

    st.line_chart(
        monthly.set_index("月份")[["退货率"]],
        use_container_width=True,
    )

    monthly_show = monthly.copy()
    monthly_show["退货率"] = monthly_show["退货率"].map(
        lambda x: "—" if pd.isna(x) else f"{x:.2%}"
    )
    st.dataframe(monthly_show, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("每个公司 SKU · 每月退货率")

    sales_sm = (
        sales.groupby(["公司SKU", "Order Month"])["Quantity"]
        .sum()
        .rename("销售Units")
    )

    rtv_sm = (
        rtv.groupby(["产品SKU", "Order Month"])["QTY"]
        .sum()
        .rename("退货Units")
    )
    rtv_sm.index = rtv_sm.index.set_names(["公司SKU", "Order Month"])

    sku_month = (
        pd.concat([sales_sm, rtv_sm], axis=1)
        .fillna(0)
        .reset_index()
    )
    sku_month["退货率"] = safe_divide(
        sku_month["退货Units"],
        sku_month["销售Units"],
    )
    sku_month["月份"] = sku_month["Order Month"].astype(str)

    rate_pivot = sku_month.pivot(
        index="公司SKU",
        columns="月份",
        values="退货率",
    )

    ytd_sales = sales.groupby("公司SKU")["Quantity"].sum().rename("YTD销售")
    ytd_returns = rtv.groupby("产品SKU")["QTY"].sum().rename("YTD退货")

    ytd = pd.concat([ytd_sales, ytd_returns], axis=1).fillna(0)
    ytd["YTD退货率"] = safe_divide(ytd["YTD退货"], ytd["YTD销售"])

    matrix = rate_pivot.join(ytd, how="outer").sort_values(
        "YTD退货率",
        ascending=False,
    )

    format_dict = {
        str(m): "{:.2%}"
        for m in months
        if str(m) in matrix.columns
    }
    format_dict.update({
        "YTD销售": "{:,.0f}",
        "YTD退货": "{:,.0f}",
        "YTD退货率": "{:.2%}",
    })

    st.dataframe(
        matrix.style.format(format_dict, na_rep="—"),
        use_container_width=True,
        height=650,
    )

with tab2:
    st.subheader("公司 SKU · 2026 YTD 总体退货率")

    sales_sku = sales.groupby("公司SKU")["Quantity"].sum().rename("2026销售Units")
    rtv_sku = rtv.groupby("产品SKU")["QTY"].sum().rename("2026退货Units")

    summary = pd.concat([sales_sku, rtv_sku], axis=1).fillna(0)
    summary["总体退货率"] = safe_divide(
        summary["2026退货Units"],
        summary["2026销售Units"],
    )
    summary = summary.reset_index().rename(columns={"index": "公司SKU"})

    if len(meta):
        summary = summary.merge(meta, on="公司SKU", how="left")

    sort_choice = st.selectbox(
        "排序方式",
        ["总体退货率 ↓", "退货Units ↓", "销售Units ↓"],
    )

    if sort_choice == "总体退货率 ↓":
        summary = summary.sort_values("总体退货率", ascending=False)
    elif sort_choice == "退货Units ↓":
        summary = summary.sort_values("2026退货Units", ascending=False)
    else:
        summary = summary.sort_values("2026销售Units", ascending=False)

    display_cols = [
        c for c in [
            "公司SKU", "Brand", "产品名称",
            "2026销售Units", "2026退货Units", "总体退货率"
        ]
        if c in summary.columns
    ]

    st.dataframe(
        summary[display_cols].style.format(
            {
                "2026销售Units": "{:,.0f}",
                "2026退货Units": "{:,.0f}",
                "总体退货率": "{:.2%}",
            },
            na_rep="—",
        ),
        use_container_width=True,
        height=680,
        hide_index=True,
    )

with tab3:
    st.subheader("单 SKU 退货表现")

    detail_skus = sorted(
        set(sales["公司SKU"].astype(str))
        | set(rtv["产品SKU"].astype(str))
    )

    if not detail_skus:
        st.info("当前筛选条件下没有 SKU。")
    else:
        sku = st.selectbox("选择公司 SKU", detail_skus)

        s = sales[sales["公司SKU"].astype(str) == sku].copy()
        r = rtv[rtv["产品SKU"].astype(str) == sku].copy()

        sku_sales = s["Quantity"].sum()
        sku_returns = r["QTY"].sum()
        sku_rate = sku_returns / sku_sales if sku_sales else np.nan

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("YTD销售Units", f"{sku_sales:,.0f}")
        d2.metric("YTD退货Units", f"{sku_returns:,.0f}")
        d3.metric("YTD退货率", "—" if pd.isna(sku_rate) else f"{sku_rate:.2%}")
        d4.metric("RTV记录数", f"{len(r):,}")

        s_m = (
            s.groupby("Order Month")["Quantity"]
            .sum()
            .reindex(months, fill_value=0)
        )
        r_m = (
            r.groupby("Order Month")["QTY"]
            .sum()
            .reindex(months, fill_value=0)
        )

        detail = pd.DataFrame({
            "月份": [str(m) for m in months],
            "销售Units": s_m.values,
            "退货Units": r_m.values,
        })
        detail["退货率"] = safe_divide(
            detail["退货Units"],
            detail["销售Units"],
        )

        st.markdown("#### 月度退货率趋势")
        st.line_chart(
            detail.set_index("月份")[["退货率"]],
            use_container_width=True,
        )

        detail_show = detail.copy()
        detail_show["退货率"] = detail_show["退货率"].map(
            lambda x: "—" if pd.isna(x) else f"{x:.2%}"
        )
        st.dataframe(detail_show, use_container_width=True, hide_index=True)

        st.markdown("#### RTV 明细")

        preferred_cols = [
            "RTV Date", "RTV Number", "PO#", "Order Date",
            "PART#", "原始产品SKU", "产品SKU", "产品名称", "Brand",
            "Reason", "QTY", "UNIT COST", "Unit Cost",
            "Total Cost", "10%运费", "总扣款", "备注",
        ]
        detail_cols = [c for c in preferred_cols if c in r.columns]

        if detail_cols:
            r_show = r[detail_cols].copy()

            if "RTV Date" in r_show.columns:
                r_show = r_show.sort_values("RTV Date", ascending=False)
                r_show["RTV Date"] = r_show["RTV Date"].dt.strftime("%Y-%m-%d")

            if "Order Date" in r_show.columns:
                r_show["Order Date"] = r_show["Order Date"].dt.strftime("%Y-%m-%d")

            st.dataframe(
                r_show,
                use_container_width=True,
                height=500,
                hide_index=True,
            )
        else:
            st.info("RTV文件中没有可显示的明细字段。")

with tab4:
    st.subheader("退货原因分析")

    if "Reason" not in rtv.columns:
        st.info("2026-RTV Sheet 中没有 Reason 字段。")
    else:
        reason_df = (
            rtv.assign(Reason=rtv["Reason"].fillna("未填写"))
            .groupby("Reason")
            .agg(
                退货Units=("QTY", "sum"),
                RTV记录数=("QTY", "size"),
            )
            .reset_index()
            .sort_values("退货Units", ascending=False)
        )

        st.bar_chart(
            reason_df.set_index("Reason")[["退货Units"]],
            use_container_width=True,
        )
        st.dataframe(
            reason_df,
            use_container_width=True,
            hide_index=True,
        )

with tab5:
    st.subheader("数据质量")

    unmapped_sales = sales_raw[sales_raw["公司SKU"] == "未匹配"].copy()

    if known_skus:
        rtv_check = rtv_raw.copy()
        rtv_check["_标准化产品SKU"] = normalize_company_sku(rtv_check["产品SKU"])
        rtv_unknown = rtv_check[
            ~rtv_check["_标准化产品SKU"].isin(known_skus)
        ].copy()
    else:
        rtv_unknown = pd.DataFrame()

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("订单最新日期", latest_order_date.strftime("%Y-%m-%d"))

    if "RTV Date" in rtv_raw.columns and rtv_raw["RTV Date"].notna().any():
        q2.metric("RTV最新日期", rtv_raw["RTV Date"].max().strftime("%Y-%m-%d"))
    else:
        q2.metric("RTV最新日期", "—")

    q3.metric("销售未匹配Vendor SKU", f"{unmapped_sales['Vendor SKU'].nunique():,}")
    q4.metric("RTV中不在Mapping的SKU", f"{rtv_unknown['产品SKU'].nunique():,}")

    if len(rtv_unknown):
        st.markdown("#### RTV 中不在 sku_mapping.xlsx 的产品SKU")
        unknown_summary = (
            rtv_unknown.groupby(["产品SKU", "_标准化产品SKU"], dropna=False)["QTY"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"_标准化产品SKU": "标准化SKU"})
        )
        st.dataframe(
            unknown_summary,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("RTV 中的公司SKU都能在当前 Mapping 中找到。")

    if len(unmapped_sales):
        st.markdown("#### 销售订单中尚未匹配的 Vendor SKU")
        sales_unknown = (
            unmapped_sales.groupby("Vendor SKU")["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        st.dataframe(
            sales_unknown,
            use_container_width=True,
            hide_index=True,
            height=350,
        )
