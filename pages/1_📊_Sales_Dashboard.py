import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Sales Performance Dashboard", page_icon="📊", layout="wide")

BASE = Path(__file__).resolve().parents[1]
DEFAULT_ORDERS = BASE / "data" / "orders_2026_ytd.csv"
DEFAULT_MAPPING = BASE / "data" / "sku_mapping.xlsx"

def read_orders_file(source):
    """
    Read either:
    1) a raw Home Depot order export with metadata rows before the header, or
    2) an already-cleaned CSV whose first row is the actual header.

    Then apply the fixed dashboard business rules:
    - Keep Order Date >= 2026-01-01
    - Exclude Line Status = Cancelled
    - Exclude rows where Unit Cost Currency contains 'USD'
    """

    def _read_with_skiprows(skiprows):
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source, skiprows=skiprows, dtype=str)

    # First inspect only the first few text lines safely, instead of asking
    # pandas to parse the whole raw export before we know where the header is.
    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8-sig", errors="replace") as f:
            first_lines = [f.readline() for _ in range(6)]
    else:
        source.seek(0)
        raw_bytes = source.read()
        if isinstance(raw_bytes, bytes):
            preview_text = raw_bytes.decode("utf-8-sig", errors="replace")
        else:
            preview_text = str(raw_bytes)
        first_lines = preview_text.splitlines(True)[:6]
        source.seek(0)

    preview = "".join(first_lines)

    # HD raw exports start with metadata such as "Search Name:" / "Exported at".
    # Their real CSV header is on row 5, so skip the first 4 rows directly.
    looks_like_hd_raw = (
        "Search Name:" in preview
        or "Exported at" in preview
        or ("Line Status" not in first_lines[0] if first_lines else False)
    )

    if looks_like_hd_raw:
        df = _read_with_skiprows(4)
    else:
        try:
            df = _read_with_skiprows(0)
        except pd.errors.ParserError:
            # Fallback for any HD-like export whose metadata format changed slightly.
            df = _read_with_skiprows(4)

    # One more fallback if the detected header still is not the actual order header.
    if "Order Date" not in df.columns or "Vendor SKU" not in df.columns:
        df = _read_with_skiprows(4)

    required = ["Order Date", "Vendor SKU", "Quantity", "Unit Cost"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"订单CSV缺少必要字段: {', '.join(missing)}")

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

    # 2026 YTD only.
    df = df[df["Order Date"] >= pd.Timestamp("2026-01-01")].copy()

    # Remove cancelled orders.
    if "Line Status" in df.columns:
        status = df["Line Status"].astype("string").str.strip().str.upper()
        df = df[status.ne("CANCELLED") | status.isna()].copy()

    # Remove other-platform rows whose Unit Cost Currency contains USD.
    if "Unit Cost Currency" in df.columns:
        currency = df["Unit Cost Currency"].astype("string").str.strip().str.upper()
        df = df[~currency.str.contains("USD", na=False)].copy()

    return df

def load_default():
    # Always read the latest GitHub data files on rerun.
    # The order reader accepts both raw HD exports and cleaned CSVs.
    return read_orders_file(DEFAULT_ORDERS), pd.read_excel(DEFAULT_MAPPING, dtype=str)

def normalize(s):
    return s.astype("string").str.strip().str.upper()

def prepare(orders, mapping):
    orders = orders.copy()
    mapping = mapping.copy()

    orders["Order Date"] = pd.to_datetime(orders["Order Date"], errors="coerce")
    orders["Quantity"] = pd.to_numeric(orders["Quantity"], errors="coerce").fillna(0)
    orders["Unit Cost"] = pd.to_numeric(orders["Unit Cost"], errors="coerce").fillna(0)
    orders = orders[orders["Order Date"].notna()].copy()
    orders["_order_row_id"] = np.arange(len(orders))
    orders["_key"] = normalize(orders["Vendor SKU"])

    mapping["_key"] = normalize(mapping["MFG Model #"])

    if "Valid From" not in mapping.columns:
        mapping["Valid From"] = pd.NaT
    if "Valid To" not in mapping.columns:
        mapping["Valid To"] = pd.NaT

    mapping["Valid From"] = pd.to_datetime(mapping["Valid From"], errors="coerce")
    mapping["Valid To"] = pd.to_datetime(mapping["Valid To"], errors="coerce")

    mapping = mapping[
        mapping["_key"].notna() &
        (mapping["_key"] != "") &
        (mapping["_key"] != "#N/A")
    ].drop_duplicates().copy()

    candidates = orders[["_order_row_id", "_key", "Order Date"]].merge(
        mapping[
            [
                "_key", "Brand", "公司SKU", "产品名称", "MFG Model #", "OMSID",
                "Valid From", "Valid To"
            ]
        ],
        on="_key",
        how="left"
    )

    valid_mask = (
        (candidates["Valid From"].isna() | (candidates["Order Date"] >= candidates["Valid From"])) &
        (candidates["Valid To"].isna() | (candidates["Order Date"] <= candidates["Valid To"]))
    )
    valid = candidates[valid_mask & candidates["MFG Model #"].notna()].copy()

    if not valid.empty:
        valid["_specificity"] = (
            valid["Valid From"].notna().astype(int) +
            valid["Valid To"].notna().astype(int)
        )
        valid["_from_rank"] = valid["Valid From"].fillna(pd.Timestamp("1900-01-01"))
        valid = valid.sort_values(
            ["_order_row_id", "_specificity", "_from_rank"],
            ascending=[True, False, False]
        )
        valid = valid.drop_duplicates(subset=["_order_row_id"], keep="first")
        chosen = valid[
            [
                "_order_row_id", "Brand", "公司SKU", "产品名称",
                "MFG Model #", "OMSID", "Valid From", "Valid To"
            ]
        ]
    else:
        chosen = pd.DataFrame(
            columns=[
                "_order_row_id", "Brand", "公司SKU", "产品名称",
                "MFG Model #", "OMSID", "Valid From", "Valid To"
            ]
        )

    df = orders.merge(chosen, on="_order_row_id", how="left")

    df["Sales Amount"] = df["Unit Cost"] * df["Quantity"]
    df["公司SKU"] = df["公司SKU"].fillna("未匹配")
    df["MFG Model #"] = df["MFG Model #"].fillna(df["Vendor SKU"])
    df["OMSID"] = df["OMSID"].fillna("未匹配")
    df["Brand"] = df["Brand"].fillna("未匹配")
    df["产品名称"] = df["产品名称"].fillna(df["Description"])

    return df

def pct_change(current, previous):
    if previous == 0:
        return np.nan if current == 0 else np.inf
    return (current - previous) / previous * 100

def fmt_change(x):
    if pd.isna(x):
        return "—"
    if np.isinf(x):
        return "NEW"
    return f"{x:+.1f}%"

def status_from_change(x):
    if pd.isna(x):
        return "—"
    if np.isinf(x):
        return "🚀 Fast Growing"
    if x > 20:
        return "🚀 Fast Growing"
    if x > 5:
        return "↑ Growing"
    if x >= -5:
        return "→ Stable"
    if x >= -20:
        return "↓ Declining"
    return "⚠ Sharp Decline"

def trend_color(v):
    if pd.isna(v):
        return ""
    if np.isinf(v):
        return "background-color: #dcfce7; color: #166534; font-weight: 700;"
    if v > 20:
        return "background-color: #dcfce7; color: #166534; font-weight: 700;"
    if v > 5:
        return "background-color: #f0fdf4; color: #15803d; font-weight: 600;"
    if v >= -5:
        return "background-color: #f8fafc; color: #475569;"
    if v >= -20:
        return "background-color: #fff7ed; color: #c2410c; font-weight: 600;"
    return "background-color: #fee2e2; color: #b91c1c; font-weight: 700;"

def daily_series(data, start_date, end_date):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    idx = pd.date_range(start, end, freq="D")
    daily = data.groupby(data["Order Date"].dt.normalize()).agg(
        Units=("Quantity", "sum"),
        Revenue=("Sales Amount", "sum")
    ).reindex(idx, fill_value=0)
    daily.index.name = "Date"
    daily["7D Avg Units"] = daily["Units"].rolling(7, min_periods=1).mean()
    return daily

orders, mapping = load_default()

with st.sidebar:
    st.header("数据与筛选")
    use_upload = st.toggle("上传更新数据", value=False)
    if use_upload:
        order_file = st.file_uploader("订单 CSV", type=["csv"])
        map_file = st.file_uploader("SKU Mapping Excel", type=["xlsx"])
        if order_file is not None:
            try:
                orders = read_orders_file(order_file)
            except Exception as e:
                st.error(f"订单CSV读取失败：{e}")
                st.stop()
        if map_file is not None:
            mapping = pd.read_excel(map_file, dtype=str)

df = prepare(orders, mapping)
min_date = df["Order Date"].min().date()
max_date = df["Order Date"].max().date()

with st.sidebar:
    quick_range = st.selectbox(
        "快速时间范围",
        ["Single Day", "Last 30 Days", "Last 7 Days", "Last 90 Days", "YTD", "Custom"],
        index=1
    )
    if quick_range == "Single Day":
        single_date = st.date_input(
            "选择单日",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
        start_date = end_date = single_date
    elif quick_range == "Last 7 Days":
        start_date = (pd.Timestamp(max_date) - pd.Timedelta(days=6)).date()
        end_date = max_date
    elif quick_range == "Last 30 Days":
        start_date = (pd.Timestamp(max_date) - pd.Timedelta(days=29)).date()
        end_date = max_date
    elif quick_range == "Last 90 Days":
        start_date = (pd.Timestamp(max_date) - pd.Timedelta(days=89)).date()
        end_date = max_date
    elif quick_range == "YTD":
        start_date = max(min_date, pd.Timestamp(f"{max_date.year}-01-01").date())
        end_date = max_date
    else:
        date_range = st.date_input(
            "日期范围",
            value=((pd.Timestamp(max_date)-pd.Timedelta(days=29)).date(), max_date),
            min_value=min_date,
            max_value=max_date
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range

    brands = sorted(df["Brand"].dropna().unique())
    selected_brands = st.multiselect("Brand", brands, default=brands)
    view_by = st.radio("SKU查看维度", ["公司SKU", "MFG Model #", "OMSID"], horizontal=False)

brand_df = df[df["Brand"].isin(selected_brands)].copy()
filtered = brand_df[
    (brand_df["Order Date"].dt.date >= start_date) &
    (brand_df["Order Date"].dt.date <= end_date)
].copy()

st.title("Sales Performance Dashboard")
st.caption("销售额 = Unit Cost × Quantity；Vendor SKU = MFG Model #；支持 Valid From / Valid To 历史SKU映射；Merchant SKU 不参与分析。")

matched = (df["公司SKU"] != "未匹配").sum()
match_rate = matched / len(df) if len(df) else 0
if match_rate < 0.95:
    st.warning(
        f"当前 SKU Mapping 按订单行匹配率为 {match_rate:.1%}。"
        "未匹配的 Vendor SKU 仍保留在看板中，可在“数据质量”页查看。"
    )

# Core KPI calculations based on the selected period.
start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)
days = (end_ts - start_ts).days + 1
units = filtered["Quantity"].sum()
revenue = filtered["Sales Amount"].sum()
avg_daily = units / days if days else 0
active_skus = filtered.loc[filtered[view_by] != "未匹配", view_by].nunique()

# WoW: latest 7 calendar days ending on selected end date vs previous 7 days.
last7_start = end_ts - pd.Timedelta(days=6)
prev7_start = end_ts - pd.Timedelta(days=13)
prev7_end = end_ts - pd.Timedelta(days=7)
last7_units = brand_df[(brand_df["Order Date"] >= last7_start) & (brand_df["Order Date"] < end_ts + pd.Timedelta(days=1))]["Quantity"].sum()
prev7_units = brand_df[(brand_df["Order Date"] >= prev7_start) & (brand_df["Order Date"] < prev7_end + pd.Timedelta(days=1))]["Quantity"].sum()
wow = pct_change(last7_units, prev7_units)

# MoM: latest 30 calendar days ending on selected end date vs previous 30 days.
last30_start = end_ts - pd.Timedelta(days=29)
prev30_start = end_ts - pd.Timedelta(days=59)
prev30_end = end_ts - pd.Timedelta(days=30)
last30_units = brand_df[(brand_df["Order Date"] >= last30_start) & (brand_df["Order Date"] < end_ts + pd.Timedelta(days=1))]["Quantity"].sum()
prev30_units = brand_df[(brand_df["Order Date"] >= prev30_start) & (brand_df["Order Date"] < prev30_end + pd.Timedelta(days=1))]["Quantity"].sum()
mom = pct_change(last30_units, prev30_units)

if quick_range == "Single Day":
    # "订单数"按唯一 PO Number 统计；订单行数另行显示，避免把一个订单的多个 SKU 行误算成多张订单。
    order_count = filtered["PO Number"].nunique() if "PO Number" in filtered.columns else len(filtered)
    order_lines = len(filtered)
    prev_day = start_ts - pd.Timedelta(days=1)
    prev_day_df = brand_df[
        (brand_df["Order Date"] >= prev_day) &
        (brand_df["Order Date"] < start_ts)
    ]
    prev_day_units = prev_day_df["Quantity"].sum()
    dod_selected_day = pct_change(units, prev_day_units)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("当天订单数", f"{order_count:,}")
    c2.metric("当天销量", f"{units:,.0f}", fmt_change(dod_selected_day))
    c3.metric("当天销售额", f"${revenue:,.2f}")
    c4.metric("Active SKU", f"{active_skus:,}")
    c5.metric("订单行数", f"{order_lines:,}")
    c6.metric("vs 前一天销量", fmt_change(dod_selected_day))
    st.caption(
        f"当前查看单日：{pd.Timestamp(start_date).strftime('%Y-%m-%d')} · "
        "订单数 = Unique PO Number；销量变化与前一自然日比较"
    )
else:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("期间销量", f"{units:,.0f}")
    c2.metric("期间销售额", f"${revenue:,.2f}")
    c3.metric("日均销量", f"{avg_daily:,.1f}")
    c4.metric("WoW · 7D", fmt_change(wow))
    c5.metric("MoM · 30D", fmt_change(mom))
    c6.metric("Active SKU", f"{active_skus:,}")

    st.caption(
        f"当前查看：{pd.Timestamp(start_date).strftime('%Y-%m-%d')} 至 "
        f"{pd.Timestamp(end_date).strftime('%Y-%m-%d')} · 默认建议使用 Last 30 Days 进行日常监控"
    )

tabs = st.tabs(["总览", "SKU表现", "SKU详情", "数据质量"])

with tabs[0]:
    daily = daily_series(filtered, start_date, end_date)

    if quick_range == "Single Day":
        st.subheader("当天 SKU 销售明细")
        single_day_sku = (
            filtered.groupby(view_by, dropna=False)
            .agg(
                Orders=("PO Number", "nunique"),
                Units=("Quantity", "sum"),
                Sales=("Sales Amount", "sum")
            )
            .reset_index()
            .sort_values(["Units", "Sales"], ascending=False)
        )
        single_day_sku["Sales"] = single_day_sku["Sales"].map(lambda x: f"${x:,.2f}")
        st.dataframe(single_day_sku, use_container_width=True, hide_index=True)
    else:
        metric_choice = st.radio("趋势指标", ["销量", "销售额"], horizontal=True)
        if metric_choice == "销量":
            st.subheader("日销与 7 日移动平均")
            st.line_chart(daily[["Units", "7D Avg Units"]], use_container_width=True)
        else:
            st.subheader("每日销售额")
            st.line_chart(daily[["Revenue"]], use_container_width=True)

    left, right = st.columns(2)
    with left:
        rank = filtered[filtered[view_by] != "未匹配"].groupby(view_by)["Quantity"].sum().sort_values(ascending=False).head(15)
        st.subheader("Top 15 SKU · 销量")
        st.bar_chart(rank, horizontal=True)
    with right:
        rank_rev = filtered[filtered[view_by] != "未匹配"].groupby(view_by)["Sales Amount"].sum().sort_values(ascending=False).head(15)
        st.subheader("Top 15 SKU · 销售额")
        st.bar_chart(rank_rev, horizontal=True)

with tabs[1]:
    base = brand_df.copy()

    def grouped_between(start, end, col):
        x = base[(base["Order Date"] >= start) & (base["Order Date"] < end + pd.Timedelta(days=1))]
        return x.groupby(view_by)[col].sum()

    if quick_range == "Single Day":
        selected_day = pd.Timestamp(start_date)
        previous_day = selected_day - pd.Timedelta(days=1)

        day_df = base[
            (base["Order Date"] >= selected_day) &
            (base["Order Date"] < selected_day + pd.Timedelta(days=1))
        ].copy()
        prev_day_df = base[
            (base["Order Date"] >= previous_day) &
            (base["Order Date"] < selected_day)
        ].copy()

        day_perf = day_df[day_df[view_by] != "未匹配"].groupby(view_by).agg(
            Orders=("PO Number", "nunique"),
            Units=("Quantity", "sum"),
            Sales=("Sales Amount", "sum"),
            Order_Lines=("PO Number", "size")
        )

        prev_units = prev_day_df[prev_day_df[view_by] != "未匹配"].groupby(view_by)["Quantity"].sum()
        day_perf["Prev Day Units"] = prev_units
        day_perf = day_perf.fillna(0).reset_index()
        day_perf["DoD %"] = day_perf.apply(
            lambda r: pct_change(r["Units"], r["Prev Day Units"]), axis=1
        )
        day_perf["Trend"] = day_perf["DoD %"].apply(status_from_change)
        day_perf["Avg Unit Value"] = np.where(
            day_perf["Units"] != 0,
            day_perf["Sales"] / day_perf["Units"],
            np.nan
        )

        sort_choice = st.selectbox(
            "排序方式",
            ["当天销量 ↓", "当天销售额 ↓", "DoD % ↑", "DoD % ↓"],
            index=0
        )
        if sort_choice == "当天销量 ↓":
            day_perf = day_perf.sort_values("Units", ascending=False)
        elif sort_choice == "当天销售额 ↓":
            day_perf = day_perf.sort_values("Sales", ascending=False)
        elif sort_choice == "DoD % ↑":
            day_perf = day_perf.sort_values("DoD %", ascending=False)
        else:
            day_perf = day_perf.sort_values("DoD %", ascending=True)

        st.caption(
            f"Single Day SKU表现 · {selected_day.strftime('%Y-%m-%d')}，"
            f"DoD 与 {previous_day.strftime('%Y-%m-%d')} 比较"
        )

        show = day_perf[[
            view_by, "Orders", "Units", "Prev Day Units", "DoD %",
            "Trend", "Sales", "Avg Unit Value", "Order_Lines"
        ]].copy()
        show = show.rename(columns={
            "Orders": "当天订单数",
            "Units": "当天销量",
            "Prev Day Units": "前一天销量",
            "DoD %": "DoD %",
            "Trend": "趋势",
            "Sales": "当天销售额",
            "Avg Unit Value": "平均Unit Cost",
            "Order_Lines": "订单行数"
        })

        styled = (
            show.style
            .map(trend_color, subset=["DoD %"])
            .format({
                "当天订单数": "{:,.0f}",
                "当天销量": "{:,.0f}",
                "前一天销量": "{:,.0f}",
                "DoD %": lambda x: "NEW" if np.isinf(x) else ("—" if pd.isna(x) else f"{x:+.1f}%"),
                "当天销售额": "${:,.2f}",
                "平均Unit Cost": lambda x: "—" if pd.isna(x) else f"${x:,.2f}",
                "订单行数": "{:,.0f}"
            })
        )
        st.dataframe(styled, use_container_width=True, hide_index=True, height=560)

    else:
        perf_index = pd.Index(base[view_by].dropna().unique(), name=view_by)
        perf = pd.DataFrame(index=perf_index)
        perf["7D Units"] = grouped_between(last7_start, end_ts, "Quantity")
        perf["Prev 7D Units"] = grouped_between(prev7_start, prev7_end, "Quantity")
        perf["30D Units"] = grouped_between(last30_start, end_ts, "Quantity")
        perf["30D Revenue"] = grouped_between(last30_start, end_ts, "Sales Amount")
        perf = perf.fillna(0).reset_index()
        perf = perf[perf[view_by] != "未匹配"].copy()
        perf["7D Daily Avg"] = perf["7D Units"] / 7
        perf["WoW %"] = perf.apply(lambda r: pct_change(r["7D Units"], r["Prev 7D Units"]), axis=1)
        perf["Trend"] = perf["WoW %"].apply(status_from_change)
        perf["ASP 30D"] = np.where(perf["30D Units"] != 0, perf["30D Revenue"] / perf["30D Units"], np.nan)

        sort_choice = st.selectbox(
            "排序方式",
            ["30D Units ↓", "30D Revenue ↓", "WoW % ↑", "WoW % ↓"],
            index=0
        )
        if sort_choice == "30D Units ↓":
            perf = perf.sort_values("30D Units", ascending=False)
        elif sort_choice == "30D Revenue ↓":
            perf = perf.sort_values("30D Revenue", ascending=False)
        elif sort_choice == "WoW % ↑":
            perf = perf.sort_values("WoW %", ascending=False)
        else:
            perf = perf.sort_values("WoW %", ascending=True)

        show = perf[[view_by, "7D Units", "Prev 7D Units", "7D Daily Avg", "WoW %", "Trend", "30D Units", "30D Revenue", "ASP 30D"]].copy()
        show = show.rename(columns={
            "7D Units": "7D销量",
            "Prev 7D Units": "前7D销量",
            "7D Daily Avg": "7D日均",
            "WoW %": "WoW %",
            "Trend": "趋势",
            "30D Units": "30D销量",
            "30D Revenue": "30D销售额",
            "ASP 30D": "ASP 30D"
        })

        styled = (
            show.style
            .map(trend_color, subset=["WoW %"])
            .format({
                "7D销量": "{:,.0f}",
                "前7D销量": "{:,.0f}",
                "7D日均": "{:,.1f}",
                "WoW %": lambda x: "NEW" if np.isinf(x) else ("—" if pd.isna(x) else f"{x:+.1f}%"),
                "30D销量": "{:,.0f}",
                "30D销售额": "${:,.2f}",
                "ASP 30D": lambda x: "—" if pd.isna(x) else f"${x:,.2f}"
            })
        )
        st.dataframe(styled, use_container_width=True, hide_index=True, height=560)

with tabs[2]:
    if quick_range == "Single Day":
        # Single Day only lists SKUs that actually had orders on the selected date.
        choice_source = filtered
    else:
        choice_source = brand_df

    choices = sorted(
        choice_source.loc[choice_source[view_by] != "未匹配", view_by]
        .dropna().astype(str).unique()
    )

    if choices:
        selected_sku = st.selectbox(f"选择 {view_by}", choices)
        detail_all = brand_df[brand_df[view_by].astype(str) == selected_sku].copy()
        detail_period = detail_all[
            (detail_all["Order Date"].dt.date >= start_date) &
            (detail_all["Order Date"].dt.date <= end_date)
        ].copy()

        if quick_range == "Single Day":
            selected_day = pd.Timestamp(start_date)
            previous_day = selected_day - pd.Timedelta(days=1)

            selected_day_rows = detail_all[
                (detail_all["Order Date"] >= selected_day) &
                (detail_all["Order Date"] < selected_day + pd.Timedelta(days=1))
            ].copy()
            previous_day_rows = detail_all[
                (detail_all["Order Date"] >= previous_day) &
                (detail_all["Order Date"] < selected_day)
            ].copy()

            day_units = selected_day_rows["Quantity"].sum()
            prev_day_units = previous_day_rows["Quantity"].sum()
            day_sales = selected_day_rows["Sales Amount"].sum()
            day_orders = selected_day_rows["PO Number"].nunique()
            day_lines = len(selected_day_rows)
            day_dod = pct_change(day_units, prev_day_units)
            avg_unit_value = day_sales / day_units if day_units else 0

            st.subheader(f"{selected_sku} · {selected_day.strftime('%Y-%m-%d')} 单日SKU诊断")

            a, b, c, d, e, f = st.columns(6)
            a.metric("当天订单数", f"{day_orders:,}")
            b.metric("当天销量", f"{day_units:,.0f}", fmt_change(day_dod))
            c.metric("前一天销量", f"{prev_day_units:,.0f}")
            d.metric("当天销售额", f"${day_sales:,.2f}")
            e.metric("平均Unit Cost", f"${avg_unit_value:,.2f}")
            f.metric("订单行数", f"{day_lines:,}")

            p1, p2, p3 = st.columns(3)
            p1.metric("DoD", fmt_change(day_dod))
            p2.metric("当天销售订单", f"{day_orders:,}")
            p3.metric("当天Units / Order", f"{(day_units/day_orders if day_orders else 0):,.2f}")

            st.subheader("当天 Unit Cost 分布")
            if len(selected_day_rows):
                uc1, uc2, uc3, uc4 = st.columns(4)
                weighted_uc = (
                    selected_day_rows["Sales Amount"].sum() / selected_day_rows["Quantity"].sum()
                    if selected_day_rows["Quantity"].sum() else 0
                )
                uc1.metric("加权平均Unit Cost", f"${weighted_uc:,.2f}")
                uc2.metric("最低Unit Cost", f"${selected_day_rows['Unit Cost'].min():,.2f}")
                uc3.metric("最高Unit Cost", f"${selected_day_rows['Unit Cost'].max():,.2f}")
                uc4.metric("不同Unit Cost数", f"{selected_day_rows['Unit Cost'].nunique():,}")

                cost_dist = (
                    selected_day_rows.groupby("Unit Cost")
                    .agg(
                        订单数=("PO Number", "nunique"),
                        销量=("Quantity", "sum"),
                        销售额=("Sales Amount", "sum")
                    )
                    .reset_index()
                    .sort_values("Unit Cost")
                )
                cost_dist["Unit Cost"] = cost_dist["Unit Cost"].map(lambda x: f"${x:,.2f}")
                cost_dist["销售额"] = cost_dist["销售额"].map(lambda x: f"${x:,.2f}")
                st.dataframe(cost_dist, use_container_width=True, hide_index=True)

            st.subheader("当天订单明细")
            detail_cols = [
                c for c in [
                    "PO Number", "Order Date", "Vendor SKU", "公司SKU",
                    "MFG Model #", "OMSID", "Quantity", "Unit Cost",
                    "Sales Amount", "Customer Order Number",
                    "ShipTo City", "ShipTo State"
                ] if c in selected_day_rows.columns
            ]
            order_detail = selected_day_rows[detail_cols].copy()
            if "Order Date" in order_detail.columns:
                order_detail["Order Date"] = pd.to_datetime(order_detail["Order Date"]).dt.strftime("%Y-%m-%d")
            if "Unit Cost" in order_detail.columns:
                order_detail["Unit Cost"] = order_detail["Unit Cost"].map(lambda x: f"${x:,.2f}")
            if "Sales Amount" in order_detail.columns:
                order_detail["Sales Amount"] = order_detail["Sales Amount"].map(lambda x: f"${x:,.2f}")
            st.dataframe(order_detail, use_container_width=True, hide_index=True, height=420)

            st.subheader("对应关系")

            # 1) 当天有效 Mapping：直接读取 Mapping Excel，并按 Valid From / Valid To 判断
            mapping_view = mapping.copy()
            if "Valid From" not in mapping_view.columns:
                mapping_view["Valid From"] = pd.NaT
            if "Valid To" not in mapping_view.columns:
                mapping_view["Valid To"] = pd.NaT

            mapping_view["Valid From"] = pd.to_datetime(mapping_view["Valid From"], errors="coerce")
            mapping_view["Valid To"] = pd.to_datetime(mapping_view["Valid To"], errors="coerce")

            mapping_view = mapping_view[
                mapping_view[view_by].astype(str).str.strip().eq(str(selected_sku).strip())
            ].copy()

            selected_map_day = pd.Timestamp(start_date)
            valid_on_day = (
                (mapping_view["Valid From"].isna() | (selected_map_day >= mapping_view["Valid From"])) &
                (mapping_view["Valid To"].isna() | (selected_map_day <= mapping_view["Valid To"]))
            )
            valid_mapping_view = mapping_view[valid_on_day].copy()

            cols = ["Brand", "公司SKU", "MFG Model #", "OMSID", "产品名称", "Valid From", "Valid To"]
            cols = [c for c in cols if c in valid_mapping_view.columns]

            if "Valid From" in valid_mapping_view.columns:
                valid_mapping_view["Valid From"] = valid_mapping_view["Valid From"].dt.strftime("%Y-%m-%d").fillna("")
            if "Valid To" in valid_mapping_view.columns:
                valid_mapping_view["Valid To"] = valid_mapping_view["Valid To"].dt.strftime("%Y-%m-%d").fillna("")

            st.markdown("#### 当天有效 Mapping")
            st.caption(f"{selected_map_day.strftime('%Y-%m-%d')} 按 Valid From / Valid To 判断有效的 Mapping 规则")
            st.dataframe(
                valid_mapping_view[cols].drop_duplicates(),
                use_container_width=True,
                hide_index=True
            )

            # 2) 当天实际订单 Vendor SKU：只看当天订单里真实出现过的 Vendor SKU
            # 即使该型号在 Mapping 规则上已失效，也保留展示，便于追溯历史异常订单。
            st.markdown("#### 当天实际订单 Vendor SKU")
            actual_vendor = (
                selected_day_rows.groupby("Vendor SKU", dropna=False)
                .agg(
                    订单数=("PO Number", "nunique"),
                    销量=("Quantity", "sum"),
                    销售额=("Sales Amount", "sum")
                )
                .reset_index()
                .sort_values(["销量", "订单数"], ascending=False)
            )

            if len(actual_vendor):
                actual_vendor["销售额"] = actual_vendor["销售额"].map(lambda x: f"${x:,.2f}")

                # Add whether the Vendor SKU is also in the day's valid mapping rules.
                valid_vendor_set = set(
                    valid_mapping_view["MFG Model #"].dropna().astype(str).str.strip()
                )
                actual_vendor["Mapping状态"] = actual_vendor["Vendor SKU"].astype(str).str.strip().apply(
                    lambda x: "当天有效" if x in valid_vendor_set else "历史/异常订单型号"
                )

                st.caption("以下为该公司 SKU 在所选日期订单中实际出现的 Vendor SKU")
                st.dataframe(
                    actual_vendor[["Vendor SKU", "Mapping状态", "订单数", "销量", "销售额"]],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("所选日期没有该 SKU 的实际订单 Vendor SKU。")

        else:
            d_last7 = detail_all[(detail_all["Order Date"] >= last7_start) & (detail_all["Order Date"] < end_ts + pd.Timedelta(days=1))]["Quantity"].sum()
            d_prev7 = detail_all[(detail_all["Order Date"] >= prev7_start) & (detail_all["Order Date"] < prev7_end + pd.Timedelta(days=1))]["Quantity"].sum()
            d_wow = pct_change(d_last7, d_prev7)

            d_last30 = detail_all[(detail_all["Order Date"] >= last30_start) & (detail_all["Order Date"] < end_ts + pd.Timedelta(days=1))]
            d_prev30 = detail_all[(detail_all["Order Date"] >= prev30_start) & (detail_all["Order Date"] < prev30_end + pd.Timedelta(days=1))]
            d_mom = pct_change(d_last30["Quantity"].sum(), d_prev30["Quantity"].sum())

            today_units = detail_all[detail_all["Order Date"].dt.normalize() == end_ts.normalize()]["Quantity"].sum()
            yesterday = end_ts - pd.Timedelta(days=1)
            yesterday_units = detail_all[detail_all["Order Date"].dt.normalize() == yesterday.normalize()]["Quantity"].sum()
            d_dod = pct_change(today_units, yesterday_units)

            st.subheader(f"{selected_sku} · SKU诊断")
            a, b, c, d, e, f = st.columns(6)
            a.metric("7D销量", f"{d_last7:,.0f}")
            b.metric("前7D销量", f"{d_prev7:,.0f}")
            c.metric("WoW", fmt_change(d_wow))
            d.metric("30D销量", f"{d_last30['Quantity'].sum():,.0f}")
            e.metric("30D销售额", f"${d_last30['Sales Amount'].sum():,.2f}")
            f.metric("30D日均", f"{d_last30['Quantity'].sum()/30:,.1f}")

            p1, p2, p3 = st.columns(3)
            p1.metric("DoD", fmt_change(d_dod))
            p2.metric("WoW", fmt_change(d_wow))
            p3.metric("MoM · 30D", fmt_change(d_mom))

            d_daily = daily_series(detail_period, start_date, end_date)
            st.subheader("日销与 7 日移动平均")
            st.line_chart(d_daily[["Units", "7D Avg Units"]], use_container_width=True)

            st.subheader("每日销售额")
            st.line_chart(d_daily[["Revenue"]], use_container_width=True)

            st.subheader("每日 Unit Cost")
            unit_cost_daily = (
                detail_period.groupby(detail_period["Order Date"].dt.normalize())
                .agg(
                    Units=("Quantity", "sum"),
                    Orders=("PO Number", "nunique"),
                    Sales_Amount=("Sales Amount", "sum"),
                    Min_Unit_Cost=("Unit Cost", "min"),
                    Max_Unit_Cost=("Unit Cost", "max")
                )
                .reset_index()
                .rename(columns={"Order Date": "Date"})
            )
            unit_cost_daily["Avg_Unit_Cost"] = np.where(
                unit_cost_daily["Units"] != 0,
                unit_cost_daily["Sales_Amount"] / unit_cost_daily["Units"],
                np.nan
            )
            unit_cost_daily["Unit_Cost_Change_%"] = unit_cost_daily["Avg_Unit_Cost"].pct_change() * 100

            all_days = pd.DataFrame({
                "Date": pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")
            })
            unit_cost_daily = all_days.merge(unit_cost_daily, on="Date", how="left")
            unit_cost_daily["Units"] = unit_cost_daily["Units"].fillna(0)
            unit_cost_daily["Orders"] = unit_cost_daily["Orders"].fillna(0)
            unit_cost_daily["Sales_Amount"] = unit_cost_daily["Sales_Amount"].fillna(0)

            uc_left, uc_right = st.columns(2)
            with uc_left:
                st.markdown("##### Unit Cost 日趋势")
                st.line_chart(
                    unit_cost_daily.set_index("Date")[["Avg_Unit_Cost"]],
                    use_container_width=True
                )
            with uc_right:
                st.markdown("##### 销量 vs Unit Cost")
                scatter_source = unit_cost_daily[
                    unit_cost_daily["Avg_Unit_Cost"].notna()
                ][["Avg_Unit_Cost", "Units"]].copy()
                if len(scatter_source):
                    st.scatter_chart(
                        scatter_source,
                        x="Avg_Unit_Cost",
                        y="Units",
                        use_container_width=True
                    )
                else:
                    st.info("当前期间没有可用于价格关系分析的销售数据。")

            st.caption(
                "Avg Unit Cost = 当天 Sales Amount ÷ 当天 Units，为按销量加权的平均 Unit Cost；"
                "用于观察价格变化与销量变化是否同步。"
            )

            unit_cost_table = unit_cost_daily.copy()
            unit_cost_table["Date"] = unit_cost_table["Date"].dt.strftime("%Y-%m-%d")
            unit_cost_table = unit_cost_table.rename(columns={
                "Units": "当天销量",
                "Orders": "当天订单数",
                "Avg_Unit_Cost": "加权平均Unit Cost",
                "Min_Unit_Cost": "最低Unit Cost",
                "Max_Unit_Cost": "最高Unit Cost",
                "Unit_Cost_Change_%": "Unit Cost DoD %",
                "Sales_Amount": "当天销售额"
            })
            show_cols = [
                "Date", "当天订单数", "当天销量", "加权平均Unit Cost",
                "最低Unit Cost", "最高Unit Cost", "Unit Cost DoD %", "当天销售额"
            ]
            styled_cost = (
                unit_cost_table[show_cols].style
                .format({
                    "当天订单数": "{:,.0f}",
                    "当天销量": "{:,.0f}",
                    "加权平均Unit Cost": lambda x: "—" if pd.isna(x) else f"${x:,.2f}",
                    "最低Unit Cost": lambda x: "—" if pd.isna(x) else f"${x:,.2f}",
                    "最高Unit Cost": lambda x: "—" if pd.isna(x) else f"${x:,.2f}",
                    "Unit Cost DoD %": lambda x: "—" if pd.isna(x) else f"{x:+.1f}%",
                    "当天销售额": "${:,.2f}"
                })
            )
            st.dataframe(
                styled_cost,
                use_container_width=True,
                hide_index=True,
                height=420
            )

            st.subheader("对应关系")
            mapping_view = mapping.copy()
            mapping_view = mapping_view[
                mapping_view[view_by].astype(str).str.strip().eq(str(selected_sku).strip())
            ].copy()

            cols = ["Brand", "公司SKU", "MFG Model #", "OMSID", "产品名称", "Valid From", "Valid To"]
            cols = [c for c in cols if c in mapping_view.columns]
            st.caption("以下直接来自 SKU Mapping 表，包含当前及历史 Mapping 规则")
            st.dataframe(
                mapping_view[cols].drop_duplicates(),
                use_container_width=True,
                hide_index=True
            )
    else:
        if quick_range == "Single Day":
            st.info("该日期在当前 Brand / SKU 查看维度下没有已匹配的 SKU 订单。")
        else:
            st.info("当前筛选条件下没有可用 SKU。")

with tabs[3]:
    st.subheader("Mapping 文件状态")
    st.write(f"当前 Mapping 总行数：{len(mapping):,}")
    if "公司SKU" in mapping.columns:
        ydz_rules = mapping[mapping["公司SKU"].astype(str).str.strip().eq("YDZ-32A")].copy()
        if len(ydz_rules):
            st.caption("当前程序实际读取到的 YDZ-32A Mapping：")
            cols_check = [c for c in ["Brand","公司SKU","MFG Model #","OMSID","Valid From","Valid To"] if c in ydz_rules.columns]
            st.dataframe(ydz_rules[cols_check], use_container_width=True, hide_index=True)

    unmatched = df[df["公司SKU"] == "未匹配"].copy()
    st.metric("Mapping 匹配率", f"{match_rate:.1%}")
    st.write(f"未匹配订单行：{len(unmatched):,}；未匹配 Vendor SKU：{unmatched['Vendor SKU'].nunique():,}")
    if len(unmatched):
        u = unmatched.groupby("Vendor SKU").agg(
            Order_Lines=("Vendor SKU", "size"),
            Units=("Quantity", "sum"),
            Sales_Amount=("Sales Amount", "sum")
        ).reset_index().sort_values("Units", ascending=False)
        st.dataframe(u, use_container_width=True, hide_index=True)
        st.download_button(
            "下载未匹配 Vendor SKU",
            u.to_csv(index=False).encode("utf-8-sig"),
            file_name="unmatched_vendor_sku.csv",
            mime="text/csv"
        )
