import streamlit as st

st.set_page_config(
    page_title="Home Depot Business Dashboard",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Home Depot Business Dashboard")
st.caption("Sales + RTV unified workspace")

st.markdown("""
### 左侧选择看板

- **📊 Sales Dashboard**：销量、销售额、SKU表现、Unit Cost、Single Day、Mapping等
- **↩️ RTV Dashboard**：月度退货率、YTD退货率、SKU退货表现、退货原因与明细

两个看板共用同一套：
- `orders_2026_ytd.csv`
- `sku_mapping.xlsx`

RTV Dashboard 额外读取：
- `rtv_2026.xlsx`

> RTV月份统一按照 **Order Date** 归属，而不是 RTV Date。
""")
