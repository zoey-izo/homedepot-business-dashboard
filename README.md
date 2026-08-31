# Home Depot Business Dashboard

Unified Streamlit multipage dashboard.

## Pages
- Sales Dashboard V11
- RTV Dashboard

## Repository structure

```text
home-depot-business-dashboard/
├── app.py
├── requirements.txt
├── pages/
│   ├── 1_📊_Sales_Dashboard.py
│   └── 2_↩️_RTV_Dashboard.py
└── data/
    ├── orders_2026_ytd.csv
    ├── sku_mapping.xlsx
    └── rtv_2026.xlsx
```

Deploy `app.py` as the Streamlit main file.

The two pages share the same sales and mapping files. RTV additionally reads `rtv_2026.xlsx`.
