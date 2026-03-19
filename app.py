#!/usr/bin/env python3
"""
Workflow AHT Adherence Report
Generates a per-DA/QA (or per-Site), per-week adherence report showing how many
workflow+locale combinations meet the expected AHT goal.

Output format matches the manager template:
  DA Alias | Manager | Site | Wk.N (#audited, #>goal, #<=goal, adherence%) | ...
  -- OR --
  Site | Wk.N (#audited, #>goal, #<=goal, adherence%) | ...
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AHT Adherence Report",
    page_icon="📋",
    layout="wide",
)

# ---------------------------------------------------------------------------
# COLUMN MAPPING
# ---------------------------------------------------------------------------

COLUMN_MAPPING = {
    "Column-1:Transformation Type": "workflow",
    "Column-2:Locale": "locale",
    "Column-3:Site": "site",
    "Select Date Part": "date_part",
    "Average Handle Time(In Secs)": "aht",
    "Processed Units": "processed_units",
    "Column-4:Ops Manager": "ops_manager",
    "Column-5:Team Manager": "team_manager",
    "Column-6:DA": "da_name",
    "Column-7:Demand Category": "demand_category",
    "Column-8:Customer": "customer",
}

REGION_MAPPING = {
    "BOS": "AMER",
    "SJO": "AMER",
    "AMS": "EMEA",
    "CBG": "EMEA",
    "GDN": "EMEA",
    "LHR": "EMEA",
    "AMM": "APAJ",
    "HYD": "APAJ",
    "KIX": "APAJ",
    "MAA": "APAJ",
}


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

@st.cache_data
def load_data(uploaded_file):
    """Load and clean the main data CSV."""
    df = pd.read_csv(uploaded_file)
    df = df.rename(columns=COLUMN_MAPPING)
    df.columns = df.columns.str.strip()

    for col in ["site", "workflow", "locale"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df = df[df[col].notna() & (df[col] != "") & (df[col] != "nan")]

    for col in ["ops_manager", "team_manager", "da_name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(["nan", ""], pd.NA)
        else:
            df[col] = pd.NA

    # Add region
    df["region"] = df["site"].map(REGION_MAPPING).fillna("Unknown")

    df["aht"] = pd.to_numeric(df["aht"], errors="coerce")
    df["processed_units"] = pd.to_numeric(df["processed_units"], errors="coerce")
    df = df.dropna(subset=["aht", "processed_units"])
    df = df[(df["processed_units"] > 0) & (df["aht"] > 0)].copy()

    df["date_part"] = df["date_part"].astype(str)
    if df["date_part"].str.contains("Week", case=False).any():
        df["year"] = df["date_part"].str.extract(r"(\d{4})").astype(int)
        df["week_num"] = df["date_part"].str.extract(r"Week\s*(\d+)", expand=False).astype(int)
        df["period_sort"] = df["year"] * 100 + df["week_num"]
        df["week_label"] = "Wk." + df["week_num"].astype(str)
    else:
        df["week_num"] = df["date_part"].astype(int)
        df["period_sort"] = df["week_num"]
        df["week_label"] = "Wk." + df["week_num"].astype(str)

    df["workflow_locale"] = df["workflow"] + " | " + df["locale"]

    return df


@st.cache_data
def load_expected_aht(uploaded_file):
    """Load expected AHT goals CSV (Workflow, Locale, Expected AHT)."""
    edf = pd.read_csv(uploaded_file)
    if len(edf.columns) >= 3:
        edf = edf.iloc[:, :3].copy()
        edf.columns = ["workflow", "locale", "expected_aht"]
    else:
        edf.columns = edf.columns.str.strip().str.lower().str.replace(" ", "_")

    edf["workflow"] = edf["workflow"].astype(str).str.strip()
    edf["locale"] = edf["locale"].astype(str).str.strip()
    edf["expected_aht"] = pd.to_numeric(edf["expected_aht"], errors="coerce")
    edf = edf.dropna(subset=["expected_aht"])
    edf = edf[edf["expected_aht"] > 0]
    edf["workflow_locale"] = edf["workflow"] + " | " + edf["locale"]

    return edf[["workflow_locale", "expected_aht"]].drop_duplicates()


@st.cache_data
def compute_network_expected_aht(df):
    """Compute network-level weighted mean AHT per workflow+locale as default goals."""
    def weighted_mean(group):
        return np.average(group["aht"], weights=group["processed_units"])

    expected = (
        df.groupby("workflow_locale")
        .apply(weighted_mean, include_groups=False)
        .reset_index()
    )
    expected.columns = ["workflow_locale", "expected_aht"]
    return expected


# ---------------------------------------------------------------------------
# REPORT GENERATION
# ---------------------------------------------------------------------------

def generate_adherence_report_person(df, expected_aht_df, person_col, manager_col):
    """
    Generate the adherence report at the PERSON level.

    For each person+week, count:
      - # of workflow_locale combinations worked
      - # where person's AHT > expected AHT (not meeting goal)
      - # where person's AHT <= expected AHT (meeting goal)
      - Adherence rate = #meeting / #total
    """
    merged = df.merge(expected_aht_df, on="workflow_locale", how="inner")

    person_wf_week = (
        merged.groupby([person_col, manager_col, "site", "workflow_locale", "week_label", "period_sort"])
        .agg(
            actual_aht=("aht", lambda x: np.average(x, weights=merged.loc[x.index, "processed_units"])),
            expected_aht=("expected_aht", "first"),
            units=("processed_units", "sum"),
        )
        .reset_index()
    )

    person_wf_week["meets_goal"] = person_wf_week["actual_aht"] <= person_wf_week["expected_aht"]

    person_week = (
        person_wf_week.groupby([person_col, manager_col, "site", "week_label", "period_sort"])
        .agg(
            num_wf_locale=("workflow_locale", "nunique"),
            num_above_goal=("meets_goal", lambda x: (~x).sum()),
            num_at_or_below=("meets_goal", "sum"),
        )
        .reset_index()
    )

    person_week["adherence_rate"] = (
        person_week["num_at_or_below"] / person_week["num_wf_locale"]
    )

    weeks_sorted = (
        person_week[["week_label", "period_sort"]]
        .drop_duplicates()
        .sort_values("period_sort")
    )
    week_order = weeks_sorted["week_label"].tolist()

    rows = []
    persons = (
        person_week.groupby([person_col, manager_col, "site"])
        .size()
        .reset_index()[[person_col, manager_col, "site"]]
    )

    for _, person_row in persons.iterrows():
        row = {
            "DA/QA Alias": person_row[person_col],
            "Manager": person_row[manager_col],
            "Site": person_row["site"],
        }

        person_data = person_week[
            (person_week[person_col] == person_row[person_col])
            & (person_week[manager_col] == person_row[manager_col])
            & (person_week["site"] == person_row["site"])
        ]

        for wk in week_order:
            wk_data = person_data[person_data["week_label"] == wk]
            if not wk_data.empty:
                r = wk_data.iloc[0]
                row[f"{wk}|#audited"] = int(r["num_wf_locale"])
                row[f"{wk}|#AHT>WF AHT"] = int(r["num_above_goal"])
                row[f"{wk}|#AHT<=WF AHT"] = int(r["num_at_or_below"])
                row[f"{wk}|Adherence%"] = r["adherence_rate"]
            else:
                row[f"{wk}|#audited"] = 0
                row[f"{wk}|#AHT>WF AHT"] = 0
                row[f"{wk}|#AHT<=WF AHT"] = 0
                row[f"{wk}|Adherence%"] = np.nan

        rows.append(row)

    report = pd.DataFrame(rows)
    return report, week_order


def generate_adherence_report_site(df, expected_aht_df, include_region=True):
    """
    Generate the adherence report at the SITE level.

    For each site+week, compute the site-level weighted AHT per workflow_locale,
    then count:
      - # of workflow_locale combinations at the site
      - # where site AHT > expected AHT
      - # where site AHT <= expected AHT
      - Adherence rate
    """
    merged = df.merge(expected_aht_df, on="workflow_locale", how="inner")

    # Aggregate to site + workflow_locale + week
    site_wf_week = (
        merged.groupby(["site", "region", "workflow_locale", "week_label", "period_sort"])
        .agg(
            actual_aht=("aht", lambda x: np.average(x, weights=merged.loc[x.index, "processed_units"])),
            expected_aht=("expected_aht", "first"),
            units=("processed_units", "sum"),
            num_das=("da_name", "nunique"),
        )
        .reset_index()
    )

    site_wf_week["meets_goal"] = site_wf_week["actual_aht"] <= site_wf_week["expected_aht"]

    # Aggregate to site + week
    site_week = (
        site_wf_week.groupby(["site", "region", "week_label", "period_sort"])
        .agg(
            num_wf_locale=("workflow_locale", "nunique"),
            num_above_goal=("meets_goal", lambda x: (~x).sum()),
            num_at_or_below=("meets_goal", "sum"),
            total_units=("units", "sum"),
            num_das=("num_das", "max"),
        )
        .reset_index()
    )

    site_week["adherence_rate"] = (
        site_week["num_at_or_below"] / site_week["num_wf_locale"]
    )

    weeks_sorted = (
        site_week[["week_label", "period_sort"]]
        .drop_duplicates()
        .sort_values("period_sort")
    )
    week_order = weeks_sorted["week_label"].tolist()

    rows = []
    sites = site_week[["site", "region"]].drop_duplicates().sort_values(["region", "site"])

    for _, site_row in sites.iterrows():
        row = {
            "Site": site_row["site"],
        }
        if include_region:
            row["Region"] = site_row["region"]

        site_data = site_week[site_week["site"] == site_row["site"]]

        for wk in week_order:
            wk_data = site_data[site_data["week_label"] == wk]
            if not wk_data.empty:
                r = wk_data.iloc[0]
                row[f"{wk}|#audited"] = int(r["num_wf_locale"])
                row[f"{wk}|#AHT>WF AHT"] = int(r["num_above_goal"])
                row[f"{wk}|#AHT<=WF AHT"] = int(r["num_at_or_below"])
                row[f"{wk}|Adherence%"] = r["adherence_rate"]
                row[f"{wk}|Units"] = int(r["total_units"])
                row[f"{wk}|DAs"] = int(r["num_das"])
            else:
                row[f"{wk}|#audited"] = 0
                row[f"{wk}|#AHT>WF AHT"] = 0
                row[f"{wk}|#AHT<=WF AHT"] = 0
                row[f"{wk}|Adherence%"] = np.nan
                row[f"{wk}|Units"] = 0
                row[f"{wk}|DAs"] = 0

        rows.append(row)

    report = pd.DataFrame(rows)
    return report, week_order


def generate_adherence_report_region(df, expected_aht_df):
    """
    Generate the adherence report at the REGION level.

    For each region+week, compute the region-level weighted AHT per workflow_locale,
    then count adherence.
    """
    merged = df.merge(expected_aht_df, on="workflow_locale", how="inner")

    region_wf_week = (
        merged.groupby(["region", "workflow_locale", "week_label", "period_sort"])
        .agg(
            actual_aht=("aht", lambda x: np.average(x, weights=merged.loc[x.index, "processed_units"])),
            expected_aht=("expected_aht", "first"),
            units=("processed_units", "sum"),
            num_sites=("site", "nunique"),
            num_das=("da_name", "nunique"),
        )
        .reset_index()
    )

    region_wf_week["meets_goal"] = region_wf_week["actual_aht"] <= region_wf_week["expected_aht"]

    region_week = (
        region_wf_week.groupby(["region", "week_label", "period_sort"])
        .agg(
            num_wf_locale=("workflow_locale", "nunique"),
            num_above_goal=("meets_goal", lambda x: (~x).sum()),
            num_at_or_below=("meets_goal", "sum"),
            total_units=("units", "sum"),
            num_sites=("num_sites", "max"),
            num_das=("num_das", "max"),
        )
        .reset_index()
    )

    region_week["adherence_rate"] = (
        region_week["num_at_or_below"] / region_week["num_wf_locale"]
    )

    weeks_sorted = (
        region_week[["week_label", "period_sort"]]
        .drop_duplicates()
        .sort_values("period_sort")
    )
    week_order = weeks_sorted["week_label"].tolist()

    rows = []
    regions = sorted(region_week["region"].unique())

    for region in regions:
        row = {"Region": region}
        region_data = region_week[region_week["region"] == region]

        for wk in week_order:
            wk_data = region_data[region_data["week_label"] == wk]
            if not wk_data.empty:
                r = wk_data.iloc[0]
                row[f"{wk}|#audited"] = int(r["num_wf_locale"])
                row[f"{wk}|#AHT>WF AHT"] = int(r["num_above_goal"])
                row[f"{wk}|#AHT<=WF AHT"] = int(r["num_at_or_below"])
                row[f"{wk}|Adherence%"] = r["adherence_rate"]
                row[f"{wk}|Units"] = int(r["total_units"])
                row[f"{wk}|Sites"] = int(r["num_sites"])
            else:
                row[f"{wk}|#audited"] = 0
                row[f"{wk}|#AHT>WF AHT"] = 0
                row[f"{wk}|#AHT<=WF AHT"] = 0
                row[f"{wk}|Adherence%"] = np.nan
                row[f"{wk}|Units"] = 0
                row[f"{wk}|Sites"] = 0

        rows.append(row)

    report = pd.DataFrame(rows)
    return report, week_order


# ---------------------------------------------------------------------------
# EXCEL EXPORT
# ---------------------------------------------------------------------------

def build_excel_report(report_df, week_order, fixed_cols, agg_mode):
    """
    Build a formatted Excel file with merged header rows.
    """
    from io import BytesIO

    output = BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet("AHT Adherence Report")
        writer.sheets["AHT Adherence Report"] = worksheet

        # Formats
        header_fmt = workbook.add_format({
            "bold": True, "align": "center", "valign": "vcenter",
            "border": 1, "bg_color": "#4472C4", "font_color": "white",
            "text_wrap": True,
        })
        subheader_fmt = workbook.add_format({
            "bold": True, "align": "center", "valign": "vcenter",
            "border": 1, "bg_color": "#D6E4F0", "text_wrap": True,
            "font_size": 9,
        })
        pct_fmt = workbook.add_format({
            "num_format": "0%", "align": "center", "border": 1,
        })
        num_fmt = workbook.add_format({
            "align": "center", "border": 1,
        })
        text_fmt = workbook.add_format({
            "align": "left", "border": 1,
        })
        green_pct_fmt = workbook.add_format({
            "num_format": "0%", "align": "center", "border": 1,
            "bg_color": "#C6EFCE", "font_color": "#006100",
        })
        red_pct_fmt = workbook.add_format({
            "num_format": "0%", "align": "center", "border": 1,
            "bg_color": "#FFC7CE", "font_color": "#9C0006",
        })
        yellow_pct_fmt = workbook.add_format({
            "num_format": "0%", "align": "center", "border": 1,
            "bg_color": "#FFEB9C", "font_color": "#9C6500",
        })

        n_fixed = len(fixed_cols)

        # Determine sub-headers based on aggregation mode
        if agg_mode == "person":
            sub_headers = [
                "#of workflow_locale\ncombination audited",
                "#with AHT\n>WF AHT",
                "#with AHT\n<=WF AHT",
                "Workflow AHT\nAdherence Rate",
            ]
            n_sub = 4
        elif agg_mode == "site":
            sub_headers = [
                "#of workflow_locale\ncombination audited",
                "#with AHT\n>WF AHT",
                "#with AHT\n<=WF AHT",
                "Workflow AHT\nAdherence Rate",
                "Total\nUnits",
                "# DAs",
            ]
            n_sub = 6
        else:  # region
            sub_headers = [
                "#of workflow_locale\ncombination audited",
                "#with AHT\n>WF AHT",
                "#with AHT\n<=WF AHT",
                "Workflow AHT\nAdherence Rate",
                "Total\nUnits",
                "# Sites",
            ]
            n_sub = 6

        # Write fixed column headers (merged across rows 0-1)
        for i, col_name in enumerate(fixed_cols):
            worksheet.merge_range(0, i, 1, i, col_name, header_fmt)

        # Write week group headers
        col_offset = n_fixed
        for wk in week_order:
            worksheet.merge_range(0, col_offset, 0, col_offset + n_sub - 1, wk, header_fmt)
            for j, sub in enumerate(sub_headers):
                worksheet.write(1, col_offset + j, sub, subheader_fmt)
            col_offset += n_sub

        # Write data rows
        for row_idx, (_, row) in enumerate(report_df.iterrows()):
            data_row = row_idx + 2

            for i, col_name in enumerate(fixed_cols):
                worksheet.write(data_row, i, row.get(col_name, ""), text_fmt)

            col_offset = n_fixed
            for wk in week_order:
                audited = row.get(f"{wk}|#audited", 0)
                above = row.get(f"{wk}|#AHT>WF AHT", 0)
                at_below = row.get(f"{wk}|#AHT<=WF AHT", 0)
                adherence = row.get(f"{wk}|Adherence%", None)

                worksheet.write(data_row, col_offset, audited, num_fmt)
                worksheet.write(data_row, col_offset + 1, above, num_fmt)
                worksheet.write(data_row, col_offset + 2, at_below, num_fmt)

                if adherence is not None and not np.isnan(adherence):
                    if adherence >= 0.8:
                        worksheet.write(data_row, col_offset + 3, adherence, green_pct_fmt)
                    elif adherence < 0.6:
                        worksheet.write(data_row, col_offset + 3, adherence, red_pct_fmt)
                    else:
                        worksheet.write(data_row, col_offset + 3, adherence, yellow_pct_fmt)
                else:
                    worksheet.write(data_row, col_offset + 3, "—", num_fmt)

                if n_sub > 4:
                    if agg_mode == "site":
                        worksheet.write(data_row, col_offset + 4, row.get(f"{wk}|Units", 0), num_fmt)
                        worksheet.write(data_row, col_offset + 5, row.get(f"{wk}|DAs", 0), num_fmt)
                    elif agg_mode == "region":
                        worksheet.write(data_row, col_offset + 4, row.get(f"{wk}|Units", 0), num_fmt)
                        worksheet.write(data_row, col_offset + 5, row.get(f"{wk}|Sites", 0), num_fmt)

                col_offset += n_sub

        # Column widths
        for i in range(n_fixed):
            worksheet.set_column(i, i, 18)
        worksheet.set_column(n_fixed, n_fixed + len(week_order) * n_sub, 14)

        worksheet.set_row(0, 25)
        worksheet.set_row(1, 50)

    output.seek(0)
    return output


# ---------------------------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------------------------

def main():
    st.title("📋 Workflow AHT Adherence Report")
    st.markdown(
        "Generates a per-person, per-site, or per-region weekly report of how many "
        "workflow+locale combinations meet the expected AHT goal."
    )

    # --- File uploads ---
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        data_file = st.file_uploader(
            "Upload Data CSV",
            type=["csv"],
            help="Your standard workflow data export",
        )

    with col_f2:
        expected_file = st.file_uploader(
            "Upload Expected AHT CSV (Optional)",
            type=["csv"],
            help="Workflow, Locale, Expected AHT. If omitted, network weighted mean is used.",
        )

    if data_file is None:
        st.info("📂 Upload your data CSV to get started.")
        return

    # --- Load data ---
    with st.spinner("Loading data..."):
        df = load_data(data_file)

    st.success(
        f"✅ Loaded {len(df):,} rows | "
        f"{df['period_sort'].nunique()} weeks | "
        f"{df['workflow_locale'].nunique()} workflow+locale combos | "
        f"{df['site'].nunique()} sites"
    )

    if expected_file:
        expected_aht_df = load_expected_aht(expected_file)
        st.success(f"✅ Loaded {len(expected_aht_df)} workflow AHT goals")
    else:
        expected_aht_df = compute_network_expected_aht(df)
        st.info(
            f"ℹ️ No goals file — using network weighted mean AHT as baseline "
            f"({len(expected_aht_df)} workflow+locale combos)"
        )

    st.markdown("---")

    # --- Settings ---
    st.markdown("### ⚙️ Report Settings")

    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        agg_mode = st.radio(
            "Aggregation level:",
            options=["👤 Person", "🏢 Site", "🌍 Region"],
            horizontal=True,
            help="Person: per-DA/QA. Site: all DAs at a site combined. Region: all sites in a region combined.",
        )

    # Person-level settings
    person_col = None
    manager_col_use = None
    filter_persons = []
    filter_sites = []
    filter_regions = []

    if "Person" in agg_mode:
        with col_s2:
            person_options = {}
            if "da_name" in df.columns and df["da_name"].notna().any():
                person_options["Data Associate (DA)"] = "da_name"
            if "team_manager" in df.columns and df["team_manager"].notna().any():
                person_options["Team Manager"] = "team_manager"
            if "ops_manager" in df.columns and df["ops_manager"].notna().any():
                person_options["Ops Manager"] = "ops_manager"

            if not person_options:
                st.error("No person-level columns found in data.")
                return

            person_label = st.selectbox("Report by:", options=list(person_options.keys()))
            person_col = person_options[person_label]

        with col_s3:
            manager_options = {"(None)": None}
            if "team_manager" in df.columns and df["team_manager"].notna().any() and person_col != "team_manager":
                manager_options["Team Manager"] = "team_manager"
            if "ops_manager" in df.columns and df["ops_manager"].notna().any() and person_col != "ops_manager":
                manager_options["Ops Manager"] = "ops_manager"

            manager_label = st.selectbox("Manager column:", options=list(manager_options.keys()))
            manager_col = manager_options[manager_label]

        # Person filter
        all_persons = sorted(
            [str(x) for x in df[person_col].dropna().unique() if str(x) not in ("nan", "")]
        )
        filter_persons = st.multiselect(
            f"Filter to specific {person_label}s (optional):",
            options=all_persons,
            default=[],
            help="Leave empty to include everyone.",
        )

    elif "Site" in agg_mode:
        with col_s2:
            all_sites = sorted(df["site"].unique())
            filter_sites = st.multiselect(
                "Filter to specific sites (optional):",
                options=all_sites,
                default=[],
                help="Leave empty to include all sites.",
            )
        with col_s3:
            st.empty()

    else:  # Region
        with col_s2:
            all_regions = sorted(df["region"].unique())
            filter_regions = st.multiselect(
                "Filter to specific regions (optional):",
                options=all_regions,
                default=[],
                help="Leave empty to include all regions.",
            )
        with col_s3:
            st.empty()

    # Week filter
    all_weeks_sorted = (
        df[["week_label", "period_sort"]]
        .drop_duplicates()
        .sort_values("period_sort")["week_label"]
        .tolist()
    )

    if len(all_weeks_sorted) >= 2:
        start_wk, end_wk = st.select_slider(
            "Week range:",
            options=all_weeks_sorted,
            value=(all_weeks_sorted[0], all_weeks_sorted[-1]),
        )
        start_idx = all_weeks_sorted.index(start_wk)
        end_idx = all_weeks_sorted.index(end_wk)
        selected_weeks = all_weeks_sorted[start_idx : end_idx + 1]
    else:
        selected_weeks = all_weeks_sorted

    st.markdown("---")

    # --- Generate Report ---
    if st.button("📊 Generate Report", type="primary", use_container_width=True):

        df_report = df[df["week_label"].isin(selected_weeks)].copy()

        if "Person" in agg_mode:
            # Apply person filter
            if filter_persons:
                df_report = df_report[df_report[person_col].isin(filter_persons)]
            df_report = df_report.dropna(subset=[person_col])

            if manager_col is None:
                manager_col_use = "_manager_placeholder"
                df_report[manager_col_use] = "—"
            else:
                manager_col_use = manager_col
                df_report = df_report.dropna(subset=[manager_col_use])

            if df_report.empty:
                st.error("No data after applying filters.")
                return

            with st.spinner("Generating report..."):
                report_df, week_order = generate_adherence_report_person(
                    df_report, expected_aht_df, person_col, manager_col_use
                )

            fixed_cols = ["DA/QA Alias", "Manager", "Site"]
            excel_agg_mode = "person"

        elif "Site" in agg_mode:
            if filter_sites:
                df_report = df_report[df_report["site"].isin(filter_sites)]

            if df_report.empty:
                st.error("No data after applying filters.")
                return

            with st.spinner("Generating report..."):
                report_df, week_order = generate_adherence_report_site(
                    df_report, expected_aht_df, include_region=True
                )

            fixed_cols = ["Site", "Region"]
            excel_agg_mode = "site"

        else:  # Region
            if filter_regions:
                df_report = df_report[df_report["region"].isin(filter_regions)]

            if df_report.empty:
                st.error("No data after applying filters.")
                return

            with st.spinner("Generating report..."):
                report_df, week_order = generate_adherence_report_region(
                    df_report, expected_aht_df
                )

            fixed_cols = ["Region"]
            excel_agg_mode = "region"

        if report_df.empty:
            st.error(
                "No matching data found. Check that your data and goals files "
                "have overlapping workflow+locale combinations."
            )
            return

        # --- Summary metrics ---
        st.markdown("### 📊 Report Results")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            if "Person" in agg_mode:
                st.metric("People", report_df["DA/QA Alias"].nunique())
            elif "Site" in agg_mode:
                st.metric("Sites", report_df["Site"].nunique())
            else:
                st.metric("Regions", report_df["Region"].nunique())

        with col_m2:
            st.metric("Weeks", len(week_order))

        with col_m3:
            adh_cols = [f"{wk}|Adherence%" for wk in week_order]
            overall_adh = report_df[adh_cols].mean().mean()
            st.metric("Avg Adherence", f"{overall_adh:.0%}")

        with col_m4:
            audited_cols = [f"{wk}|#audited" for wk in week_order]
            total_audited = report_df[audited_cols].sum().sum()
            st.metric("Total WF+Locale Audited", f"{int(total_audited):,}")

        # --- Display table ---
        st.markdown("#### Adherence Table")

        display_df = report_df.copy()

        rename_map = {}
        for wk in week_order:
            rename_map[f"{wk}|#audited"] = f"{wk} | # Audited"
            rename_map[f"{wk}|#AHT>WF AHT"] = f"{wk} | # > Goal"
            rename_map[f"{wk}|#AHT<=WF AHT"] = f"{wk} | # ≤ Goal"
            rename_map[f"{wk}|Adherence%"] = f"{wk} | Adherence %"
            if f"{wk}|Units" in display_df.columns:
                rename_map[f"{wk}|Units"] = f"{wk} | Units"
            if f"{wk}|DAs" in display_df.columns:
                rename_map[f"{wk}|DAs"] = f"{wk} | # DAs"
            if f"{wk}|Sites" in display_df.columns:
                rename_map[f"{wk}|Sites"] = f"{wk} | # Sites"

        display_df = display_df.rename(columns=rename_map)

        format_dict = {}
        for wk in week_order:
            col_name = f"{wk} | Adherence %"
            if col_name in display_df.columns:
                format_dict[col_name] = "{:.0%}"

        styled = display_df.style.format(format_dict, na_rep="—")

        def color_adherence(val):
            if pd.isna(val):
                return ""
            if isinstance(val, (int, float)):
                if val >= 0.8:
                    return "background-color: #C6EFCE; color: #006100"
                elif val < 0.6:
                    return "background-color: #FFC7CE; color: #9C0006"
                else:
                    return "background-color: #FFEB9C; color: #9C6500"
            return ""

        adh_display_cols = [f"{wk} | Adherence %" for wk in week_order if f"{wk} | Adherence %" in display_df.columns]
        styled = styled.map(color_adherence, subset=adh_display_cols)

        st.dataframe(styled, use_container_width=True, hide_index=True)

        # --- Downloads ---
        st.markdown("#### 📥 Download")

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            csv_data = display_df.to_csv(index=False)
            st.download_button(
                "💾 Download CSV",
                csv_data,
                f"aht_adherence_{excel_agg_mode}_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True,
            )

        with col_d2:
            excel_data = build_excel_report(report_df, week_order, fixed_cols, excel_agg_mode)
            st.download_button(
                "📊 Download Excel (Formatted)",
                excel_data,
                f"aht_adherence_{excel_agg_mode}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.ml.sheet",
                use_container_width=True,
            )

        # --- Drill-down ---
        with st.expander("🔍 Drill-down: Workflow-level detail", expanded=False):

            if "Person" in agg_mode:
                drill_entity = st.selectbox(
                    "Select person:",
                    options=sorted(report_df["DA/QA Alias"].unique()),
                    key="drill_entity",
                )
                drill_filter_col = person_col
                drill_filter_val = drill_entity
            elif "Site" in agg_mode:
                drill_entity = st.selectbox(
                    "Select site:",
                    options=sorted(report_df["Site"].unique()),
                    key="drill_entity",
                )
                drill_filter_col = "site"
                drill_filter_val = drill_entity
            else:
                drill_entity = st.selectbox(
                    "Select region:",
                    options=sorted(report_df["Region"].unique()),
                    key="drill_entity",
                )
                drill_filter_col = "region"
                drill_filter_val = drill_entity

            drill_week = st.selectbox(
                "Select week:",
                options=week_order,
                key="drill_week",
            )

            df_drill = df_report[
                (df_report[drill_filter_col] == drill_filter_val)
                & (df_report["week_label"] == drill_week)
            ]

            if df_drill.empty:
                st.info("No data for this selection.")
            else:
                merged_drill = df_drill.merge(expected_aht_df, on="workflow_locale", how="inner")

                if merged_drill.empty:
                    st.info("No matching workflow goals for this selection's workflows.")
                else:
                    detail = (
                        merged_drill.groupby("workflow_locale")
                        .agg(
                            actual_aht=("aht", lambda x: np.average(
                                x, weights=merged_drill.loc[x.index, "processed_units"]
                            )),
                            expected_aht=("expected_aht", "first"),
                            units=("processed_units", "sum"),
                        )
                        .reset_index()
                    )
                    detail["meets_goal"] = detail["actual_aht"] <= detail["expected_aht"]
                    detail["difference"] = detail["actual_aht"] - detail["expected_aht"]
                    detail["status"] = detail["meets_goal"].map(
                        {True: "✅ ≤ Goal", False: "❌ > Goal"}
                    )

                    detail = detail.sort_values("meets_goal")

                    st.dataframe(
                        detail[
                            ["workflow_locale", "actual_aht", "expected_aht",
                             "difference", "units", "status"]
                        ]
                        .rename(columns={
                            "workflow_locale": "Workflow | Locale",
                            "actual_aht": "Actual AHT (s)",
                            "expected_aht": "Expected AHT (s)",
                            "difference": "Diff (s)",
                            "units": "Units",
                            "status": "Status",
                        })
                        .style.format({
                            "Actual AHT (s)": "{:.1f}",
                            "Expected AHT (s)": "{:.1f}",
                            "Diff (s)": "{:+.1f}",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )


if __name__ == "__main__":
    main()
