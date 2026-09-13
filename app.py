import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Mental Health in Tech — EDA Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#3498db"
GREEN = "#27ae60"
RED = "#e74c3c"
YELLOW = "#f39c12"

SIZE_ORDER = ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]
WI_ORDER = ["Never", "Rarely", "Sometimes", "Often", "Not applicable"]
LEAVE_ORDER = ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"]


# ----------------------------------------------------------------------------
# Data loading & cleaning (mirrors the source notebook)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data(path="survey.csv"):
    df = pd.read_csv(path)
    raw_shape = df.shape

    # Clean Age: keep plausible working ages, impute rest with median
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Age"] = df["Age"].apply(lambda x: x if 18 <= x <= 72 else np.nan)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Age"] = df["Age"].astype(int)

    # Clean Gender -> Male / Female / Other
    def clean_gender(g):
        if pd.isna(g):
            return "Other"
        g = str(g).strip().lower()
        male_terms = [
            "male", "m", "man", "cis male", "male ", "maile", "mal",
            "male (cis)", "make", "guy (-ish) ^_^", "msle", "mail",
            "cis man", "male-ish", "something kinda male?",
            "male leaning androgynous", "ostensibly male", "malr",
        ]
        female_terms = [
            "female", "f", "woman", "cis female", "female ", "femake",
            "female (cis)", "cis-female/femme", "femail",
            "female (trans)", "trans-female", "trans woman", "queer/she/they",
        ]
        if g in male_terms:
            return "Male"
        elif g in female_terms:
            return "Female"
        return "Other"

    df["Gender"] = df["Gender"].apply(clean_gender)

    df.drop(columns=["Timestamp", "comments"], errors="ignore", inplace=True)
    df["self_employed"] = df["self_employed"].fillna("No")
    df["work_interfere"] = df["work_interfere"].fillna("Not applicable")

    df["Age_Group"] = pd.cut(
        df["Age"], bins=[17, 25, 35, 45, 55, 75],
        labels=["18-25", "26-35", "36-45", "46-55", "56+"],
    )

    return df, raw_shape


@st.cache_data
def missing_summary(path="survey.csv"):
    raw = pd.read_csv(path)
    miss = raw.isnull().sum()
    pct = (miss / len(raw) * 100).round(2)
    out = pd.DataFrame({"Missing Count": miss, "Missing %": pct})
    return out[out["Missing Count"] > 0].sort_values("Missing %", ascending=False)


@st.cache_data
def build_correlation(df):
    df_enc = df.copy()
    binary_map = {"Yes": 1, "No": 0}
    for col in ["self_employed", "family_history", "treatment", "remote_work",
                "tech_company", "obs_consequence"]:
        df_enc[col] = df_enc[col].map(binary_map)

    df_enc["work_interfere"] = df_enc["work_interfere"].map(
        {"Not applicable": 0, "Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4}
    )
    df_enc["leave"] = df_enc["leave"].map(
        {"Don't know": 0, "Very difficult": 1, "Somewhat difficult": 2,
         "Somewhat easy": 3, "Very easy": 4}
    )
    tri_map = {"No": 0, "Maybe": 1, "Yes": 2}
    for col in ["mental_health_consequence", "phys_health_consequence", "coworkers",
                "supervisor", "mental_health_interview", "phys_health_interview"]:
        df_enc[col] = df_enc[col].map(tri_map)

    tri2_map = {"No": 0, "Don't know": 1, "Yes": 2}
    for col in ["benefits", "care_options", "wellness_program", "seek_help", "anonymity"]:
        df_enc[col] = df_enc[col].map(tri2_map)

    df_enc["mental_vs_physical"] = df_enc["mental_vs_physical"].map(
        {"Don't know": 0, "No": 1, "Yes": 2}
    )
    df_enc["Gender"] = df_enc["Gender"].map({"Male": 0, "Female": 1, "Other": 2})

    numeric_cols = df_enc.select_dtypes(include=[np.number]).columns.tolist()
    return df_enc[numeric_cols].corr()


def pct_stacked_bar(df, group_col, split_col, order=None, colors=None, title=""):
    ct = pd.crosstab(df[group_col], df[split_col], normalize="index") * 100
    if order is not None:
        ct = ct.reindex([o for o in order if o in ct.index])
    ct = ct.reset_index().melt(id_vars=group_col, var_name=split_col, value_name="Percent")
    fig = px.bar(
        ct, x=group_col, y="Percent", color=split_col, barmode="stack",
        color_discrete_sequence=colors, title=title,
    )
    fig.update_layout(yaxis_title="Percentage (%)", legend_title=split_col)
    return fig


def count_bar(series, order=None, colors=None, title="", xlabel=""):
    vc = series.value_counts()
    if order is not None:
        vc = vc.reindex([o for o in order if o in vc.index])
    fig = px.bar(
        x=vc.index.astype(str), y=vc.values, title=title,
        color=vc.index.astype(str),
        color_discrete_sequence=colors,
    )
    fig.update_layout(showlegend=False, xaxis_title=xlabel, yaxis_title="Count")
    return fig


# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
df, raw_shape = load_data()

st.title("🧠 Mental Health in Tech Survey — EDA Dashboard")
st.caption(
    f"2014 OSMI survey on attitudes toward mental health in the tech workplace • "
    f"{raw_shape[0]:,} respondents • {raw_shape[1]} original columns"
)

# ----------------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------------
st.sidebar.header("Filters")

gender_sel = st.sidebar.multiselect(
    "Gender", options=sorted(df["Gender"].unique()), default=sorted(df["Gender"].unique())
)
age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
age_sel = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

top_countries_all = df["Country"].value_counts().head(15).index.tolist()
country_sel = st.sidebar.multiselect(
    "Country (top 15 shown, empty = all)", options=top_countries_all, default=[]
)

tech_sel = st.sidebar.multiselect(
    "Tech company?", options=sorted(df["tech_company"].dropna().unique()),
    default=sorted(df["tech_company"].dropna().unique())
)

fdf = df[
    df["Gender"].isin(gender_sel)
    & df["Age"].between(age_sel[0], age_sel[1])
    & df["tech_company"].isin(tech_sel)
]
if country_sel:
    fdf = fdf[fdf["Country"].isin(country_sel)]

st.sidebar.markdown(f"**{len(fdf):,}** respondents match current filters")
if st.sidebar.button("Reset filters"):
    st.rerun()

if len(fdf) == 0:
    st.warning("No respondents match the current filters. Please broaden your selection.")
    st.stop()

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tabs = st.tabs([
    "Overview", "Demographics", "Treatment & Work",
    "Correlations", "Attitudes & Support", "Geography", "Key Findings",
])

# ---- Overview -------------------------------------------------------------
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents (filtered)", f"{len(fdf):,}")
    treat_rate = (fdf["treatment"] == "Yes").mean() * 100
    c2.metric("Sought treatment", f"{treat_rate:.1f}%")
    fam_rate = (fdf["family_history"] == "Yes").mean() * 100
    c3.metric("Family history of MH illness", f"{fam_rate:.1f}%")
    remote_rate = (fdf["remote_work"] == "Yes").mean() * 100
    c4.metric("Remote workers", f"{remote_rate:.1f}%")

    st.subheader("Missing values in the raw data")
    miss = missing_summary()
    if len(miss):
        fig = px.bar(
            miss, x=miss.index, y="Missing %",
            title="Missing Values by Column (raw survey.csv)",
            color_discrete_sequence=[RED],
        )
        fig.update_layout(xaxis_title="Column", yaxis_title="Missing %")
        st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Cleaning applied: Age clipped to 18–72 and median-imputed; Gender free-text "
        "standardized to Male/Female/Other; `self_employed` NAs → 'No'; "
        "`work_interfere` NAs → 'Not applicable'; Timestamp & comments dropped."
    )

    with st.expander("Preview cleaned data"):
        st.dataframe(fdf.head(50), use_container_width=True)

# ---- Demographics -----------------------------------------------------------
with tabs[1]:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            fdf, x="Age", nbins=30, title="Age Distribution",
            color_discrete_sequence=[PRIMARY],
        )
        fig.add_vline(x=fdf["Age"].mean(), line_dash="dash", line_color=RED,
                       annotation_text=f"Mean {fdf['Age'].mean():.1f}")
        fig.add_vline(x=fdf["Age"].median(), line_dash="dash", line_color=YELLOW,
                       annotation_text=f"Median {fdf['Age'].median():.0f}")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        gc = fdf["Gender"].value_counts()
        fig = px.pie(
            values=gc.values, names=gc.index, title="Gender Distribution",
            color_discrete_sequence=[PRIMARY, RED, GREEN],
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        top_c = fdf["Country"].value_counts().head(10)
        fig = px.bar(
            x=top_c.values, y=top_c.index, orientation="h",
            title="Top 10 Countries by Respondent Count",
            color_discrete_sequence=[PRIMARY],
        )
        fig.update_layout(xaxis_title="Count", yaxis_title="Country",
                           yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = count_bar(fdf["no_employees"], order=SIZE_ORDER,
                         title="Company Size Distribution", xlabel="Number of Employees")
        st.plotly_chart(fig, use_container_width=True)

# ---- Treatment & Work -------------------------------------------------------
with tabs[2]:
    col1, col2 = st.columns(2)
    with col1:
        tc = fdf["treatment"].value_counts()
        fig = px.pie(values=tc.values, names=tc.index, title="Have You Sought Treatment?",
                     color_discrete_sequence=[GREEN, RED])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = count_bar(fdf["work_interfere"], order=WI_ORDER,
                         title="Does MH Condition Interfere with Work?",
                         xlabel="Level of Interference")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = pct_stacked_bar(fdf, "Gender", "treatment", colors=[RED, GREEN],
                               title="Treatment Rate by Gender")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.violin(fdf, x="treatment", y="Age", color="treatment", box=True,
                         color_discrete_sequence=[RED, GREEN],
                         title="Age Distribution by Treatment Status")
        st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        ctf = pd.crosstab(fdf["family_history"], fdf["treatment"])
        fig = px.imshow(ctf, text_auto=True, color_continuous_scale="Blues",
                         title="Family History vs Treatment (Counts)")
        st.plotly_chart(fig, use_container_width=True)
    with col6:
        fig = pct_stacked_bar(fdf, "remote_work", "treatment", colors=[YELLOW, PRIMARY],
                               title="Treatment Rate by Remote Work Status")
        st.plotly_chart(fig, use_container_width=True)

    col7, col8 = st.columns(2)
    with col7:
        fig = pct_stacked_bar(fdf, "no_employees", "mental_health_consequence",
                               order=SIZE_ORDER, colors=[GREEN, YELLOW, RED],
                               title="Fear of Negative Consequences by Company Size")
        st.plotly_chart(fig, use_container_width=True)
    with col8:
        fig = pct_stacked_bar(fdf, "Age_Group", "treatment", colors=[RED, GREEN],
                               title="Treatment Rate by Age Group")
        st.plotly_chart(fig, use_container_width=True)

# ---- Correlations -----------------------------------------------------------
with tabs[3]:
    corr = build_correlation(fdf)
    st.subheader("Correlation Heatmap of All Encoded Features")
    fig = px.imshow(
        corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        aspect="auto",
    )
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Correlations with `treatment`")
    if "treatment" in corr.columns:
        tcorr = corr["treatment"].drop("treatment").sort_values()
        colors = [GREEN if v > 0 else RED for v in tcorr.values]
        fig2 = px.bar(x=tcorr.values, y=tcorr.index, orientation="h",
                      color=tcorr.index, color_discrete_sequence=colors)
        fig2.update_layout(showlegend=False, xaxis_title="Correlation coefficient",
                            yaxis_title="", height=600)
        st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Categorical fields are ordinally/binary encoded (e.g. treatment: Yes=1/No=0, "
        "work_interfere: Never→Often = 1→4) purely to compute Pearson correlation — "
        "not a causal or statistically validated relationship."
    )

# ---- Attitudes & Support -----------------------------------------------------
with tabs[4]:
    col1, col2 = st.columns(2)
    with col1:
        fig = count_bar(fdf["coworkers"], order=["No", "Some of them", "Yes"],
                         title="Discuss MH with Coworkers?")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = count_bar(fdf["supervisor"], order=["No", "Some of them", "Yes"],
                         title="Discuss MH with Supervisor?")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        mhi = fdf["mental_health_interview"].value_counts()
        fig = px.pie(values=mhi.values, names=mhi.index,
                     title="Bring Up Mental Health in Interview?",
                     color_discrete_sequence=[RED, YELLOW, GREEN])
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        obs = fdf["obs_consequence"].value_counts()
        fig = px.bar(x=obs.index, y=obs.values, title="Observed Negative Consequences for MH?",
                     color=obs.index, color_discrete_sequence=[GREEN, RED])
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Employer Resources Overview")
    support_cols = ["benefits", "care_options", "wellness_program", "seek_help", "anonymity", "leave"]
    titles = ["MH Benefits Provided?", "Know Care Options?", "Wellness Program?",
              "Employer Provides MH Resources?", "Anonymity Protected?", "Ease of Leave"]
    grid_cols = st.columns(3)
    for i, (col, title) in enumerate(zip(support_cols, titles)):
        order = LEAVE_ORDER if col == "leave" else None
        with grid_cols[i % 3]:
            fig = count_bar(fdf[col], order=order, title=title)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tech vs. Non-Tech Company")
    col5, col6 = st.columns(2)
    with col5:
        fig = pct_stacked_bar(fdf, "tech_company", "treatment", colors=[RED, GREEN],
                               title="Treatment Rate: Tech vs Non-Tech")
        st.plotly_chart(fig, use_container_width=True)
    with col6:
        fig = pct_stacked_bar(fdf, "tech_company", "benefits",
                               title="Benefits Availability: Tech vs Non-Tech")
        st.plotly_chart(fig, use_container_width=True)

# ---- Geography ---------------------------------------------------------------
with tabs[5]:
    top10 = fdf["Country"].value_counts().head(10).index
    df_top10 = fdf[fdf["Country"].isin(top10)]
    if len(df_top10):
        fig = pct_stacked_bar(df_top10, "Country", "treatment", colors=[RED, GREEN],
                               title="Treatment Rate by Country (Top 10)")
        st.plotly_chart(fig, use_container_width=True)

    us_df = fdf[fdf["Country"] == "United States"]
    if len(us_df):
        top_states = us_df["state"].value_counts().head(15)
        fig = px.bar(x=top_states.values, y=top_states.index, orientation="h",
                     title="Top 15 US States by Respondent Count",
                     color_discrete_sequence=[PRIMARY])
        fig.update_layout(xaxis_title="Count", yaxis_title="State",
                           yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No United States respondents in the current filter selection.")

# ---- Key Findings --------------------------------------------------------------
with tabs[6]:
    st.subheader("Key Findings (full, unfiltered dataset)")
    full_treat = (df["treatment"] == "Yes").mean() * 100
    full_male = (df["Gender"] == "Male").mean() * 100
    fam_treat = pd.crosstab(df["family_history"], df["treatment"], normalize="index")["Yes"] * 100
    top_size = df["no_employees"].value_counts().idxmax()
    top_wi = df["work_interfere"].value_counts().idxmax()
    dont_know_leave = (df["leave"] == "Don't know").mean() * 100
    interview_yes = (df["mental_health_interview"] == "Yes").mean() * 100
    gender_treat = pd.crosstab(df["Gender"], df["treatment"], normalize="index")["Yes"] * 100
    us_share = (df["Country"] == "United States").mean() * 100
    uk_share = (df["Country"] == "United Kingdom").mean() * 100
    tech_treat = pd.crosstab(df["tech_company"], df["treatment"], normalize="index")["Yes"] * 100

    findings = [
        f"~{full_treat:.0f}% of respondents have sought mental health treatment",
        f"~{full_male:.0f}% of respondents are Male",
        f"Family history is a strong predictor of seeking treatment "
        f"({fam_treat.get('Yes', float('nan')):.0f}% with history vs "
        f"{fam_treat.get('No', float('nan')):.0f}% without)",
        f"Most respondents work in companies with **{top_size}** employees",
        f"**\"{top_wi}\"** is the most common work-interference level",
        f"~{dont_know_leave:.0f}% of people **don't know** how easy it is to take MH leave",
        f"Only ~{interview_yes:.0f}% would bring up MH in a job interview",
        f"Treatment-seeking rate — Female: {gender_treat.get('Female', float('nan')):.0f}%, "
        f"Male: {gender_treat.get('Male', float('nan')):.0f}%, "
        f"Other: {gender_treat.get('Other', float('nan')):.0f}%",
        f"US dominates the survey (~{us_share:.0f}%), followed by UK (~{uk_share:.0f}%)",
        f"Tech companies treatment rate {tech_treat.get('Yes', float('nan')):.0f}% vs "
        f"non-tech {tech_treat.get('No', float('nan')):.0f}%",
    ]
    for f in findings:
        st.markdown(f"- {f}")

    st.divider()
    st.subheader("Summary Dashboard")
    c1, c2 = st.columns(2)
    with c1:
        tc = df["treatment"].value_counts()
        fig = px.pie(values=tc.values, names=tc.index, title="Sought Treatment (all data)",
                     color_discrete_sequence=[GREEN, RED])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = pct_stacked_bar(df, "family_history", "treatment", colors=[RED, GREEN],
                               title="Treatment by Family History (all data)")
        st.plotly_chart(fig, use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        fig = px.histogram(df, x="Age", nbins=25, title="Age Distribution (all data)",
                            color_discrete_sequence=[PRIMARY])
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        top5 = df["Country"].value_counts().head(5)
        fig = px.bar(x=top5.values, y=top5.index, orientation="h", title="Top 5 Countries (all data)",
                     color_discrete_sequence=[PRIMARY])
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Count", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(
    "Data: OSMI 2014 Mental Health in Tech Survey. This dashboard is for exploratory "
    "analysis only and is not a clinical or statistical inference tool."
)
