from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "final_crop.csv"
MODEL_FILES = {
    "model": APP_DIR / "crop_model.pkl",
    "scaler": APP_DIR / "scaler.pkl",
    "crop_encoder": APP_DIR / "crop_encoder.pkl",
    "district_encoder": APP_DIR / "dist_encoder.pkl",
}

DISTRICT_SPECIALTY_NOTES = {
    "Ahmednagar": {
        "headline": "Dryland cereals and sugarcane remain classic Ahmednagar strengths.",
        "detail": "Official gazetteer material highlights jowar and bajri as major staples, with wheat, gram, cotton and sugarcane also important in the district's farm economy.",
        "source": "https://gazetteers.maharashtra.gov.in/cultural.maharashtra.gov.in/english/gazetteer/Ahmadnagar/agri_seasons.html",
    },
    "Akola": {
        "headline": "Akola is strongly associated with jowar-cotton rotations.",
        "detail": "Official gazetteer references describe jowar and cotton as the main kharif crops, with tur and til also occupying significant area.",
        "source": "https://gazetteers.maharashtra.gov.in/cultural.maharashtra.gov.in/english/gazetteer/Akola%20District/agriculture_crops.html",
    },
    "Beed": {
        "headline": "Beed is known for cotton-led dryland agriculture.",
        "detail": "The district overview highlights abundant crops of cotton, sunflower, sugarcane and groundnut, while the agriculture pages also point to jowar and bajra as important field crops.",
        "source": "https://www.gazetteers.maharashtra.gov.in/cultural.maharashtra.gov.in/english/gazetteer/Beed/about_bhir.html",
    },
    "Bhandara": {
        "headline": "Bhandara has a strong rice identity.",
        "detail": "Official district pages describe Bhandara as famous for a bumper rice crop, and gazetteer agriculture notes also emphasize rice and tur in the kharif season.",
        "source": "https://gazetteers.maharashtra.gov.in/cultural.maharashtra.gov.in/english/gazetteer/BHANDARA/about_bhandara.html",
    },
    "Buldhana": {
        "headline": "Buldhana is closely tied to hybrid jowar and cotton belts.",
        "detail": "The official district profile specifically calls out bumper production of hybrid jowar, cotton, udad and moong, with soyabean and tur also listed among the main crops.",
        "source": "https://gazetteers.maharashtra.gov.in/cultural.maharashtra.gov.in/english/gazetteer/BULDHANA/about_buldhana.html",
    },
    "Chandrapur": {
        "headline": "Chandrapur mixes paddy areas with major cotton and soybean zones.",
        "detail": "Official district information lists rice, cotton and soyabean as the main crops, while agriculture pages also mention tur and jowar across the seasonal pattern.",
        "source": "https://gazetteers.maharashtra.gov.in/cultural.maharashtra.gov.in/english/gazetteer/CHANDRAPUR/about_chandrapur.html",
    },
    "Jalgaon": {
        "headline": "Jalgaon stands out for banana along with staple cereals and cotton.",
        "detail": "Official gazetteer text describes Jalgaon as one of Maharashtra's important banana-producing districts, alongside jowar, bajri, wheat, sugarcane, groundnut and cotton.",
        "source": "https://gazetteers.maharashtra.gov.in/cultural.maharashtra.gov.in/english/gazetteer/JALGAON/gen_eco_food_crops.html",
    },
    "Kolhapur": {
        "headline": "Kolhapur combines rice country with sugarcane and groundnut strength.",
        "detail": "Official gazetteer material highlights rice, jowar and other cereals, while also noting sugarcane as a major irrigated crop and groundnut as a key oilseed.",
        "source": "https://www.gazetteers.maharashtra.gov.in/cultural.maharashtra.gov.in/english/gazetteer/KOLHAPUR/gen_eco_agri.html",
    },
    "Nagpur": {
        "headline": "Nagpur pairs diversified field crops with its orange identity.",
        "detail": "Official district pages note cotton, sorghum, tur, soybean and groundnut in several talukas, while the district gazetteer also calls out Nagpur oranges as a signature specialty.",
        "source": "https://zpnagpur.maharashtra.gov.in/en/department-of-agriculture/",
    },
    "Pune": {
        "headline": "Pune's local profile is typically cereal-plus-sugarcane oriented.",
        "detail": "Historical gazetteer references for neighboring dryland belts and Pune's broader farm pattern point to jowar, bajri, wheat and sugarcane as key anchors, supported by horticulture and vegetable zones.",
        "source": "https://gazetteers.maharashtra.gov.in/cultural.maharashtra.gov.in/english/gazetteer/Ahmednagar%20District/agriculture_crop.html",
    },
    "Ratnagiri": {
        "headline": "Ratnagiri is a horticulture-heavy district led by mango and cashew.",
        "detail": "Official district information names Ratnagiri as Maharashtra's horticulture district and highlights Alphonso mango, cashew, jackfruit and coconut as signature crops.",
        "source": "https://www.gazetteers.maharashtra.gov.in/cultural.maharashtra.gov.in/english/gazetteer/RATNAGIRI/about_ratnagiri.html",
    },
    "Solapur": {
        "headline": "Solapur remains strongly identified with jowar, wheat and sugarcane.",
        "detail": "The official district profile lists jowar, wheat and sugarcane as the main crops, and seasonal notes also point to bajri, groundnut, cotton and tur in kharif areas.",
        "source": "https://www.gazetteers.maharashtra.gov.in/cultural.maharashtra.gov.in/english/gazetteer/Solapur/about_solapur.html",
    },
}


st.set_page_config(
    page_title="Crop Yield Planner",
    page_icon=":seedling:",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --bg: #f4efe6;
        --surface: rgba(255, 252, 246, 0.92);
        --surface-strong: #fffdf8;
        --ink: #1d3528;
        --muted: #587062;
        --line: rgba(29, 53, 40, 0.12);
        --accent: #2f7d4d;
        --accent-2: #c86b2a;
        --shadow: 0 20px 50px rgba(43, 57, 44, 0.12);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(200, 107, 42, 0.10), transparent 30%),
            radial-gradient(circle at top right, rgba(47, 125, 77, 0.14), transparent 32%),
            linear-gradient(180deg, #f2ecdf 0%, #f7f2e8 45%, #f4efe6 100%);
        color: var(--ink);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .hero {
        padding: 2rem 2.1rem;
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(255, 253, 248, 0.97), rgba(239, 244, 232, 0.92));
        border: 1px solid rgba(29, 53, 40, 0.08);
        box-shadow: var(--shadow);
        margin-bottom: 1.25rem;
    }

    .eyebrow {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(47, 125, 77, 0.10);
        color: var(--accent);
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .hero h1 {
        margin: 0.8rem 0 0.4rem 0;
        font-size: 2.8rem;
        line-height: 1;
        color: var(--ink);
    }

    .hero p {
        margin: 0;
        color: var(--muted);
        font-size: 1.05rem;
        max-width: 760px;
    }

    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
        margin: 1rem 0 0 0;
    }

    .stat-card, .info-card, .result-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 22px;
        box-shadow: var(--shadow);
    }

    .stat-card {
        padding: 1rem 1.1rem;
    }

    .stat-card .label {
        color: var(--muted);
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .stat-card .value {
        color: var(--ink);
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }

    .section-title {
        color: var(--ink);
        font-size: 1.25rem;
        font-weight: 700;
        margin: 1rem 0 0.75rem 0;
    }

    .info-card {
        padding: 1.1rem 1.2rem;
        margin-bottom: 1rem;
    }

    .result-card {
        padding: 1rem 1.1rem;
        margin-bottom: 0.9rem;
        background: linear-gradient(145deg, rgba(255, 252, 246, 0.98), rgba(239, 247, 240, 0.92));
    }

    .result-rank {
        color: var(--accent-2);
        text-transform: uppercase;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
    }

    .result-name {
        color: var(--ink);
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }

    .result-score {
        color: var(--accent);
        font-size: 1rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }

    .result-copy {
        color: var(--muted);
        margin-top: 0.35rem;
        line-height: 1.5;
    }

    .signal-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
        margin: 0.5rem 0 1rem 0;
    }

    .signal-card {
        background: linear-gradient(155deg, rgba(255, 251, 243, 0.98), rgba(236, 244, 231, 0.94));
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        box-shadow: var(--shadow);
    }

    .signal-label {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.74rem;
        font-weight: 700;
    }

    .signal-value {
        color: var(--ink);
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }

    .signal-copy {
        color: var(--muted);
        margin-top: 0.35rem;
        line-height: 1.45;
    }

    .spotlight {
        padding: 1.15rem 1.2rem;
        border-radius: 24px;
        border: 1px solid rgba(47, 125, 77, 0.16);
        background: linear-gradient(135deg, rgba(241, 249, 240, 0.95), rgba(255, 249, 239, 0.96));
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
    }

    .spotlight-kicker {
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.76rem;
        font-weight: 700;
    }

    .spotlight-title {
        color: var(--ink);
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    .spotlight-copy {
        color: var(--muted);
        line-height: 1.5;
        margin-top: 0.4rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255, 251, 245, 0.98), rgba(238, 244, 233, 0.95));
        border-right: 1px solid rgba(29, 53, 40, 0.06);
    }

    div[data-testid="stMetric"] {
        background: var(--surface-strong);
        border: 1px solid var(--line);
        padding: 0.9rem 1rem;
        border-radius: 18px;
    }

    @media (max-width: 900px) {
        .hero h1 {
            font-size: 2.2rem;
        }

        .stat-grid {
            grid-template-columns: 1fr;
        }

        .signal-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    if data.columns[0].startswith("Unnamed") or data.columns[0] == "":
        data = data.drop(columns=data.columns[0])

    numeric_columns = [
        "Year",
        "Area(1000 ha)",
        "Production(1000 tons)",
        "Yield(Kg per ha)",
        "Total Rainfall",
        "Avg Temp",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["Crop"] = data["Crop"].astype(str).str.strip()
    data["Dist Name"] = data["Dist Name"].astype(str).str.strip()
    return data.dropna(
        subset=["Year", "Dist Name", "Crop", "Area(1000 ha)", "Yield(Kg per ha)", "Total Rainfall", "Avg Temp"]
    )


@st.cache_data(show_spinner=False)
def build_crop_profiles(data: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        data.groupby(["Dist Name", "Crop"], as_index=False)
        .agg(
            avg_yield=("Yield(Kg per ha)", "mean"),
            avg_area=("Area(1000 ha)", "mean"),
            avg_rainfall=("Total Rainfall", "mean"),
            avg_temp=("Avg Temp", "mean"),
            records=("Crop", "count"),
        )
    )

    grouped["yield_score"] = grouped.groupby("Dist Name")["avg_yield"].transform(
        lambda series: (series - series.min()) / ((series.max() - series.min()) or 1)
    )
    grouped["area_score"] = grouped.groupby("Dist Name")["avg_area"].transform(
        lambda series: (series - series.min()) / ((series.max() - series.min()) or 1)
    )
    return grouped


@st.cache_data(show_spinner=False)
def district_baselines(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby("Dist Name", as_index=False)
        .agg(
            typical_area=("Area(1000 ha)", "median"),
            typical_rainfall=("Total Rainfall", "median"),
            typical_temp=("Avg Temp", "median"),
            typical_yield=("Yield(Kg per ha)", "median"),
            crop_count=("Crop", "nunique"),
        )
        .sort_values("Dist Name")
    )


@st.cache_data(show_spinner=False)
def district_crop_preferences(data: pd.DataFrame) -> pd.DataFrame:
    ranked = (
        data.groupby(["Dist Name", "Crop"], as_index=False)
        .agg(
            avg_yield=("Yield(Kg per ha)", "mean"),
            avg_area=("Area(1000 ha)", "mean"),
            observations=("Crop", "count"),
        )
    )
    ranked["yield_score"] = ranked.groupby("Dist Name")["avg_yield"].transform(
        lambda series: (series - series.min()) / ((series.max() - series.min()) or 1)
    )
    ranked["area_score"] = ranked.groupby("Dist Name")["avg_area"].transform(
        lambda series: (series - series.min()) / ((series.max() - series.min()) or 1)
    )
    ranked["preference_score"] = (
        ranked["yield_score"] * 0.55
        + ranked["area_score"] * 0.30
        + (ranked["observations"] / ranked.groupby("Dist Name")["observations"].transform("max")) * 0.15
    )
    return ranked


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    if not all(path.exists() for path in MODEL_FILES.values()):
        return None

    return {
        "model": joblib.load(MODEL_FILES["model"]),
        "scaler": joblib.load(MODEL_FILES["scaler"]),
        "crop_encoder": joblib.load(MODEL_FILES["crop_encoder"]),
        "district_encoder": joblib.load(MODEL_FILES["district_encoder"]),
    }


def heuristic_recommendations(
    data: pd.DataFrame,
    district: str,
    area: float,
    rainfall: float,
    temperature: float,
) -> pd.DataFrame:
    scoped = data[data["Dist Name"] == district].copy()
    if scoped.empty:
        return pd.DataFrame()

    rainfall_span = max(scoped["Total Rainfall"].max() - scoped["Total Rainfall"].min(), 1.0)
    temp_span = max(scoped["Avg Temp"].max() - scoped["Avg Temp"].min(), 1.0)
    area_span = max(scoped["Area(1000 ha)"].max() - scoped["Area(1000 ha)"].min(), 1.0)

    scoped["rainfall_fit"] = 1 - (scoped["Total Rainfall"].sub(rainfall).abs() / rainfall_span).clip(upper=1)
    scoped["temp_fit"] = 1 - (scoped["Avg Temp"].sub(temperature).abs() / temp_span).clip(upper=1)
    scoped["area_fit"] = 1 - (scoped["Area(1000 ha)"].sub(area).abs() / area_span).clip(upper=1)
    scoped["condition_fit"] = (
        scoped["rainfall_fit"] * 0.45
        + scoped["temp_fit"] * 0.35
        + scoped["area_fit"] * 0.20
    )

    crop_rank = (
        scoped.groupby("Crop", as_index=False)
        .agg(
            matched_yield=("Yield(Kg per ha)", "mean"),
            matched_area=("Area(1000 ha)", "mean"),
            matched_rainfall=("Total Rainfall", "mean"),
            matched_temp=("Avg Temp", "mean"),
            condition_fit=("condition_fit", "mean"),
            supporting_records=("Crop", "count"),
        )
    )

    crop_rank["yield_score"] = (
        (crop_rank["matched_yield"] - crop_rank["matched_yield"].min())
        / ((crop_rank["matched_yield"].max() - crop_rank["matched_yield"].min()) or 1)
    )
    crop_rank["support_score"] = (
        (crop_rank["supporting_records"] - crop_rank["supporting_records"].min())
        / ((crop_rank["supporting_records"].max() - crop_rank["supporting_records"].min()) or 1)
    )
    crop_rank["score"] = (
        crop_rank["condition_fit"] * 0.60
        + crop_rank["yield_score"] * 0.30
        + crop_rank["support_score"] * 0.10
    )
    crop_rank["confidence"] = (crop_rank["score"] * 100).clip(lower=1, upper=99)
    crop_rank["source"] = "Historical climate match"
    return crop_rank.sort_values(["score", "supporting_records"], ascending=[False, False]).head(10)


def model_recommendations(
    bundle,
    district: str,
    area: float,
    rainfall: float,
    temperature: float,
) -> pd.DataFrame:
    reference_year = pd.Timestamp.now().year
    district_encoded = bundle["district_encoder"].transform([district])[0]
    input_frame = pd.DataFrame(
        [[reference_year, district_encoded, area, rainfall, temperature]],
        columns=["Year", "Dist Name", "Area(1000 ha)", "Total Rainfall", "Avg Temp"],
    )
    scaled = bundle["scaler"].transform(input_frame)
    probabilities = bundle["model"].predict_proba(scaled)[0]
    crop_labels = bundle["crop_encoder"].inverse_transform(bundle["model"].classes_)

    prediction_frame = pd.DataFrame(
        {"Crop": crop_labels, "confidence": probabilities * 100, "source": "Trained prediction model"}
    )
    return prediction_frame.sort_values("confidence", ascending=False).head(10)


def recommendation_copy(row: pd.Series, rainfall: float, temperature: float) -> str:
    details = []
    if "matched_yield" in row:
        details.append(f"Historic yield around {row['matched_yield']:.0f} kg/ha")
    elif "avg_yield" in row:
        details.append(f"Historic yield around {row['avg_yield']:.0f} kg/ha")

    if "matched_rainfall" in row:
        details.append(f"matches seasons near {row['matched_rainfall']:.0f} mm rainfall")
    elif "avg_rainfall" in row:
        details.append(f"works well near {row['avg_rainfall']:.0f} mm rainfall")

    if "matched_temp" in row:
        details.append(f"and around {row['matched_temp']:.1f} C average temperature")
    elif "avg_temp" in row:
        details.append(f"performs around {row['avg_temp']:.1f} C average temperature")

    if not details:
        details = [
            f"Recommended for roughly {rainfall:.0f} mm rainfall",
            f"and about {temperature:.1f} C average temperature",
        ]

    return " | ".join(details)


def describe_condition(value: float, baseline: float, tolerance: float, low_label: str, high_label: str) -> str:
    if value < baseline - tolerance:
        return low_label
    if value > baseline + tolerance:
        return high_label
    return "Near district norm"


def land_scale_note(area: float, baseline: float) -> str:
    if area < baseline * 0.75:
        return "Compact footprint"
    if area > baseline * 1.25:
        return "Broad acreage"
    return "Typical district scale"


def district_specialty_block(preferences: pd.DataFrame, district: str):
    district_pref = preferences[preferences["Dist Name"] == district].sort_values(
        ["preference_score", "avg_area"], ascending=[False, False]
    )
    top_crops = district_pref.head(3)["Crop"].tolist()
    local_profile = ", ".join(top_crops) if top_crops else "No clear local pattern"
    official_note = DISTRICT_SPECIALTY_NOTES.get(district)
    return official_note, local_profile


data = load_dataset()
profiles = build_crop_profiles(data)
baseline_df = district_baselines(data)
preference_df = district_crop_preferences(data)
model_bundle = load_model_bundle()

district_options = baseline_df["Dist Name"].tolist()
district_default = district_options.index("Pune") if "Pune" in district_options else 0


with st.sidebar:
    st.markdown("## Shape your crop plan")
    st.write("Tune local field conditions and the shortlist updates live.")

    district = st.selectbox("District", district_options, index=district_default)
    district_snapshot = baseline_df.loc[baseline_df["Dist Name"] == district].iloc[0]

    area = st.number_input(
        "Farm area (1000 ha)",
        min_value=0.1,
        value=float(round(district_snapshot["typical_area"], 1)),
        step=0.1,
        help="Enter the land area using the same unit used in the training dataset.",
    )
    rainfall = st.slider(
        "Expected rainfall (mm)",
        min_value=int(data["Total Rainfall"].min()),
        max_value=int(data["Total Rainfall"].max()),
        value=int(round(district_snapshot["typical_rainfall"])),
    )
    temperature = st.slider(
        "Average temperature (C)",
        min_value=float(np.floor(data["Avg Temp"].min())),
        max_value=50.0,
        value=float(round(district_snapshot["typical_temp"], 1)),
        step=0.1,
    )

    st.markdown("---")
    st.caption("Tip: start from your district defaults, then shift rainfall and temperature to match the field outlook.")


st.markdown(
    f"""
    <section class="hero">
        <span class="eyebrow">User-first crop guidance</span>
        <h1>Plan the field, then let the crop shortlist respond</h1>
        <p>
            This planner combines district history with live field inputs so a farmer, agronomist, or field operator
            can explore practical crop options through an interactive, agriculture-focused planning experience.
        </p>
        <div class="stat-grid">
            <div class="stat-card">
                <div class="label">Districts covered</div>
                <div class="value">{baseline_df['Dist Name'].nunique()}</div>
            </div>
            <div class="stat-card">
                <div class="label">Crop records used</div>
                <div class="value">{len(data):,}</div>
            </div>
            <div class="stat-card">
                <div class="label">Prediction mode</div>
                <div class="value">{"Model + fallback" if model_bundle else "Historical fallback"}</div>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)


info_col, metric_col = st.columns([1.4, 1])
with info_col:
    st.markdown('<div class="section-title">What this app helps with</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card">
            <strong>Work from field conditions, not dataset years.</strong><br>
            The experience is now centered on area, rainfall, and temperature, which makes more sense for real crop planning.
        </div>
        <div class="info-card">
            <strong>See recommendations update live.</strong><br>
            The shortlist refreshes as soon as you adjust the controls, so exploring scenarios feels immediate.
        </div>
        <div class="info-card">
            <strong>Read the field signals quickly.</strong><br>
            The dashboard translates your inputs into rainfall, heat, and land-scale cues before ranking crops.
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric_col:
    st.markdown('<div class="section-title">Current district snapshot</div>', unsafe_allow_html=True)
    st.metric("District", district)
    st.metric("Typical rainfall", f"{district_snapshot['typical_rainfall']:.0f} mm")
    st.metric("Typical temperature", f"{district_snapshot['typical_temp']:.1f} C")
    st.metric("Known crops", int(district_snapshot["crop_count"]))

official_note, local_profile = district_specialty_block(preference_df, district)
st.markdown('<div class="section-title">District specialty</div>', unsafe_allow_html=True)
if official_note:
    st.info(
        f"{official_note['headline']} {official_note['detail']} Local dataset leaders here: {local_profile}."
    )
    st.caption(f"Official reference: {official_note['source']}")
else:
    st.info(
        f"Official district specialty note is not yet curated for {district}. Local dataset leaders in this app are: {local_profile}."
    )

rainfall_signal = describe_condition(
    rainfall,
    float(district_snapshot["typical_rainfall"]),
    120.0,
    "Drier than usual",
    "Wetter than usual",
)
temperature_signal = describe_condition(
    temperature,
    float(district_snapshot["typical_temp"]),
    1.5,
    "Cooler than usual",
    "Warmer than usual",
)
area_signal = land_scale_note(area, float(district_snapshot["typical_area"]))

st.markdown('<div class="section-title">Field signals</div>', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="signal-grid">
        <div class="signal-card">
            <div class="signal-label">Moisture signal</div>
            <div class="signal-value">{rainfall_signal}</div>
            <div class="signal-copy">Input rainfall: {rainfall:.0f} mm compared with a district norm of {district_snapshot['typical_rainfall']:.0f} mm.</div>
        </div>
        <div class="signal-card">
            <div class="signal-label">Heat signal</div>
            <div class="signal-value">{temperature_signal}</div>
            <div class="signal-copy">Input temperature: {temperature:.1f} C compared with a district norm of {district_snapshot['typical_temp']:.1f} C.</div>
        </div>
        <div class="signal-card">
            <div class="signal-label">Land footprint</div>
            <div class="signal-value">{area_signal}</div>
            <div class="signal-copy">Selected area: {area:.1f} in the dataset's `Area(1000 ha)` unit.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if model_bundle and district in model_bundle["district_encoder"].classes_:
    recommendations = model_recommendations(
        model_bundle,
        district=district,
        area=area,
        rainfall=rainfall,
        temperature=temperature,
    )
else:
    recommendations = heuristic_recommendations(
        data,
        district=district,
        area=area,
        rainfall=rainfall,
        temperature=temperature,
    )

st.markdown('<div class="section-title">Recommended crops</div>', unsafe_allow_html=True)
if recommendations.empty:
    st.error("No recommendations could be generated for this district with the available data.")
else:
    recommendations = recommendations.reset_index(drop=True)
    recommendations["Rank"] = recommendations.index + 1
    top_pick = recommendations.reset_index(drop=True).iloc[0]
    st.markdown(
        f"""
        <div class="spotlight">
            <div class="spotlight-kicker">Best current fit</div>
            <div class="spotlight-title">{top_pick['Crop']}</div>
            <div class="spotlight-copy">{recommendation_copy(top_pick, rainfall=rainfall, temperature=temperature)}</div>
            <div class="spotlight-copy">Decision source: {top_pick['source']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(int(round(float(top_pick["confidence"]))))

    for index, row in recommendations.head(3).iterrows():
        crop_name = row["Crop"]
        confidence = row["confidence"]
        source = row["source"]
        copy = recommendation_copy(row, rainfall=rainfall, temperature=temperature)

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-rank">Option {index + 1}</div>
                <div class="result-name">{crop_name}</div>
                <div class="result-score">Suitability score: {confidence:.1f}%</div>
                <div class="result-copy">{copy}</div>
                <div class="result-copy">Decision source: {source}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Top 10 ranked crops</div>', unsafe_allow_html=True)
    ranking_table = recommendations[["Rank", "Crop", "confidence", "source"]].rename(
        columns={"confidence": "Suitability (%)"}
    )
    st.dataframe(ranking_table, use_container_width=True, hide_index=True)

    tab1, tab2, tab3 = st.tabs(["Comparison", "District history", "How it works"])

    with tab1:
        chart_df = recommendations.head(10)[["Crop", "confidence"]].set_index("Crop")
        st.bar_chart(chart_df)
        st.caption("The table above shows the full top-10 order for the current district and field conditions.")

    with tab2:
        district_history = (
            data[data["Dist Name"] == district]
            .groupby("Crop", as_index=False)
            .agg(
                avg_yield=("Yield(Kg per ha)", "mean"),
                avg_area=("Area(1000 ha)", "mean"),
                seasons=("Year", "nunique"),
            )
            .sort_values("avg_yield", ascending=False)
        )
        st.dataframe(district_history, use_container_width=True, hide_index=True)

    with tab3:
        if model_bundle:
            st.write(
                "The app uses the trained classifier when the model bundle is available for the selected district."
            )
        else:
            st.write(
                "The trained model files are not present in this repository, so recommendations currently come from historical district patterns."
            )
        st.write(
            "Fallback scores combine yield history with similarity to the entered rainfall, temperature, and area values."
        )
        st.write(
            "This interface is intentionally year-free so users can focus on field conditions and crop fit instead of a dataset timeline."
        )
