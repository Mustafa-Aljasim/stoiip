# STOIIP_streamlit_app.py
# Streamlit application to calculate STOIIP using the volumetric equation
# STOIIP (STB) = 7758 * A(acres) * h(ft) * phi(frac) * (1 - Sw(frac)) / Boi(bbl/STB)
# - Allows fixed inputs or ranges for Monte Carlo sampling
# - Computes P10, P50, P90 from the simulated STOIIP distribution

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="STOIIP Calculator", layout="wide")

st.title("STOIIP Calculator — Volumetric Method")
st.markdown(
    """
    This app computes STOIIP using the volumetric equation:

    **STOIIP (STB)** = 7758 × A(acres) × h(ft) × φ(fraction) × (1 − S_w(fraction)) / B_oi(bbl/STB)

    You can provide fixed values or ranges for any input. When ranges are given the app
    runs a Monte Carlo sampling (uniform by default, or triangular if you provide a mode)
    and returns P10, P50, P90 plus statistics and a histogram of the STOIIP distribution.
    """
)

# Helper functions

def sample_param(is_range, low, high, size, distribution, mode=None):
    if not is_range:
        return np.full(size, low)
    if distribution == "Uniform":
        return np.random.uniform(low, high, size)
    elif distribution == "Triangular":
        if mode is None:
            # fallback to uniform if mode missing
            return np.random.uniform(low, high, size)
        return np.random.triangular(low, mode, high, size)
    else:
        return np.random.uniform(low, high, size)


def compute_stoiip(A, h, phi, Sw, Boi):
    # A in acres, h in feet, phi fraction (0-1), Sw fraction (0-1), Boi in bbl/STB
    return 7758.0 * A * h * phi * (1.0 - Sw) / Boi


# Sidebar: sampling settings
st.sidebar.header("Monte Carlo & Sampling settings")
sample_size = st.sidebar.slider("Number of Monte Carlo samples", min_value=100, max_value=200000, value=20000, step=100)
distribution_choice = st.sidebar.selectbox("Sampling distribution (for ranged inputs)", ["Uniform", "Triangular"]) 
seed = st.sidebar.number_input("Random seed (0 = don't set)", min_value=0, value=0)
if seed != 0:
    np.random.seed(int(seed))

st.sidebar.markdown("**Note:** If all inputs are fixed values, sampling will return a single deterministic STOIIP value.")

# Input panels
col1, col2 = st.columns(2)

with col1:
    st.subheader("Reservoir & Petrophysical Inputs")

    st.markdown("---")

    def input_with_range(label, default_val, min_val, max_val, step=None):
        st.write(f"**{label}**")
        is_range = st.checkbox(f"Use range for {label}", value=False, key=f"rng_{label}")
        if not is_range:
            v = st.number_input(f"{label}", value=float(default_val), min_value=min_val, max_value=max_val, step=step, key=f"val_{label}")
            return False, v, v
        else:
            c1, c2 = st.columns(2)
            with c1:
                low = st.number_input(f"{label} — Min", value=float(default_val*0.8), min_value=min_val, max_value=max_val, step=step, key=f"low_{label}")
            with c2:
                high = st.number_input(f"{label} — Max", value=float(default_val*1.2), min_value=min_val, max_value=max_val, step=step, key=f"high_{label}")
            # If Triangular distribution selected, optionally ask for mode
            mode = None
            if distribution_choice == "Triangular":
                mode = st.number_input(f"{label} — Mode (for triangular)", value=float((low + high) / 2.0), min_value=min_val, max_value=max_val, step=step, key=f"mode_{label}")
            return True, low, high if high >= low else low

    # Area A (acres)
    A_is_range, A_low, A_high = input_with_range("Area A (acres)", default_val=100.0, min_val=0.0, max_val=1e7, step=0.1)

    # Net pay h (ft)
    h_is_range, h_low, h_high = input_with_range("Net pay h (ft)", default_val=25.0, min_val=0.0, max_val=1e5, step=0.1)

    # Porosity phi (fraction)
    phi_is_range, phi_low, phi_high = input_with_range("Porosity φ (fraction 0-1)", default_val=0.18, min_val=0.0, max_val=1.0, step=0.001)

with col2:
    st.subheader("Fluid & Formation Inputs")

    st.markdown("---")

    # Water saturation Sw (fraction)
    Sw_is_range, Sw_low, Sw_high = input_with_range("Water saturation S_w (fraction 0-1)", default_val=0.25, min_val=0.0, max_val=1.0, step=0.001)

    # Formation volume factor Boi (bbl/STB)
    Boi_is_range, Boi_low, Boi_high = input_with_range("Formation vol. factor B_oi (bbl/STB)", default_val=1.2, min_val=0.01, max_val=10.0, step=0.01)

    st.markdown("---")
    st.write("**Percentiles to report**")
    p10 = st.checkbox("Report P10", value=True)
    p50 = st.checkbox("Report P50 (median)", value=True)
    p90 = st.checkbox("Report P90", value=True)

# Build samples
st.write("# Results")

# Prepare parameter arrays
A_samples = sample_param(A_is_range, A_low, A_high, sample_size, distribution_choice, mode=None if distribution_choice!= "Triangular" else (A_low + A_high)/2)
h_samples = sample_param(h_is_range, h_low, h_high, sample_size, distribution_choice, mode=None if distribution_choice!= "Triangular" else (h_low + h_high)/2)
phi_samples = sample_param(phi_is_range, phi_low, phi_high, sample_size, distribution_choice, mode=None if distribution_choice!= "Triangular" else (phi_low + phi_high)/2)
Sw_samples = sample_param(Sw_is_range, Sw_low, Sw_high, sample_size, distribution_choice, mode=None if distribution_choice!= "Triangular" else (Sw_low + Sw_high)/2)
Boi_samples = sample_param(Boi_is_range, Boi_low, Boi_high, sample_size, distribution_choice, mode=None if distribution_choice!= "Triangular" else (Boi_low + Boi_high)/2)

# compute STOIIP samples
stoiip_samples = compute_stoiip(A_samples, h_samples, phi_samples, Sw_samples, Boi_samples)

# If all inputs were fixed, show single value
all_fixed = not (A_is_range or h_is_range or phi_is_range or Sw_is_range or Boi_is_range)

if all_fixed:
    deterministic_value = stoiip_samples[0]
    st.metric("Deterministic STOIIP (STB)", f"{deterministic_value:,.0f}")
    st.write("All inputs provided as fixed values — STOIIP computed deterministically.")
else:
    # Show percentiles and distribution
    percentiles = {}
    if p10:
        percentiles['P10'] = np.percentile(stoiip_samples, 10)
    if p50:
        percentiles['P50'] = np.percentile(stoiip_samples, 50)
    if p90:
        percentiles['P90'] = np.percentile(stoiip_samples, 90)

    stats_col1, stats_col2 = st.columns(2)
    with stats_col1:
        st.subheader("Statistics")
        st.write(pd.DataFrame({
            'Mean': [np.mean(stoiip_samples)],
            'Std Dev': [np.std(stoiip_samples)],
            'Min': [np.min(stoiip_samples)],
            'Max': [np.max(stoiip_samples)]
        }).T)

    with stats_col2:
        st.subheader("Requested Percentiles")
        if percentiles:
            for k, v in percentiles.items():
                st.metric(k, f"{v:,.0f} STB")
        else:
            st.write("No percentiles selected to display.")

    # Histogram
    st.subheader("STOIIP Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(stoiip_samples, bins=60)
    ax.set_xlabel('STOIIP (STB)')
    ax.set_ylabel('Frequency')
    ax.set_title('Monte Carlo STOIIP Distribution')
    st.pyplot(fig)

    # Show sample summary table (P10, P50, P90 plus mean/stdev)
    summary = {
        'Mean': np.mean(stoiip_samples),
        'StdDev': np.std(stoiip_samples),
        'Min': np.min(stoiip_samples),
        'Max': np.max(stoiip_samples),
        'P10': np.percentile(stoiip_samples, 10),
        'P50': np.percentile(stoiip_samples, 50),
        'P90': np.percentile(stoiip_samples, 90)
    }
    st.subheader("Summary table")
    st.table(pd.DataFrame.from_dict(summary, orient='index', columns=['Value']).style.format("{:.0f}"))

    # Provide parameter sample diagnostics
    st.subheader("Parameter sample means (for diagnostics)")
    param_means = pd.DataFrame({
        'Parameter': ['A(acres)', 'h(ft)', 'phi', 'Sw', 'Boi(bbl/STB)'],
        'Mean sample value': [np.mean(A_samples), np.mean(h_samples), np.mean(phi_samples), np.mean(Sw_samples), np.mean(Boi_samples)]
    })
    st.dataframe(param_means)

    # Allow download of raw Monte Carlo results as CSV
    st.subheader("Download results")
    df_out = pd.DataFrame({
        'A_acres': A_samples,
        'h_ft': h_samples,
        'phi': phi_samples,
        'Sw': Sw_samples,
        'Boi': Boi_samples,
        'STOIIP_STB': stoiip_samples
    })

    csv = df_out.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV of samples", data=csv, file_name='stoiip_samples.csv', mime='text/csv')

st.markdown("---")
st.markdown(
    """
    **Notes & assumptions:**
    - The 7758 constant assumes area in acres and thickness in feet to give stock tank barrels.
    - Porosity and water saturation are fractions (0–1).
    - B_oi is the oil formation volume factor (bbl/STB).
    - By default ranged inputs use the sampling distribution selected in the sidebar.
    - Triangular distribution requires a mode input (the most likely value) — it defaults to the midpoint if not provided.
    """
)

st.caption("Made with ❤️ — provide feedback or ask for additional features like log-normal sampling, custom distributions, sensitivity tornado plots, or unit conversions.")
