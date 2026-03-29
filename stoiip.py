import html
import io
import math
from datetime import datetime
from statistics import NormalDist
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

st.set_page_config(page_title="STOIIP Monte Carlo", layout="wide")

try:
    from scipy.special import erf as scipy_erf
    from scipy.special import erfinv as scipy_erfinv
except Exception:
    scipy_erf = None
    scipy_erfinv = None

STANDARD_NORMAL = NormalDist()


# -----------------------------
# Sampling and math utilities
# -----------------------------
def sample_distribution(dist_name, p1, p2, p3, n):
    """Return random samples for the selected distribution."""
    if dist_name == "Triangular":
        left, mode, right = p1, p2, p3
        return np.random.triangular(left, mode, right, n)
    if dist_name == "Uniform":
        low, high = p1, p2
        return np.random.uniform(low, high, n)
    if dist_name == "Normal":
        mean, std = p1, p2
        return np.random.normal(mean, std, n)
    raise ValueError(f"Unsupported distribution: {dist_name}")


def make_psd(matrix: np.ndarray) -> np.ndarray:
    """Force a symmetric matrix to be positive semi-definite."""
    sym = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals[eigvals < 1e-10] = 1e-10
    corrected = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(corrected))
    corrected = corrected / np.outer(d, d)
    np.fill_diagonal(corrected, 1.0)
    return corrected


def normal_cdf(x):
    """Standard normal CDF using scipy when available, otherwise math.erf."""
    x = np.asarray(x, dtype=float)
    if scipy_erf is not None:
        return 0.5 * (1.0 + scipy_erf(x / np.sqrt(2.0)))
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


def inverse_normal_cdf(u):
    """Inverse standard normal CDF with a no-scipy fallback."""
    u = np.clip(np.asarray(u, dtype=float), 1e-10, 1.0 - 1e-10)
    if scipy_erfinv is not None:
        return np.sqrt(2.0) * scipy_erfinv(2.0 * u - 1.0)
    return np.vectorize(STANDARD_NORMAL.inv_cdf)(u)


def latin_like_correlated_uniforms(corr_matrix: np.ndarray, n: int):
    """Gaussian copula approach: correlated normal -> uniform."""
    z = np.random.multivariate_normal(
        mean=np.zeros(corr_matrix.shape[0]),
        cov=corr_matrix,
        size=n,
    )
    u = normal_cdf(z)
    u = np.clip(u, 1e-10, 1.0 - 1e-10)
    return u


def sample_from_uniforms(dist_name, p1, p2, p3, u):
    """Transform uniform [0,1] samples into the required marginal distribution."""
    if dist_name == "Uniform":
        return p1 + u * (p2 - p1)

    if dist_name == "Triangular":
        left, mode, right = p1, p2, p3
        c = (mode - left) / (right - left)
        out = np.where(
            u < c,
            left + np.sqrt(u * (right - left) * (mode - left)),
            right - np.sqrt((1.0 - u) * (right - left) * (right - mode)),
        )
        return out

    if dist_name == "Normal":
        mean, std = p1, p2
        z = inverse_normal_cdf(u)
        return mean + std * z

    raise ValueError(f"Unsupported distribution: {dist_name}")


def create_cumulative_plot(values, stats, xlabel, title):
    """Build an empirical cumulative distribution plot."""
    sorted_values = np.sort(np.asarray(values, dtype=float))
    cumulative_probability = np.arange(1, len(sorted_values) + 1) / len(sorted_values) * 100.0

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sorted_values, cumulative_probability, linewidth=2.0, color="#1f77b4")
    ax.scatter(
        [stats["P90"], stats["P50"], stats["P10"]],
        [10, 50, 90],
        color="#d62728",
        zorder=3,
        label="P90 / P50 / P10",
    )
    ax.axvline(stats["P90"], linestyle="--", linewidth=1.2, color="#ff7f0e", label=f"P90 = {stats['P90']:,.2f}")
    ax.axvline(stats["P50"], linestyle="--", linewidth=1.2, color="#2ca02c", label=f"P50 = {stats['P50']:,.2f}")
    ax.axvline(stats["P10"], linestyle="--", linewidth=1.2, color="#9467bd", label=f"P10 = {stats['P10']:,.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative Probability (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend()
    return fig


# -----------------------------
# UI helpers
# -----------------------------
def distribution_inputs(name, defaults):
    st.subheader(name)
    dist = st.selectbox(
        f"{name} distribution",
        ["Triangular", "Uniform", "Normal"],
        index=0,
        key=f"dist_{name}",
    )

    if dist == "Triangular":
        c1, c2, c3 = st.columns(3)
        with c1:
            p1 = st.number_input(f"{name} min", value=float(defaults[0]), key=f"{name}_min")
        with c2:
            p2 = st.number_input(f"{name} mode", value=float(defaults[1]), key=f"{name}_mode")
        with c3:
            p3 = st.number_input(f"{name} max", value=float(defaults[2]), key=f"{name}_max")
        valid = p1 <= p2 <= p3 and p1 != p3
        if not valid:
            st.error(f"For {name}, Triangular requires min <= mode <= max and min != max.")
        params = (p1, p2, p3)

    elif dist == "Uniform":
        c1, c2 = st.columns(2)
        with c1:
            p1 = st.number_input(f"{name} min", value=float(defaults[0]), key=f"{name}_umin")
        with c2:
            p2 = st.number_input(f"{name} max", value=float(defaults[2]), key=f"{name}_umax")
        valid = p1 < p2
        if not valid:
            st.error(f"For {name}, Uniform requires min < max.")
        params = (p1, p2, None)

    else:
        c1, c2 = st.columns(2)
        with c1:
            p1 = st.number_input(f"{name} mean", value=float(defaults[1]), key=f"{name}_mean")
        with c2:
            p2 = st.number_input(f"{name} standard deviation", value=float(defaults[3]), key=f"{name}_std")
        valid = p2 > 0
        if not valid:
            st.error(f"For {name}, Normal requires standard deviation > 0.")
        params = (p1, p2, None)

    return dist, params, valid


def render_correlation_editor(var_names):
    st.subheader("Input correlation matrix")
    st.caption(
        "Enter pairwise correlations between uncertain inputs. Values should be between -1 and +1. "
        "Diagonal remains 1.00. This is implemented using a Gaussian copula, so marginal distributions remain as defined above."
    )

    base_matrix = np.array(
        [
            [1.00, 0.30, 0.20, -0.15, 0.10, 0.15],
            [0.30, 1.00, 0.25, -0.10, 0.10, 0.10],
            [0.20, 0.25, 1.00, -0.35, -0.10, 0.20],
            [-0.15, -0.10, -0.35, 1.00, 0.10, -0.20],
            [0.10, 0.10, -0.10, 0.10, 1.00, -0.10],
            [0.15, 0.10, 0.20, -0.20, -0.10, 1.00],
        ]
    )
    matrix_size = len(var_names)
    default_matrix = np.eye(matrix_size, dtype=float)
    overlap = min(matrix_size, base_matrix.shape[0])
    default_matrix[:overlap, :overlap] = base_matrix[:overlap, :overlap]

    corr_df = pd.DataFrame(default_matrix, index=var_names, columns=var_names)

    edited = st.data_editor(
        corr_df,
        use_container_width=True,
        key="corr_editor",
    )

    mat = edited.to_numpy(dtype=float)
    mat = (mat + mat.T) / 2.0
    np.fill_diagonal(mat, 1.0)

    if np.any(mat < -1.0) or np.any(mat > 1.0):
        st.error("All correlation values must be between -1 and +1.")
        return None, False

    try:
        _ = np.linalg.cholesky(make_psd(mat))
    except Exception:
        st.error("The correlation matrix is not numerically valid. Adjust the values.")
        return None, False

    return make_psd(mat), True


# -----------------------------
# App header
# -----------------------------
st.title("STOIIP & Recoverable Reserves Monte Carlo Calculator")
st.markdown(
    """
    This app estimates:
    - **STOIIP** using volumetric Monte Carlo simulation
    - **Recoverable reserves** by applying an uncertain **Recovery Factor (RF)**

    Equations used:

    **STOIIP = 7758 x A x h x NTG x phi x (1 - Sw) / Boi**

    **Recoverable Reserves = STOIIP x RF**

    Where:
    - **A** = Area, acres
    - **h** = Gross thickness or gross pay, ft
    - **NTG** = Net-to-gross, fraction
    - **phi** = Porosity, fraction
    - **Sw** = Water saturation, fraction
    - **Boi** = Oil formation volume factor, rb/stb
    - **RF** = Recovery factor, fraction
    """
)

with st.sidebar:
    st.header("Report metadata")
    company_name = st.text_input("Company name", value="Your Company")
    project_name = st.text_input("Project / field name", value="Field A STOIIP Study")
    analyst_name = st.text_input("Analyst name", value="Mustafa")
    logo_file = st.file_uploader("Company logo (PNG/JPG)", type=["png", "jpg", "jpeg"])
    default_comments = """Assumptions:
- Volumetric uncertainty modeled through independent or correlated input distributions.
- Recovery factor treated as an uncertain scalar applied to STOIIP.
- Units are field units unless otherwise stated."""
    report_comments = st.text_area(
        "Comments / assumptions",
        value=default_comments,
        height=140,
    )

    st.header("Simulation settings")
    n_trials = st.number_input(
        "Number of trials",
        min_value=1000,
        max_value=500000,
        value=10000,
        step=1000,
        help="Larger values improve stability but take longer.",
    )
    random_seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    use_correlation = st.checkbox("Enable input correlation", value=True)
    run_button = st.button("Run simulation")

if "run_simulation" not in st.session_state:
    st.session_state["run_simulation"] = False

if run_button:
    st.session_state["run_simulation"] = True


# -----------------------------
# Inputs
# -----------------------------
col_left, col_right = st.columns(2)

with col_left:
    area_dist, area_params, area_ok = distribution_inputs("Area (acres)", [500, 700, 900, 100])
    gross_dist, gross_params, gross_ok = distribution_inputs("Gross Thickness (ft)", [30, 40, 60, 8])
    ntg_dist, ntg_params, ntg_ok = distribution_inputs("NTG (fraction)", [0.60, 0.75, 0.90, 0.08])

with col_right:
    poro_dist, poro_params, poro_ok = distribution_inputs("Porosity (fraction)", [0.12, 0.18, 0.24, 0.03])
    sw_dist, sw_params, sw_ok = distribution_inputs("Water Saturation Sw (fraction)", [0.15, 0.25, 0.35, 0.05])
    boi_dist, boi_params, boi_ok = distribution_inputs("Boi (rb/stb)", [1.10, 1.20, 1.35, 0.08])

rf_dist, rf_params, rf_ok = distribution_inputs("Recovery Factor RF (fraction)", [0.15, 0.25, 0.35, 0.05])

all_ok = all([area_ok, gross_ok, ntg_ok, poro_ok, sw_ok, boi_ok, rf_ok])

st.info(
    "Recommended units: Area in acres, thickness in ft, NTG/porosity/Sw/RF as fractions, and Boi in rb/stb."
)

var_names = ["Area", "Gross_h", "NTG", "Porosity", "Sw", "Boi", "RF"]

corr_matrix = None
corr_ok = True
if use_correlation:
    corr_matrix, corr_ok = render_correlation_editor(var_names)


# -----------------------------
# Run simulation
# -----------------------------
if st.session_state["run_simulation"]:
    if not all_ok or not corr_ok:
        st.stop()

    import base64

    np.random.seed(int(random_seed))

    if use_correlation:
        u = latin_like_correlated_uniforms(corr_matrix, int(n_trials))
        area = sample_from_uniforms(area_dist, *area_params, u[:, 0])
        gross_h = sample_from_uniforms(gross_dist, *gross_params, u[:, 1])
        ntg = sample_from_uniforms(ntg_dist, *ntg_params, u[:, 2])
        poro = sample_from_uniforms(poro_dist, *poro_params, u[:, 3])
        sw = sample_from_uniforms(sw_dist, *sw_params, u[:, 4])
        boi = sample_from_uniforms(boi_dist, *boi_params, u[:, 5])
        rf = sample_from_uniforms(rf_dist, *rf_params, u[:, 6])
    else:
        area = sample_distribution(area_dist, *area_params, int(n_trials))
        gross_h = sample_distribution(gross_dist, *gross_params, int(n_trials))
        ntg = sample_distribution(ntg_dist, *ntg_params, int(n_trials))
        poro = sample_distribution(poro_dist, *poro_params, int(n_trials))
        sw = sample_distribution(sw_dist, *sw_params, int(n_trials))
        boi = sample_distribution(boi_dist, *boi_params, int(n_trials))
        rf = sample_distribution(rf_dist, *rf_params, int(n_trials))

    # Physical clipping
    area = np.clip(area, 1e-9, None)
    gross_h = np.clip(gross_h, 1e-9, None)
    ntg = np.clip(ntg, 0.0, 1.0)
    poro = np.clip(poro, 0.0, 1.0)
    sw = np.clip(sw, 0.0, 1.0)
    boi = np.clip(boi, 1e-6, None)
    rf = np.clip(rf, 0.0, 1.0)

    stoiip = 7758.0 * area * gross_h * ntg * poro * (1.0 - sw) / boi
    stoiip_mmstb = stoiip / 1e6
    recoverable_stb = stoiip * rf
    recoverable_mmstb = recoverable_stb / 1e6

    # Summary stats
    def calc_summary(x):
        return {
            "P90": np.percentile(x, 10),
            "P50": np.percentile(x, 50),
            "P10": np.percentile(x, 90),
            "Mean": np.mean(x),
            "Min": np.min(x),
            "Max": np.max(x),
        }

    stoiip_stats = calc_summary(stoiip_mmstb)
    recoverable_stats = calc_summary(recoverable_mmstb)

    st.success("Simulation completed.")

    st.subheader("STOIIP results")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("P90 STOIIP", f"{stoiip_stats['P90']:,.2f} MMSTB")
    k2.metric("P50 STOIIP", f"{stoiip_stats['P50']:,.2f} MMSTB")
    k3.metric("P10 STOIIP", f"{stoiip_stats['P10']:,.2f} MMSTB")
    k4.metric("Mean STOIIP", f"{stoiip_stats['Mean']:,.2f} MMSTB")

    st.subheader("Recoverable reserves results")
    k5, k6, k7, k8 = st.columns(4)
    k5.metric("P90 Recoverable", f"{recoverable_stats['P90']:,.2f} MMSTB")
    k6.metric("P50 Recoverable", f"{recoverable_stats['P50']:,.2f} MMSTB")
    k7.metric("P10 Recoverable", f"{recoverable_stats['P10']:,.2f} MMSTB")
    k8.metric("Mean Recoverable", f"{recoverable_stats['Mean']:,.2f} MMSTB")

    summary_df = pd.DataFrame(
        {
            "Metric": ["P90", "P50", "P10", "Mean", "Min", "Max"],
            "STOIIP (MMSTB)": [
                stoiip_stats["P90"], stoiip_stats["P50"], stoiip_stats["P10"],
                stoiip_stats["Mean"], stoiip_stats["Min"], stoiip_stats["Max"],
            ],
            "Recoverable (MMSTB)": [
                recoverable_stats["P90"], recoverable_stats["P50"], recoverable_stats["P10"],
                recoverable_stats["Mean"], recoverable_stats["Min"], recoverable_stats["Max"],
            ],
        }
    )
    st.subheader("Summary table")
    st.dataframe(summary_df, use_container_width=True)

    # Distribution plots
    st.subheader("STOIIP distribution")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.hist(stoiip_mmstb, bins=50)
    ax1.axvline(stoiip_stats["P90"], linestyle="--", linewidth=1.5, label=f"P90 = {stoiip_stats['P90']:,.2f}")
    ax1.axvline(stoiip_stats["P50"], linestyle="--", linewidth=1.5, label=f"P50 = {stoiip_stats['P50']:,.2f}")
    ax1.axvline(stoiip_stats["P10"], linestyle="--", linewidth=1.5, label=f"P10 = {stoiip_stats['P10']:,.2f}")
    ax1.set_xlabel("STOIIP (MMSTB)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Monte Carlo STOIIP Results")
    ax1.legend()
    st.pyplot(fig1)

    st.subheader("STOIIP cumulative distribution")
    fig1_cum = create_cumulative_plot(
        stoiip_mmstb,
        stoiip_stats,
        "STOIIP (MMSTB)",
        "Cumulative STOIIP Results",
    )
    st.pyplot(fig1_cum)

    stoiip_img = io.BytesIO()
    fig1.savefig(stoiip_img, format="png", bbox_inches="tight")
    stoiip_img_b64 = base64.b64encode(stoiip_img.getvalue()).decode("utf-8")
    stoiip_cum_img = io.BytesIO()
    fig1_cum.savefig(stoiip_cum_img, format="png", bbox_inches="tight")
    stoiip_cum_img_b64 = base64.b64encode(stoiip_cum_img.getvalue()).decode("utf-8")

    st.subheader("Recoverable reserves distribution")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.hist(recoverable_mmstb, bins=50)
    ax2.axvline(recoverable_stats["P90"], linestyle="--", linewidth=1.5, label=f"P90 = {recoverable_stats['P90']:,.2f}")
    ax2.axvline(recoverable_stats["P50"], linestyle="--", linewidth=1.5, label=f"P50 = {recoverable_stats['P50']:,.2f}")
    ax2.axvline(recoverable_stats["P10"], linestyle="--", linewidth=1.5, label=f"P10 = {recoverable_stats['P10']:,.2f}")
    ax2.set_xlabel("Recoverable Reserves (MMSTB)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Monte Carlo Recoverable Reserves Results")
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("Recoverable reserves cumulative distribution")
    fig2_cum = create_cumulative_plot(
        recoverable_mmstb,
        recoverable_stats,
        "Recoverable Reserves (MMSTB)",
        "Cumulative Recoverable Reserves Results",
    )
    st.pyplot(fig2_cum)

    recoverable_img = io.BytesIO()
    fig2.savefig(recoverable_img, format="png", bbox_inches="tight")
    recoverable_img_b64 = base64.b64encode(recoverable_img.getvalue()).decode("utf-8")
    recoverable_cum_img = io.BytesIO()
    fig2_cum.savefig(recoverable_cum_img, format="png", bbox_inches="tight")
    recoverable_cum_img_b64 = base64.b64encode(recoverable_cum_img.getvalue()).decode("utf-8")

    # Sensitivity ranking using rank correlation (Spearman-like via pandas ranks)
    sensitivity_df = pd.DataFrame(
        {
            "Area": area,
            "Gross_h": gross_h,
            "NTG": ntg,
            "Porosity": poro,
            "Sw": sw,
            "Boi": boi,
            "RF": rf,
            "STOIIP_MMSTB": stoiip_mmstb,
            "Recoverable_MMSTB": recoverable_mmstb,
        }
    )

    rank_df = sensitivity_df.rank()
    stoiip_corr = rank_df.corr()["STOIIP_MMSTB"].drop(["STOIIP_MMSTB", "Recoverable_MMSTB"])
    recoverable_corr = rank_df.corr()["Recoverable_MMSTB"].drop(["STOIIP_MMSTB", "Recoverable_MMSTB"])

    tornado_df = pd.DataFrame(
        {
            "Variable": stoiip_corr.index,
            "STOIIP Rank Correlation": stoiip_corr.values,
            "Recoverable Rank Correlation": recoverable_corr.values,
        }
    )
    tornado_df["Abs Recoverable Corr"] = tornado_df["Recoverable Rank Correlation"].abs()
    tornado_df = tornado_df.sort_values("Abs Recoverable Corr", ascending=True)

    st.subheader("Sensitivity ranking")
    st.dataframe(
        tornado_df[["Variable", "STOIIP Rank Correlation", "Recoverable Rank Correlation"]],
        use_container_width=True,
    )

    st.subheader("Tornado chart for recoverable reserves")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.barh(tornado_df["Variable"], tornado_df["Recoverable Rank Correlation"])
    ax3.set_xlabel("Rank Correlation with Recoverable Reserves")
    ax3.set_ylabel("Input Variable")
    ax3.set_title("Tornado / Sensitivity Ranking")
    st.pyplot(fig3)

    tornado_img = io.BytesIO()
    fig3.savefig(tornado_img, format="png", bbox_inches="tight")
    tornado_img_b64 = base64.b64encode(tornado_img.getvalue()).decode("utf-8")

    if use_correlation:
        st.subheader("Realized sample correlation matrix")
        realized_corr = sensitivity_df[["Area", "Gross_h", "NTG", "Porosity", "Sw", "Boi", "RF"]].corr()
        st.dataframe(realized_corr, use_container_width=True)

    st.subheader("Input sample statistics")
    input_stats = pd.DataFrame(
        {
            "Variable": ["Area", "Gross_h", "NTG", "Porosity", "Sw", "Boi", "RF"],
            "Mean": [area.mean(), gross_h.mean(), ntg.mean(), poro.mean(), sw.mean(), boi.mean(), rf.mean()],
            "P10": [
                np.percentile(area, 90),
                np.percentile(gross_h, 90),
                np.percentile(ntg, 90),
                np.percentile(poro, 90),
                np.percentile(sw, 90),
                np.percentile(boi, 90),
                np.percentile(rf, 90),
            ],
            "P50": [
                np.percentile(area, 50),
                np.percentile(gross_h, 50),
                np.percentile(ntg, 50),
                np.percentile(poro, 50),
                np.percentile(sw, 50),
                np.percentile(boi, 50),
                np.percentile(rf, 50),
            ],
            "P90": [
                np.percentile(area, 10),
                np.percentile(gross_h, 10),
                np.percentile(ntg, 10),
                np.percentile(poro, 10),
                np.percentile(sw, 10),
                np.percentile(boi, 10),
                np.percentile(rf, 10),
            ],
        }
    )
    st.dataframe(input_stats, use_container_width=True)

    st.subheader("Simulation output data")
    st.dataframe(sensitivity_df.head(200), use_container_width=True)

    csv_buffer = io.StringIO()
    sensitivity_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download simulation results as CSV",
        data=csv_buffer.getvalue(),
        file_name="stoiip_recoverable_monte_carlo_results.csv",
        mime="text/csv",
    )

    st.subheader("Exportable summary report")

    input_config_df = pd.DataFrame(
        {
            "Variable": ["Area", "Gross_h", "NTG", "Porosity", "Sw", "Boi", "RF"],
            "Distribution": [area_dist, gross_dist, ntg_dist, poro_dist, sw_dist, boi_dist, rf_dist],
            "Param_1": [area_params[0], gross_params[0], ntg_params[0], poro_params[0], sw_params[0], boi_params[0], rf_params[0]],
            "Param_2": [area_params[1], gross_params[1], ntg_params[1], poro_params[1], sw_params[1], boi_params[1], rf_params[1]],
            "Param_3": [area_params[2], gross_params[2], ntg_params[2], poro_params[2], sw_params[2], boi_params[2], rf_params[2]],
        }
    )

    corr_html = ""
    if use_correlation:
        corr_html = realized_corr.round(3).to_html(border=0, classes="table table-sm")
    else:
        corr_html = "<p>Input correlation disabled.</p>"

    report_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    html_company_name = html.escape(company_name)
    html_project_name = html.escape(project_name)
    html_analyst_name = html.escape(analyst_name)
    html_report_comments = html.escape(report_comments)
    report_html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>{html_project_name} - STOIIP Monte Carlo Summary Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; color: #222; background: #ffffff; }}
            .cover {{ min-height: 100vh; padding: 70px 60px; background: linear-gradient(135deg, #f4f7fb 0%, #e8eef7 100%); display: flex; flex-direction: column; justify-content: center; }}
            .brand {{ font-size: 14px; letter-spacing: 2px; text-transform: uppercase; color: #5a6b85; margin-bottom: 18px; }}
            .cover h1 {{ font-size: 42px; margin: 0 0 10px 0; color: #1f2f46; }}
            .cover h2 {{ font-size: 24px; margin: 0 0 24px 0; color: #38506f; font-weight: normal; }}
            .cover-box {{ background: rgba(255,255,255,0.8); border: 1px solid #d8e1ee; border-radius: 14px; padding: 20px 24px; max-width: 760px; }}
            .cover-meta {{ margin-top: 10px; line-height: 1.8; }}
            .page {{ padding: 28px; }}
            h1, h2, h3 {{ margin-bottom: 8px; }}
            p {{ margin-top: 4px; margin-bottom: 10px; }}
            .meta {{ color: #555; font-size: 14px; }}
            .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 14px; page-break-inside: avoid; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 8px; margin-bottom: 16px; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: right; }}
            th:first-child, td:first-child {{ text-align: left; }}
            th {{ background: #f5f5f5; }}
            .eq {{ background: #fafafa; padding: 10px; border-left: 4px solid #ccc; font-family: monospace; }}
            .comment-box {{ white-space: pre-wrap; background: #fafafa; border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
            .chart {{ margin-top: 12px; margin-bottom: 8px; }}
            .chart img {{ width: 100%; max-width: 980px; border: 1px solid #ddd; border-radius: 8px; }}
            .page-break {{ page-break-before: always; }}
        </style>
    </head>
    <body>
        <section class="cover">
            <div class="brand">{html_company_name}</div>
            <h1>STOIIP & Recoverable Reserves</h1>
            <h2>Monte Carlo Uncertainty Report</h2>
            <div class="cover-box">
                <div class="cover-meta">
                    <div><b>Project / Field:</b> {html_project_name}</div>
                    <div><b>Analyst:</b> {html_analyst_name}</div>
                    <div><b>Generated:</b> {report_timestamp}</div>
                    <div><b>Trials:</b> {int(n_trials):,}</div>
                    <div><b>Input correlation enabled:</b> {'Yes' if use_correlation else 'No'}</div>
                </div>
            </div>
        </section>

        <div class="page page-break">
            <h1>{html_project_name} - Summary Report</h1>
            <p class="meta">Prepared by {html_analyst_name} | {html_company_name}</p>

            <div class="card">
                <h2>Model Equations</h2>
                <div class="eq">STOIIP = 7758 x A x h x NTG x phi x (1 - Sw) / Boi</div>
                <div class="eq" style="margin-top:8px;">Recoverable Reserves = STOIIP x RF</div>
            </div>

            <div class="card">
                <h2>Simulation Settings</h2>
                <p><b>Number of trials:</b> {int(n_trials):,}</p>
                <p><b>Random seed:</b> {int(random_seed)}</p>
                <p><b>Input correlation enabled:</b> {'Yes' if use_correlation else 'No'}</p>
            </div>

            <div class="card">
                <h2>Comments / Assumptions</h2>
                <div class="comment-box">{html_report_comments}</div>
            </div>

            <div class="card">
                <h2>Key Results</h2>
                {summary_df.round(3).to_html(index=False, border=0)}
            </div>

            <div class="card">
                <h2>Embedded Charts</h2>
                <div class="chart">
                    <h3>STOIIP Distribution</h3>
                    <img src="data:image/png;base64,{stoiip_img_b64}" alt="STOIIP distribution chart" />
                </div>
                <div class="chart">
                    <h3>STOIIP Cumulative Distribution</h3>
                    <img src="data:image/png;base64,{stoiip_cum_img_b64}" alt="STOIIP cumulative distribution chart" />
                </div>
                <div class="chart">
                    <h3>Recoverable Reserves Distribution</h3>
                    <img src="data:image/png;base64,{recoverable_img_b64}" alt="Recoverable reserves distribution chart" />
                </div>
                <div class="chart">
                    <h3>Recoverable Reserves Cumulative Distribution</h3>
                    <img src="data:image/png;base64,{recoverable_cum_img_b64}" alt="Recoverable reserves cumulative distribution chart" />
                </div>
                <div class="chart">
                    <h3>Tornado / Sensitivity Ranking</h3>
                    <img src="data:image/png;base64,{tornado_img_b64}" alt="Sensitivity tornado chart" />
                </div>
            </div>

            <div class="card">
                <h2>Input Configuration</h2>
                {input_config_df.fillna('').to_html(index=False, border=0)}
            </div>

            <div class="card">
                <h2>Input Sample Statistics</h2>
                {input_stats.round(4).to_html(index=False, border=0)}
            </div>

            <div class="card">
                <h2>Sensitivity Ranking</h2>
                {tornado_df[["Variable", "STOIIP Rank Correlation", "Recoverable Rank Correlation"]].round(4).to_html(index=False, border=0)}
            </div>

            <div class="card">
                <h2>Realized Sample Correlation Matrix</h2>
                {corr_html}
            </div>
        </div>
    </body>
    </html>
    """

    st.download_button(
        label="Download summary report as HTML",
        data=report_html,
        file_name="stoiip_monte_carlo_summary_report.html",
        mime="text/html",
    )

    report_text = f"""{company_name}
STOIIP & Recoverable Reserves Monte Carlo Summary Report
Project / Field: {project_name}
Analyst: {analyst_name}
Generated: {report_timestamp}

Comments / Assumptions
{report_comments}

Simulation settings
- Number of trials: {int(n_trials):,}
- Random seed: {int(random_seed)}
- Input correlation enabled: {'Yes' if use_correlation else 'No'}

Summary results
{summary_df.round(3).to_string(index=False)}

Sensitivity ranking
{tornado_df[['Variable', 'STOIIP Rank Correlation', 'Recoverable Rank Correlation']].round(4).to_string(index=False)}
"""

    st.download_button(
        label="Download summary report as TXT",
        data=report_text,
        file_name="stoiip_monte_carlo_summary_report.txt",
        mime="text/plain",
    )

    def fig_bytes(fig_obj):
        buf = io.BytesIO()
        fig_obj.savefig(buf, format="png", bbox_inches="tight", dpi=180)
        buf.seek(0)
        return buf

    def build_pdf_report():
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4,
            leftMargin=16 * mm,
            rightMargin=16 * mm,
            topMargin=18 * mm,
            bottomMargin=16 * mm,
            title=f"{project_name} - STOIIP Monte Carlo Summary Report",
            author=analyst_name,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1f2f46"),
            spaceAfter=8,
        )
        subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Heading2"],
            fontSize=12,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#38506f"),
            spaceAfter=14,
        )
        section_style = ParagraphStyle(
            "SectionHeader",
            parent=styles["Heading2"],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#24364d"),
            spaceBefore=10,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "BodyTextCustom",
            parent=styles["BodyText"],
            fontSize=9,
            leading=13,
            alignment=TA_LEFT,
            spaceAfter=5,
        )
        small_style = ParagraphStyle(
            "SmallText",
            parent=body_style,
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#555555"),
        )

        def df_to_table(df, col_widths=None, font_size=8):
            data = [list(df.columns)] + df.astype(str).values.tolist()
            tbl = Table(data, colWidths=col_widths, repeatRows=1)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef6")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f2f46")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 3),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c8d0db")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            return tbl

        def draw_header_footer(canvas, doc_obj):
            canvas.saveState()
            page_w, page_h = A4
            if doc_obj.page > 1:
                canvas.setStrokeColor(colors.HexColor("#d6dde8"))
                canvas.line(doc.leftMargin, page_h - 14 * mm, page_w - doc.rightMargin, page_h - 14 * mm)
                if logo_file is not None:
                    try:
                        logo_file.seek(0)
                        logo_img = Image(io.BytesIO(logo_file.getvalue()), width=22 * mm, height=22 * mm)
                        logo_img.wrapOn(canvas, 22 * mm, 22 * mm)
                        logo_img.drawOn(canvas, doc.leftMargin, page_h - 24 * mm)
                    except Exception:
                        pass
                canvas.setFont("Helvetica-Bold", 10)
                canvas.setFillColor(colors.HexColor("#1f2f46"))
                canvas.drawRightString(page_w - doc.rightMargin, page_h - 20 * mm, project_name)

            canvas.setStrokeColor(colors.HexColor("#d6dde8"))
            canvas.line(doc.leftMargin, 12 * mm, page_w - doc.rightMargin, 12 * mm)
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.HexColor("#555555"))
            canvas.drawString(doc.leftMargin, 7 * mm, company_name)
            canvas.drawRightString(page_w - doc.rightMargin, 7 * mm, f"Page {doc_obj.page}")
            canvas.restoreState()

        story = []

        if logo_file is not None:
            try:
                logo_file.seek(0)
                story.append(Image(io.BytesIO(logo_file.getvalue()), width=34 * mm, height=34 * mm))
                story.append(Spacer(1, 8))
            except Exception:
                pass

        story.append(Paragraph(html.escape(company_name), subtitle_style))
        story.append(Paragraph("STOIIP & Recoverable Reserves", title_style))
        story.append(Paragraph("Monte Carlo Uncertainty Report", subtitle_style))
        story.append(Spacer(1, 10))

        cover_meta = pd.DataFrame({
            "Item": ["Project / Field", "Analyst", "Generated", "Trials", "Correlation Enabled"],
            "Value": [project_name, analyst_name, report_timestamp, f"{int(n_trials):,}", "Yes" if use_correlation else "No"],
        })
        story.append(df_to_table(cover_meta, col_widths=[45 * mm, 120 * mm], font_size=9))
        story.append(Spacer(1, 10))
        story.append(Paragraph("Comments / Assumptions", section_style))
        story.append(Paragraph(html.escape(report_comments).replace("\n", "<br/>"), body_style))
        story.append(PageBreak())

        story.append(Paragraph("Executive Summary", section_style))
        story.append(Paragraph("Model equations:", body_style))
        story.append(Paragraph("STOIIP = 7758 x A x h x NTG x phi x (1 - Sw) / Boi", body_style))
        story.append(Paragraph("Recoverable Reserves = STOIIP x RF", body_style))
        story.append(Spacer(1, 5))
        story.append(df_to_table(summary_df.round(3), col_widths=[35 * mm, 60 * mm, 70 * mm], font_size=8))

        story.append(Paragraph("Embedded Charts", section_style))
        story.append(Paragraph("STOIIP distribution", small_style))
        story.append(Image(fig_bytes(fig1), width=170 * mm, height=88 * mm))
        story.append(Spacer(1, 6))
        story.append(Paragraph("STOIIP cumulative distribution", small_style))
        story.append(Image(fig_bytes(fig1_cum), width=170 * mm, height=88 * mm))
        story.append(Spacer(1, 6))
        story.append(Paragraph("Recoverable reserves distribution", small_style))
        story.append(Image(fig_bytes(fig2), width=170 * mm, height=88 * mm))
        story.append(Spacer(1, 6))
        story.append(Paragraph("Recoverable reserves cumulative distribution", small_style))
        story.append(Image(fig_bytes(fig2_cum), width=170 * mm, height=88 * mm))
        story.append(Spacer(1, 6))
        story.append(Paragraph("Tornado / sensitivity ranking", small_style))
        story.append(Image(fig_bytes(fig3), width=160 * mm, height=92 * mm))
        story.append(PageBreak())

        story.append(Paragraph("Input Configuration", section_style))
        story.append(df_to_table(input_config_df.fillna("").round(4), col_widths=[28*mm, 34*mm, 36*mm, 36*mm, 36*mm], font_size=7))
        story.append(Spacer(1, 6))
        story.append(Paragraph("Input Sample Statistics", section_style))
        story.append(df_to_table(input_stats.round(4), col_widths=[34*mm, 24*mm, 24*mm, 24*mm, 24*mm], font_size=7))
        story.append(Spacer(1, 6))
        story.append(Paragraph("Sensitivity Ranking", section_style))
        sens_pdf = tornado_df[["Variable", "STOIIP Rank Correlation", "Recoverable Rank Correlation"]].round(4)
        story.append(df_to_table(sens_pdf, col_widths=[45*mm, 55*mm, 60*mm], font_size=8))
        story.append(Spacer(1, 6))
        story.append(Paragraph("Realized Sample Correlation Matrix", section_style))
        if use_correlation:
            corr_pdf = realized_corr.round(3).reset_index().rename(columns={"index": "Variable"})
            story.append(df_to_table(corr_pdf, font_size=7))
        else:
            story.append(Paragraph("Input correlation disabled.", body_style))

        doc.build(story, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)
        pdf_buffer.seek(0)
        return pdf_buffer

    pdf_report = build_pdf_report()
    st.download_button(
        label="Download summary report as PDF",
        data=pdf_report,
        file_name="stoiip_monte_carlo_summary_report.pdf",
        mime="application/pdf",
    )

else:
    st.warning("Set your input ranges, then click 'Run simulation'.")

st.markdown("---")
st.caption("Prepared as a simple Streamlit app for STOIIP and recoverable reserves uncertainty analysis.")
