import streamlit as st
import pandas as pd
from collections import Counter

st.set_page_config(
    page_title="CandleCast | EUR/USD",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background: #080e18;
    color: #eef4ff;
}
[data-testid="stSidebar"] {
    background: #0b1320;
    border-right: 1px solid #223047;
}
h1, h2, h3 {
    color: #f4f7ff !important;
}
p, label, span {
    color: #b7c4d9;
}
.panel {
    background: #0d1624;
    border: 1px solid #243249;
    border-radius: 15px;
    padding: 22px;
    margin-bottom: 15px;
}
.metric {
    background: #111d2d;
    border: 1px solid #293950;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
}
.bull {
    color: #00d897;
    font-size: 25px;
    font-weight: 700;
}
.bear {
    color: #ff5265;
    font-size: 25px;
    font-weight: 700;
}
.small {
    color: #91a2bb;
    font-size: 13px;
}
div.stButton > button {
    background: #08b981;
    color: white;
    border: 0;
    border-radius: 9px;
    min-height: 46px;
    font-weight: 700;
    width: 100%;
}
div.stButton > button:hover {
    background: #00a875;
    color: white;
}
hr {
    border-color: #243249;
}
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.markdown("## 📈 CandleCast")
    st.caption("EUR/USD • Candle intelligence")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "How it works", "About"],
        label_visibility="visible",
    )

    st.divider()
    st.markdown("**MARKET SETTINGS**")
    st.markdown("Pair: `EUR/USD`")
    st.markdown("Interval: `15 minutes`")
    st.markdown("Input: `10 candles`")
    st.markdown("Forecast: `3 candles`")


# Header
st.markdown("# Candle Prediction")
st.caption(
    "Enter the last 10 completed candle colours to estimate "
    "the next 3 candles."
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        '<div class="metric"><div class="small">TRADING PAIR</div>'
        '<h3>EUR/USD</h3></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        '<div class="metric"><div class="small">TIMEFRAME</div>'
        '<h3>15 Minutes</h3></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        '<div class="metric"><div class="small">FORECAST HORIZON</div>'
        '<h3>Next 3 candles</h3></div>',
        unsafe_allow_html=True,
    )

st.write("")

if page == "Dashboard":

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Enter the last 10 candles")
    st.caption(
        "Candle 1 is the oldest. Candle 10 must be the most recent "
        "completed 15-minute candle."
    )

    st.markdown(
        "🟢 Green = Bullish / close above open"
        "　　 🔴 Red = Bearish / close below open"
    )

    default_colors = [
        "Green", "Red", "Green", "Green", "Red",
        "Red", "Green", "Red", "Green", "Red"
    ]

    with st.form("candle_input_form"):
        candles = []

        for start in range(0, 10, 5):
            cols = st.columns(5)

            for j in range(5):
                idx = start + j
                with cols[j]:
                    color = st.selectbox(
                        f"Candle {idx + 1}",
                        ["Green", "Red"],
                        index=(
                            0 if default_colors[idx] == "Green" else 1
                        ),
                        key=f"candle_{idx + 1}",
                    )
                    candles.append(color)

        submitted = st.form_submit_button(
            "✦  Predict next 3 candles",
            use_container_width=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Display input sequence
    st.markdown("### Your candle sequence")

    cols = st.columns(10)
    for i, color in enumerate(candles):
        with cols[i]:
            emoji = "🟢" if color == "Green" else "🔴"
            st.markdown(
                f'<div class="metric">'
                f'<div class="small">{i + 1}</div>'
                f'<div style="font-size:25px">{emoji}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if submitted:
        # Transparent baseline:
        # Estimate transitions using only the entered 10 candles.
        # This is NOT a trained LightGBM forecast.
        sequence = candles.copy()
        forecast = []

        for step in range(3):
            transitions = {}

            for i in range(len(sequence) - 1):
                previous = sequence[i]
                following = sequence[i + 1]
                transitions.setdefault(previous, []).append(following)

            current = sequence[-1]
            observed = transitions.get(current, [])

            if observed:
                counts = Counter(observed)
                prediction, count = counts.most_common(1)[0]
                confidence = 100 * count / len(observed)
                basis = f"{len(observed)} matching transitions"
            else:
                # Insufficient evidence: use the overall colour balance.
                counts = Counter(sequence)
                prediction, count = counts.most_common(1)[0]
                confidence = 100 * count / len(sequence)
                basis = "Overall input colour balance"

            forecast.append({
                "color": prediction,
                "confidence": confidence,
                "basis": basis,
            })
            sequence.append(prediction)

        st.markdown("---")
        st.markdown("## Next 3 candle estimates")
        st.warning(
            "Baseline estimates only. These are not generated by your "
            "trained LightGBM model and are not validated probabilities."
        )

        result_cols = st.columns(3)

        for i, (col, item) in enumerate(zip(result_cols, forecast)):
            color = item["color"]
            bullish = color == "Green"
            symbol = "🟢" if bullish else "🔴"
            label = "BULLISH" if bullish else "BEARISH"
            css = "bull" if bullish else "bear"

            with col:
                st.markdown(
                    f'<div class="panel">'
                    f'<div class="small">FORECAST {i + 1}</div>'
                    f'<div style="font-size:35px">{symbol}</div>'
                    f'<div class="{css}">{label}</div>'
                    f'<hr>'
                    f'<div class="small">Heuristic agreement</div>'
                    f'<h3>{item["confidence"]:.0f}%</h3>'
                    f'<div class="small">{item["basis"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.caption(
            "Forecasts are sequential estimates. A new estimate uses "
            "the preceding forecast as part of its input."
        )

        st.download_button(
            "Download forecast CSV",
            data=pd.DataFrame([
                {
                    "Forecast": f"Candle +{i + 1}",
                    "Estimated colour": item["color"],
                    "Heuristic agreement (%)": round(
                        item["confidence"], 2
                    ),
                    "Evidence": item["basis"],
                }
                for i, item in enumerate(forecast)
            ]).to_csv(index=False),
            file_name="candle_forecast.csv",
            mime="text/csv",
        )

elif page == "How it works":
    st.markdown("## How to use CandleCast")
    st.markdown("""
    1. Open your EUR/USD chart.
    2. Select the last 10 **completed** 15-minute candles,
       from oldest to newest.
    3. Click **Predict next 3 candles**.
    4. Review the three sequential colour estimates.

    A green candle means its close is above its open.
    A red candle means its close is below its open.
    Doji candles require a separate rule and are not included
    in this two-colour version.
    """)

elif page == "About":
    st.markdown("## About this dashboard")
    st.markdown("""
    CandleCast is a user-input interface prototype for a
    EUR/USD candle prediction project.

    The current forecast uses a simple transition-frequency
    baseline. It does not yet load the project's trained
    LightGBM model.

    To produce model-based forecasts, the app must be connected
    to the trained model and its exact feature-generation pipeline.
    """)
