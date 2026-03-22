import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# 1. Page Configuration & Custom CSS
st.set_page_config(page_title="Cricket Win Predictor Pro", layout="wide")

# Team Color Mapping
team_colors = {
    "India": "#004184",
    "Australia": "#FFD700",
    "England": "#CE1124",
    "South Africa": "#007A4D",
    "Pakistan": "#01411C",
    "New Zealand": "#000000",
    "West Indies": "#7B0031",
    "Sri Lanka": "#000080",
    "Afghanistan": "#0055A4",
    "Bangladesh": "#006A4E"
}

# 2. Initialize Match History in Session State
if "history" not in st.session_state:
    st.session_state.history = []

# 3. Load Model (V2 with Momentum)
@st.cache_resource
def load_model():
    model = joblib.load("cricket_win_model_v2.pkl")
    features = joblib.load("model_features_v2.pkl")
    return model, features

try:
    model, model_features = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'train_model.py' was run successfully.")
    st.stop()

# 4. Sidebar - Stats & Dynamic Branding
with st.sidebar:
    st.title(" Match Center")
    st.image("https://img.icons8.com/color/96/cricket.png")
    
    st.markdown("### Selection")
    batting_team = st.selectbox("Batting Team", list(team_colors.keys()))
    bowling_team = st.selectbox("Bowling Team", [t for t in team_colors.keys() if t != batting_team])
    
    accent_color = team_colors.get(batting_team, "#ff4b4b")
    st.markdown(f"""
        <div style="height:8px; background-color:{accent_color}; border-radius:10px; margin-bottom:20px;"></div>
        """, unsafe_allow_html=True)
    
    st.info("Using XGBoost V2 with Momentum Tracking")
    
    if st.button("Clear History", width="stretch"):
        st.session_state.history = []
        st.rerun()

# 5. Main UI Layout
st.title(" Real-Time Cricket Win Predictor")

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader(" Match Situation")
    
    with st.expander("Score Details", expanded=True):
        target = st.number_input("Target Score", min_value=1, value=180)
        current_score = st.number_input("Current Score", min_value=0, value=100)
        wickets_down = st.slider("Wickets Fallen", 0, 9, 3)
        overs_completed = st.number_input("Overs Completed", 0.0, 20.0, 0.1, 12.0)

    with st.expander("Recent Momentum (Last 3 Overs)", expanded=True):
        last_18_runs = st.slider("Runs in last 18 balls", 0, 60, 15)
        last_18_wickets = st.slider("Wickets in last 18 balls", 0, 5, 0)

    # Player-Specific Logic UI
    with st.expander(" Player & Pitch Impact", expanded=True):
        star_batsman = st.checkbox("Is a 'Finisher' or 'Star Batsman' at the crease?")
        pitch_type = st.select_slider("Pitch Condition", options=["Bowling Friendly", "Neutral", "Batting Paradise"], value="Neutral")

    # Prediction Trigger
    predict_clicked = st.button("Run AI Prediction", width="stretch")

# Math & Logic helper
balls_delivered = (int(overs_completed) * 6) + (int((overs_completed % 1) * 10))
balls_left = 120 - balls_delivered
runs_to_win = target - current_score
crr = current_score / (balls_delivered / 6) if balls_delivered > 0 else 0
rrr = runs_to_win / (balls_left / 6) if balls_left > 0 else 0
wickets_left = 10 - wickets_down

def apply_custom_logic(base_prob, is_star, pitch):
    adjusted_prob = base_prob
    if is_star:
        if adjusted_prob < 90:
            adjusted_prob += 7.5 
        else:
            adjusted_prob += (100 - adjusted_prob) * 0.5
    
    if pitch == "Batting Paradise":
        adjusted_prob += 3
    elif pitch == "Bowling Friendly":
        adjusted_prob -= 5
    return max(1, min(99, adjusted_prob))

if predict_clicked:
    # Prepare Input
    input_data = pd.DataFrame([[
        overs_completed, current_score, wickets_left, crr, rrr, 
        runs_to_win, balls_left, last_18_runs, last_18_wickets
    ]], columns=model_features)

    # Base Probability
    raw_win_prob = model.predict_proba(input_data)[0][1] * 100
    
    # Final Adjusted Probability
    win_prob = apply_custom_logic(raw_win_prob, star_batsman, pitch_type)

    # Save to History
    st.session_state.history.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Batting Team": batting_team,
        "Bowling Team": bowling_team,
        "Score": f"{current_score}/{wickets_down}",
        "Overs": overs_completed,
        "Runs Needed": runs_to_win,
        "RRR": f"{rrr:.2f}",
        "Win %": f"{win_prob:.1f}%",
        "Star_Player": "Yes" if star_batsman else "No"
    })

    with col2:
        st.subheader(" Probability Analysis")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Win Prob", f"{win_prob:.1f}%", delta=f"{win_prob-50:.1f}%", delta_color="normal")
        m2.metric("Req. Rate", f"{rrr:.2f}")
        m3.metric("Runs Needed", f"{runs_to_win}")

        # Plotly Scenario Graph
        st.markdown("#### Scenario Impact Analysis")
        
        # Scenario 1: Next Ball Wicket
        input_wicket = input_data.copy()
        input_wicket['wickets_left'] = max(0, wickets_left - 1)
        input_wicket['last_18_wickets'] += 1
        raw_prob_wicket = model.predict_proba(input_wicket)[0][1] * 100
        prob_if_wicket = apply_custom_logic(raw_prob_wicket, False, pitch_type)

        # Scenario 2: Good Over (12 runs)
        input_runs = input_data.copy()
        input_runs['current_score'] += 12
        input_runs['runs_to_win'] -= 12
        raw_prob_runs = model.predict_proba(input_runs)[0][1] * 100
        prob_if_runs = apply_custom_logic(raw_prob_runs, star_batsman, pitch_type)

        fig = go.Figure(go.Bar(
            x=['Current State', 'Next Ball Wicket', '12 Runs in Next Over'],
            y=[win_prob, prob_if_wicket, prob_if_runs],
            marker_color=[accent_color, '#ef4444', '#10b981'],
            text=[f"{win_prob:.1f}%", f"{prob_if_wicket:.1f}%", f"{prob_if_runs:.1f}%"],
            textposition='auto',
        ))
        fig.update_layout(height=350, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# 7. Match History Table & Download Section
if st.session_state.history:
    st.divider()
    h_col1, h_col2 = st.columns([3, 1])
    
    with h_col1:
        st.subheader(" Prediction History")
    
    with h_col2:
        # Create CSV from history
        history_df = pd.DataFrame(st.session_state.history)
        csv_data = history_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label=" Download Report (CSV)",
            data=csv_data,
            file_name=f"match_prediction_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            width="stretch"
        )
    
    # Display table (newest at top)
    st.table(history_df.iloc[::-1])