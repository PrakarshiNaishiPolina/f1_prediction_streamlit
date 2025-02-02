import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import time

# Page Config
st.set_page_config(page_title="F1 Championship Insights", page_icon="🏎️")

st.markdown("""
    <style>
        body {
            font-family: 'Helvetica', sans-serif;
            background-color: #000000;
            color: #ffffff;
        }
        h1, h2, h3 {
            color: #ff0000;
            text-align: center;
            font-size: 3rem;
            text-transform: uppercase;
        }
        .st-bq {
            color: #ff0000;
        }
        .st-cb {
            background-color: #1b1f3a;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1>Forumla 1 Race Simulator</h1>',unsafe_allow_html=True)

# Progress bar
progress_bar=st.progress(0)
message=st.empty()
message.write("Starting engines... 🏁")
time.sleep(1) # pauses for 1 second and then executes
progress_bar.progress(50)
message.write("Accelerating to full speed... 🚀")
time.sleep(1)
progress_bar.progress(100)
message.write("Race in progress... 🏎️💨")

# Tabs
tab1, tab2, tab3 = st.tabs(["🏁 Race Experience", "⚙️ Pit Stop Strategy", "📊 Race Stats"])
with tab1:
    st.header("🏁 Choose Your Driver and Team")
    team = st.selectbox("Choose Your F1 Team", ["Mercedes", "Red Bull", "Ferrari", "McLaren", "Aston Martin"])
    driver = st.selectbox("Choose Your Driver", ["Lewis Hamilton", "Max Verstappen", "Charles Leclerc", "Lando Norris", "Sebastian Vettel"])
    st.write(f"Driver: {driver} | Team: {team}")
    if st.button("Start Race"):
        st.success(f"Race started with {driver} from {team}!")
# Pit Stop Strategy Tab
with tab2:
    st.header("⚙️ Pit Stop Strategy")
    tire_type = st.radio("Choose Tire Type", ["Soft", "Medium", "Hard"])
    fuel_level = st.slider("Fuel Level", 0, 100, 50)
    st.write(f"Tire Type: {tire_type} | Fuel Level: {fuel_level}%")
# Race Stats Tab
with tab3:
    st.header("📊 Race Stats and Predictions")
    # race data 
    data = {
        "Driver": ["Lewis Hamilton", "Max Verstappen", "Charles Leclerc", "Lando Norris", "Sebastian Vettel"],
        "Team": ["Mercedes", "Red Bull", "Ferrari", "McLaren", "Aston Martin"],
        "Lap Time (s)": [90, 92, 93, 95, 97]
    }
    df = pd.DataFrame(data)
    st.table(df)

    # Simulate prediction using a model 
    st.subheader("Predict Race Outcome")

    # Feature inputs from whatever selected
    race_conditions = {"Team": [team], "Tire Type": [tire_type], "Fuel Level": [fuel_level]}
    race_df = pd.DataFrame(race_conditions)

     # Label encoding for categorical variables
    label_encoder = LabelEncoder()
    race_df['Team'] = label_encoder.fit_transform(race_df['Team'])
    race_df['Tire Type'] = label_encoder.fit_transform(race_df['Tire Type'])

    # Model
    model=RandomForestClassifier()
    model.fit([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]], [1, 2, 3, 4, 5])  # First argument is the input features the other one is target or the labels
    prediction = model.predict(race_df)  # Predicted finish position

    st.write(f"Predicted finish position: {prediction[0]}")
# Download Button for Race Data
race_data = df.to_csv(index=False)
st.download_button(
    label="Download Race Data",
    data=race_data,
    file_name="f1_race_data.csv",
    mime="text/csv"
)
# Plotting the lap time comparisons
st.subheader("Lap Time Comparison")
x = np.array([1, 2, 3, 4, 5])  #  lap numbers
y = np.array([90, 97, 60, 95, 87])  # Corresponding lap times
fig, ax = plt.subplots()
ax.plot(x, y, color='#ff0000', marker='o')
ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time (seconds)")
ax.set_title("Lap Time Comparison")
st.pyplot(fig)

# Sidebar for race-related input (weather, etc.)
st.sidebar.header("🌦️ Race Conditions")
weather = st.sidebar.selectbox("Select Weather", ["Sunny", "Rainy", "Cloudy"])
track_conditions = st.sidebar.selectbox("Select Track Condition", ["Dry", "Wet", "Mixed"])
st.sidebar.write(f"Weather: {weather} | Track: {track_conditions}")

# Time and Date Inputs for upcoming races
st.time_input("⏰ Race Time:")
st.date_input("📅 Next Race Date:")

# Image
uploaded_image = st.file_uploader("Upload your driver's image", type=["jpg", "png"])

if uploaded_image:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Driver Image", use_container_width=True)
else:
    # Show a default image if no upload
    st.image("sainz.jpg",
             caption="Carlos Sainz Jr.", use_container_width=True)

st.divider()

# F1 Tips
st.markdown("""
    ## **Formula 1 Tips**
    - Believe you are the best otherwise its better to sit at home 🏎️
    - Don't forget to defend your position on the track! 🏁
    - Strategy is key to winning the race! 🚗💨
""")
