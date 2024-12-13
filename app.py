import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

# Set page configuration
st.set_page_config(
    page_title="Cardiovascular Disease Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model, scaler, and default values
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
default_values = joblib.load("default_values.pkl")

# Initialize Session State
if "user_input" not in st.session_state:
    st.session_state["user_input"] = {}

# Custom CSS styles
st.markdown(
    """
    <style>
    .st-emotion-cache-h4xjwg {
        padding: 0;
        height: 0;
    }
    .st-emotion-cache-kgpedg {
        padding: 0;
    }
    .st-emotion-cache-1gwvy71 h2 {
        padding: 15px 10px 45px;
        font-size: 25px;
        font-weight: bold;
        line-height: 30px;
        text-align: center;
    }
    .st-emotion-cache-ysk9xe p {
        margin: 6px 15px;
        font-size: 18px;
        line-height: 25px;
    }
    .stSidebar .stNumberInput, .stSidebar .stSelectbox {
        display: flex;
        align-items: center;
    }
    
    .st-cd {
        /* overflow: hidden; */
        align-items: center;
    }
    .st-ce {
        width: 208px;
    }
    .stSidebar .stNumberInput > label, .stSidebar .stSelectbox > label {
        flex: 1;
        margin-right: 10px;
    }
    .stSidebar .stNumberInput > div, .stSidebar .stSelectbox > div {
        flex: 1;
    }
    .stSidebar button {
        display: block;
        margin: 0 auto;
    }    

    div [data-testid="stTab"] p {
        font-size: 22px;
        font-weight: bold;
    }
    .st-emotion-cache-1jicfl2 {
        width: 100%;
        height: 100%;
        padding: 30px 100px;
    }
    h1 {
        padding: 20px 0 10px;
        font-size: 38px;
    }
    h3 {
        padding: 25px 10px 20px;
        margin: 0;
        font-size: 28px;
        font-weight: bold;
    }
    h4 {
        padding: 10px 10px 3px;
        font-size: 25px;
    }
    h5 {
        padding: 10px 10px;
        font-size: 20px;
        line-height: 30px;
    }
    .st-emotion-cache-1y5f4eg p {
        margin: 10px 10px;
        font-size: 18px;
        line-height: 10px;
    }

    .st-d5 {
        margin: 20px 10px;
    }
    .st-d5 p {
        font-size: 20px;
        font-weight: bold;
    }

    .st-emotion-cache-1xf0csu {
        padding: 15px;
        padding-right: 60px;
    }
    .st-emotion-cache-1sdychz {
        padding-left: 15px;
    }
    .st-emotion-cache-wz4e1q {
        padding-left: 15px;
    }
    .st-emotion-cache-1y5f4eg.e1nzilvr5 p {
        margin: 10px;
        font-size: 18px;
        line-height: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

tabs = st.tabs(["Prediction", "Data Exploration"])

with tabs[0]:
    st.title("❤️ Cardiovascular Disease Risk Assessment")

    st.markdown(
        "<h4>Welcome to our Cardiovascular Disease Risk Self-Assessment Tool  :)</h4>",
        unsafe_allow_html=True
    )

    st.markdown(
        """<h5 style="color: grey;">The default values represent the most common user information in our database. If you're unsure about some of your body data, you can use the default values we provide for a rough prediction. However, to ensure accurate prediction results, we recommend entering your own information.</h5>""",
        unsafe_allow_html=True
    )

    # Place input components in the sidebar
    with st.sidebar:
        st.header("Enter Your Health Information")

        user_input = st.session_state["user_input"]

        # Define default values for inputs
        age_default = int(default_values["age"] if "age" not in user_input else user_input["age"])
        gender_default = "Male" if default_values["gender"] == 1 else "Female"
        height_default = int(default_values.get("height", 170))
        weight_default = int(default_values.get("weight", 70))
        ap_hi_default = int(default_values["ap_hi"])
        cholesterol_default = "Normal" if default_values["cholesterol_1"] == 1 else "Above Normal" if default_values["cholesterol_2"] == 1 else "Well Above Normal"
        gluc_default = "Normal" if default_values["gluc_1"] == 1 else "Above Normal" if default_values["gluc_2"] == 1 else "Well Above Normal"
        smoke_default = "Yes" if default_values["smoke"] == 1 else "No"
        alco_default = "Yes" if default_values["alco"] == 1 else "No"
        active_default = "Yes" if default_values["active"] == 1 else "No"

        # Age
        age = st.number_input("**Age**", min_value=0, max_value=120, value=age_default, key="age_input")
        user_input["age"] = age

        # Gender
        gender = st.selectbox("**Gender**", options=["Male", "Female"], index=0 if default_values["gender"] == 1 else 1, key="gender_input")
        user_input["gender"] = 1 if gender == "Male" else 0

        # Height
        height = st.number_input("**Height (cm)**", min_value=120, max_value=220, value=height_default, key="height_input")
        user_input["height"] = height

        # Weight
        weight = st.number_input("**Weight (kg)**", min_value=30, max_value=180, value=weight_default, key="weight_input")
        user_input["weight"] = weight

        BMI = weight / ((height / 100) ** 2)
        user_input["BMI"] = BMI

        # Systolic Blood Pressure
        ap_hi = st.number_input("**Systolic Blood Pressure (mmHg)**", min_value=80, max_value=240, value=ap_hi_default, key="ap_hi_input")
        user_input["ap_hi"] = ap_hi

        # Cholesterol Level
        cholesterol_options = ["Normal", "Above Normal", "Well Above Normal"]
        cholesterol_index = cholesterol_options.index(cholesterol_default)
        cholesterol = st.selectbox("**Cholesterol Level**", options=cholesterol_options, index=cholesterol_index, key="cholesterol_input")
        cholesterol_mapping = {"Normal": [1, 0, 0], "Above Normal": [0, 1, 0], "Well Above Normal": [0, 0, 1]}
        user_input["cholesterol_1"], user_input["cholesterol_2"], user_input["cholesterol_3"] = cholesterol_mapping[cholesterol]

        # Glucose Level
        gluc_options = ["Normal", "Above Normal", "Well Above Normal"]
        gluc_index = gluc_options.index(gluc_default)
        gluc = st.selectbox("**Glucose Level**", options=gluc_options, index=gluc_index, key="gluc_input")
        gluc_mapping = {"Normal": [1, 0, 0], "Above Normal": [0, 1, 0], "Well Above Normal": [0, 0, 1]}
        user_input["gluc_1"], user_input["gluc_2"], user_input["gluc_3"] = gluc_mapping[gluc]

        # Smoking
        smoke_options = ["No", "Yes"]
        smoke_index = smoke_options.index(smoke_default)
        smoke = st.selectbox("**Smoking**", options=smoke_options, index=smoke_index, key="smoke_input")
        user_input["smoke"] = 1 if smoke == "Yes" else 0

        # Alcohol
        alco_options = ["No", "Yes"]
        alco_index = alco_options.index(alco_default)
        alco = st.selectbox("**Alcohol**", options=alco_options, index=alco_index, key="alco_input")
        user_input["alco"] = 1 if alco == "Yes" else 0

        # Activity
        active_options = ["No", "Yes"]
        active_index = active_options.index(active_default)
        active = st.selectbox("**Activity**", options=active_options, index=active_index, key="active_input")
        user_input["active"] = 1 if active == "Yes" else 0

        # Add button
        st.markdown("<div style=\"text-align: center;\">", unsafe_allow_html=True)
        assess = st.button("**Assess Risk**")
        st.markdown("</div>", unsafe_allow_html=True)

    # Execute prediction when the button is clicked
    if assess:
        # Build input DataFrame
        input_df = pd.DataFrame(user_input, index=[0])

        # Feature order (keep unchanged)
        feature_order = ["age", "gender", "ap_hi", "smoke", "alco", "active", "BMI",
                        "cholesterol_1", "cholesterol_2", "cholesterol_3",
                        "gluc_1", "gluc_2", "gluc_3"]

        # Arrange data according to feature order
        input_df = input_df[feature_order]

        # Standardize numerical features
        num_features = ["age", "BMI", "ap_hi"]
        input_df[num_features] = scaler.transform(input_df[num_features])

        # Prediction
        prediction = model.predict(input_df)[0]
        risk_probability = model.predict_proba(input_df)[0][1]

        # Determine risk level
        if risk_probability < 0.33:
            risk = "Low Risk"
        elif risk_probability < 0.66:
            risk = "Medium Risk"
        else:
            risk = "High Risk"


        st.write(f"Your risk level is: ***{risk}***")
        st.write(f"Probability of cardiovascular disease: ***{risk_probability:.2%}***")
        height_m = user_input["height"] / 100
        st.write(f"Your BMI = weight (kg) / (height (m))² = {user_input['weight']} / ({height_m:.2f})² = ***{user_input['BMI']:.2f}***")

        # Assess risk for each indicator
        def assess_risk(value, feature_name):
            thresholds = {
                "ap_hi": {"medium": 130, "high": 140},
                "BMI": {"medium": 24, "high": 28},
            }
            if feature_name in thresholds:
                if value >= thresholds[feature_name]["high"]:
                    return "High"
                elif value >= thresholds[feature_name]["medium"]:
                    return "Medium"
                else:
                    return "Normal"
            else:
                return "Normal"

        # Inverse transform standardized values
        def inverse_transform(value, feature_name):
            idx = num_features.index(feature_name)
            return value * scaler.scale_[idx] + scaler.mean_[idx]

        # Map risk levels
        def map_risk_level(level):
            mapping = {
                "Normal": "Normal",
                "Above Normal": "Medium",
                "Well Above Normal": "High",
                "No": "Normal",
                "Yes": "High"
            }
            return mapping.get(level, "Normal")

        risk_factors = {}
        risk_factors["BMI"] = assess_risk(inverse_transform(input_df["BMI"].iloc[0], "BMI"), "BMI")
        risk_factors["Systolic BP"] = assess_risk(inverse_transform(input_df["ap_hi"].iloc[0], "ap_hi"), "ap_hi")
        risk_factors["Cholesterol"] = map_risk_level(cholesterol)
        risk_factors["Glucose"] = map_risk_level(gluc)
        risk_factors["Smoking"] = "High" if user_input["smoke"] == 1 else "Normal"
        risk_factors["Alcohol"] = "High" if user_input["alco"] == 1 else "Normal"
        risk_factors["Activity"] = "Normal" if user_input["active"] == 1 else "High"


        st.subheader("Risk Assessment of Each Indicator")

        # Define risk levels
        risk_levels = {"Normal": 1, "Medium": 2, "High": 3}

        risk_numeric = [risk_levels[risk_factors[factor]] for factor in risk_factors]

        fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
        ax.set_facecolor("white")

        color_mapping = {
            "Normal": ("#A5D6A7", "#388E3C"),   # Light green to dark green
            "Medium": ("#FFECB3", "#FBC02D"),   # Light yellow to dark yellow
            "High": ("#EF9A9A", "#D32F2F")      # Light red to dark red
        }

        x_positions = np.arange(len(risk_factors))

        for i, (factor, level) in enumerate(risk_factors.items()):
            level_num = risk_levels[level]
            x = x_positions[i]
            y = 0
            width = 0.5
            height = level_num
            color1, color2 = color_mapping[level]

            cmap = LinearSegmentedColormap.from_list("gradient", [color1, color2])

            grad = np.atleast_2d(np.linspace(0, 1, 256)).T
            ax.imshow(grad, extent=[x - width/2, x + width/2, y, y + height], origin="lower", aspect="auto", cmap=cmap, alpha=1)

        # Set x-axis range
        ax.set_xlim(-0.25, len(risk_factors) - 0.8)

        ax.tick_params(axis="x", labelsize=10, colors="black")
        ax.tick_params(axis="y", labelsize=10, colors="black")
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["Normal", "Medium", "High"], fontsize=10, color="black")
        ax.set_ylim(0, 4)
        plt.xticks(range(len(risk_factors)), risk_factors.keys(), rotation=0, color="black")

        for spine in ax.spines.values():
            spine.set_edgecolor("black")

        for label in ax.get_xticklabels():
            factor = label.get_text()
            level = risk_factors[factor]
            if level in ["Medium", "High"]:
                label.set_fontweight("bold")

        st.pyplot(fig)

        danger_factors = [factor for factor, level in risk_factors.items() if level in ("High", "Medium")]

        if danger_factors:
            st.write("The following indicators need attention:")
            st.markdown(
                "<p style='text-align: left; font-size: 18px; line-height: 1.6;'>" +
                " ".join([f"&middot; <b>{factor}</b>" for factor in danger_factors]) +
                "</p>",
                unsafe_allow_html=True
            )
            st.write("Please adjust your lifestyle accordingly or consult a healthcare professional.")
        else:
            st.write("All your health indicators are normal. Keep up the good work!")


with tabs[1]:
    st.title("❤️ Cardiovascular Disease Risk Assessment")

    raw_df = pd.read_csv("cardio_train.csv", sep=";")
    raw_df["age"] = (raw_df["age"] / 365).round().astype("int")
    raw_df = raw_df[(raw_df["height"] >= 120) & (raw_df["height"] <= 220)]
    raw_df = raw_df[(raw_df["weight"] >= 30) & (raw_df["weight"] <= 180)]
    raw_df = raw_df[(raw_df["ap_hi"] >= 70) & (raw_df["ap_hi"] <= 250)]
    raw_df = raw_df[(raw_df["ap_lo"] >= 40) & (raw_df["ap_lo"] <= 140)]
    raw_df = raw_df[raw_df["ap_hi"] >= raw_df["ap_lo"]]
    raw_df.reset_index(drop=True, inplace=True)
    raw_df["gender"] = raw_df["gender"].map({1: 0, 2: 1})
    raw_df["BMI"] = raw_df["weight"] / ((raw_df["height"] / 100) ** 2)
    raw_df = pd.get_dummies(raw_df, columns=["cholesterol", "gluc"], prefix=["cholesterol", "gluc"])

    df_cardio_1 = raw_df[raw_df["cardio"] == 1]

    total_records = len(df_cardio_1)
    median_age = df_cardio_1["age"].median()
    median_height = df_cardio_1["height"].median()
    median_weight = df_cardio_1["weight"].median()
    median_ap_hi = df_cardio_1["ap_hi"].median()
    median_ap_lo = df_cardio_1["ap_lo"].median()

    gender_mode_val = df_cardio_1["gender"].mode()[0]
    gender_mode = "Male" if gender_mode_val == 1 else "Female"

    chol_sums = df_cardio_1[["cholesterol_1","cholesterol_2","cholesterol_3"]].sum()
    chol_mode_col = chol_sums.idxmax()[-1]
    chol_map = {"1":"Normal","2":"Above Normal","3":"Well Above Normal"}
    chol_mode = chol_map[chol_mode_col]

    gluc_sums = df_cardio_1[["gluc_1","gluc_2","gluc_3"]].sum()
    gluc_mode_col = gluc_sums.idxmax()[-1]
    gluc_map = {"1":"Normal","2":"Above Normal","3":"Well Above Normal"}
    gluc_mode = gluc_map[gluc_mode_col]

    smoke_mode_val = df_cardio_1["smoke"].mode()[0]
    smoke_mode = "Yes" if smoke_mode_val == 1 else "No"

    alco_mode_val = df_cardio_1["alco"].mode()[0]
    alco_mode = "Yes" if alco_mode_val == 1 else "No"

    active_mode_val = df_cardio_1["active"].mode()[0]
    active_mode = "Yes" if active_mode_val == 1 else "No"

    data_overview = [
        ("Total Records", total_records),
        ("Age", median_age),
        ("Gender", gender_mode),
        ("Height (cm)", median_height),
        ("Weight (kg)", median_weight),
        ("Ap_hi (mmHg)", median_ap_hi),
        ("Ap_lo (mmHg)", median_ap_lo),
        ("Cholesterol", chol_mode),
        ("Glucose", gluc_mode),
        ("Smoking", smoke_mode),
        ("Alcohol", alco_mode),
        ("Activity", active_mode)
    ]

    st.markdown(
        "<h4>Data Overview</h4>",
        unsafe_allow_html=True
    )
    st.markdown(
        """<h5 style="color: grey;">The following are the most common user information in the database. Numerical class features are represented by the median, and classification features are represented by the mode.</h5>""",
        unsafe_allow_html=True
    )

    for i in range(0, len(data_overview), 4):
        row = data_overview[i:i+4]
        cols = st.columns(4)
        for j, (label, value) in enumerate(row):
            with cols[j]:
                st.markdown(f"""
                <div style="
                    display:flex; 
                    flex-direction:column; 
                    align-items:center; 
                    justify-content:center; 
                    padding:10px; 
                    margin:5px;
                ">
                    <h6 style="font-size:20px; margin:0; padding:5px 0 0 25px; text-align:center;">{label}</h6>
                    <h7 style="font-size:17px; font-weight:bold; margin:0; padding-top:5px; text-align:center;">{value}</h7>
                </div>
                """, unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(
        "<h4>Data Distribution</h4>",
        unsafe_allow_html=True
    )
    st.write("**Histogram**")
    st.image("distribution.png")

    st.write("**Boxplot**")
    st.image("boxplot.png")

    st.write("**Pairplot**")
    st.image("pairplot_num_features.png")