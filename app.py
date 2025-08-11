import streamlit as st
import pandas as pd
from joblib import load
import osS

# ===============================
# Load trained model & data
# ===============================
model_path = os.path.join(os.path.dirname(__file__), "XGBMODEL_pipeline.joblib")
model = load(model_path)
train_data = pd.read_csv("train_data.csv")  # Preprocessed data (same columns as model)

# ===============================
# Define top features & labels
# ===============================
top_features = [
    "Overall Qual", "Gr Liv Area", "Garage Cars", "Total Bsmt SF",
    "Kitchen Qual", "Exter Qual", "Full Bath", "Year Built",
    "Bsmt Qual", "Central Air", "Neighborhood"
]

feature_labels = {
    "Overall Qual": "Overall Material & Finish Quality (1=Poor, 10=Excellent)",
    "Gr Liv Area": "Above Ground Living Area (sq ft)",
    "Garage Cars": "Garage Capacity (Number of Cars)",
    "Total Bsmt SF": "Total Basement Area (sq ft)",
    "Kitchen Qual": "Kitchen Quality",
    "Exter Qual": "Exterior Material Quality",
    "Full Bath": "Number of Full Bathrooms",
    "Year Built": "Year the House Was Built",
    "Bsmt Qual": "Basement Quality",
    "Central Air": "Has Central Air Conditioning?",
    "Neighborhood": "Neighborhood"
}

# Columns that are numeric but actually discrete categories
categorical_like_numbers = [
    "Overall Qual",
    "Garage Cars",
    "Full Bath",
    "Kitchen Qual",
    "Exter Qual",
    "Bsmt Qual"
]

# Mapping codes to user-friendly names for these features
user_friendly_mappings = {
    "Overall Qual": {
        1: "Very Poor",
        2: "Poor",
        3: "Fair",
        4: "Below Average",
        5: "Average",
        6: "Above Average",
        7: "Good",
        8: "Very Good",
        9: "Excellent",
        10: "Very Excellent"
    },
    "Kitchen Qual": {
        1: "Poor",
        2: "Fair",
        3: "Typical",
        4: "Good",
        5: "Excellent"
    },
    "Exter Qual": {
        1: "Poor",
        2: "Fair",
        3: "Typical",
        4: "Good",
        5: "Excellent"
    },
    "Bsmt Qual": {
        1: "Poor",
        2: "Fair",
        3: "Typical",
        4: "Good",
        5: "Excellent"
    },
    "Garage Cars": {
        0: "None",
        1: "1 Car",
        2: "2 Cars",
        3: "3 Cars",
        4: "4 Cars",
        5: "5+ Cars"
    },
    "Full Bath": {
        0: "None",
        1: "1 Bathroom",
        2: "2 Bathrooms",
        3: "3 Bathrooms",
        4: "4 Bathrooms",
        5: "5+ Bathrooms"
    }
}

# ===============================
# Streamlit UI
# ===============================
st.title("🏠 House Price Prediction")
st.write("Fill in the details below to get an estimated sale price for your house:")

user_input = {}

for feature in top_features:
    label = feature_labels.get(feature, feature)

    if feature in categorical_like_numbers:
        # Use user-friendly names in selectbox
        mapping = user_friendly_mappings.get(feature)
        if mapping:
            options = [mapping[code] for code in sorted(mapping.keys())]
            # Get default code from training data mode or use lowest code
            defaults = train_data.mode().iloc[0]
            default_code = defaults.get(feature, min(mapping.keys()))
            default_option = mapping.get(default_code, options[0])
            selected_name = st.selectbox(label, options, index=options.index(default_option))
            # Reverse map back to code
            selected_code = [code for code, name in mapping.items() if name == selected_name][0]
            user_input[feature] = selected_code
        else:
            # fallback if no mapping defined
            options = sorted(train_data[feature].dropna().unique())
            user_input[feature] = st.selectbox(label, options)
    elif pd.api.types.is_numeric_dtype(train_data[feature]):
        min_val = float(train_data[feature].min())
        max_val = float(train_data[feature].max())
        default_val = 0.00 if min_val <= 0 else min_val
        user_input[feature] = st.number_input(
            label, min_value=min_val, max_value=max_val, value=default_val
        )
    else:
        options = sorted(train_data[feature].dropna().unique())
        defaults = train_data.mode().iloc[0]
        default_val = defaults.get(feature, options[0])
        user_input[feature] = st.selectbox(label, options, index=options.index(default_val))

# ===============================
# Fill remaining features with defaults
# ===============================
defaults = train_data.mode().iloc[0]  # Most common value for each column
input_df = defaults.copy()

for feature, value in user_input.items():
    input_df[feature] = value

input_df = pd.DataFrame([input_df])  # Convert to 1-row DataFrame

# ===============================
# Predict Button
# ===============================
if st.button("Predict Sale Price"):
    predicted_price = model.predict(input_df)[0]
    st.success(f"Predicted Price: ${predicted_price:,.2f}")
