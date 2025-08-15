import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page config
st.set_page_config(
    page_title="ChurnShield - Customer Retention Engine",
    layout="wide"
)

# Title
st.title("🛡️ ChurnShield – Customer Retention Intelligence Engine")

# Load model and preprocessor
@st.cache_resource
def load_model():
    model_path = "churn_model.pkl"
    preprocessor_path = "preprocessor.pkl"

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    if not os.path.exists(preprocessor_path):
        st.error(f"Preprocessor file not found: {preprocessor_path}")
        st.stop()

    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.stop()

model, preprocessor = load_model()

# Input Form
st.sidebar.header("Customer Information")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 1000.0, 80.0)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 960.0)

if total_charges == 0:
    total_charges = monthly_charges * tenure

clv = st.sidebar.number_input("Customer Lifetime Value ($)", 0.0, 10000.0, float(total_charges * 1.5))
support_tickets = st.sidebar.number_input("Support Tickets (last 6 months)", 0, 20, 1)
days_since_interaction = st.sidebar.slider("Days Since Last Interaction", 0, 365, 30)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Derived feature
avg_monthly_spend = total_charges / (tenure + 1)

# Prepare input data
input_df = pd.DataFrame([{
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'CLV': clv,
    'support_tickets_6m': support_tickets,
    'days_since_last_interaction': days_since_interaction,
    'avg_monthly_spend': avg_monthly_spend,
    'gender': gender,
    'Contract': contract,
    'PaymentMethod': payment_method,
    'InternetService': internet_service
}])

# Tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Feature Impact", "Business Impact"])

# Tab 1: Prediction
with tab1:
    st.header("Churn Risk Prediction")

    if st.button("Predict Churn Risk"):
        try:
            # Preprocess
            input_encoded = preprocessor.transform(input_df)

            # Predict
            churn_prob = model.predict_proba(input_encoded)[0, 1]

            # Display
            st.metric("Churn Probability", f"{churn_prob:.2%}")

            if churn_prob > 0.8:
                st.error("High Risk of Churn")
            elif churn_prob > 0.5:
                st.warning("Medium Risk of Churn")
            else:
                st.success("Low Risk of Churn")

            # Retention Action
            st.subheader("Recommended Action")
            if churn_prob > 0.8:
                if contract == "Month-to-month":
                    st.write("Offer 3-month 10% discount + plan upgrade")
                if internet_service == "Fiber optic":
                    st.write("Assign dedicated support agent")
                if days_since_interaction > 90:
                    st.write("Send personalized re-engagement email + $10 gift card")
            elif churn_prob > 0.5:
                st.write("Send a check-in email with satisfaction survey")
            else:
                st.write("No action needed. Continue nurturing.")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
    else:
        st.info("👈 Adjust inputs and click 'Predict Churn Risk'")

# Tab 2: Simulated Feature Impact
with tab2:
    st.header("Key Drivers of Churn")

    feature_importance = {
        'Contract': 0.3 if contract == "Month-to-month" else 0.05,
        'Tenure': 0.2 if tenure < 6 else 0.05,
        'Monthly Charges': 0.1 if monthly_charges > 90 else 0.04,
        'Support Tickets': 0.15 if support_tickets > 3 else 0.03,
        'Days Since Interaction': 0.15 if days_since_interaction > 180 else 0.05,
        'CLV': 0.1 if clv < 500 else 0.02,
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    features = list(feature_importance.keys())
    impacts = list(feature_importance.values())
    colors = ['red' if x > 0.1 else 'orange' if x > 0.05 else 'green' for x in impacts]
    ax.barh(features, impacts, color=colors)
    ax.set_xlabel("Estimated Impact on Churn Risk")
    ax.set_title("Feature Influence (Simulated)")
    ax.grid(axis='x', alpha=0.3)
    st.pyplot(fig)

# Tab 3: Business Impact
with tab3:
    st.header("Business Impact Analysis")

    if st.button("Calculate Impact") or 'churn_prob' in locals():
        try:
            input_encoded = preprocessor.transform(input_df)
            churn_prob = model.predict_proba(input_encoded)[0, 1]

            revenue_at_risk = clv
            intervention_cost = 25 if churn_prob > 0.8 else 5
            expected_savings = revenue_at_risk * churn_prob * 0.7
            net_benefit = expected_savings - intervention_cost
            roi = (expected_savings / intervention_cost - 1) if intervention_cost > 0 else np.inf

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Revenue at Risk", f"${revenue_at_risk:,.0f}")
            col2.metric("Intervention Cost", f"${intervention_cost}")
            col3.metric("Expected Savings", f"${expected_savings:,.0f}")
            col4.metric("Net Benefit", f"${net_benefit:,.0f}")

            if roi != np.inf:
                st.metric("Estimated ROI", f"{roi:.1f}x")
            else:
                st.metric("Estimated ROI", "Infinite")

        except Exception as e:
            st.error(f"Calculation failed: {str(e)}")
    else:
        st.info("Click 'Calculate Impact' to see financial results.")

# Footer
st.markdown("---")
st.markdown("ChurnShield – Enterprise Customer Retention Intelligence Engine | Model Version: 1.0")
