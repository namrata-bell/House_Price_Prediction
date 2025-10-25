import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

st.markdown("""
<style>
.prediction-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2.5rem;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin: 2rem 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
    color: white;
    padding: 0.75rem;
    font-size: 1.3rem;
    border-radius: 12px;
    border: none;
    font-weight: bold;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #45a049 0%, #4CAF50 100%);
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

LOCATION_MULTIPLIERS = {
    'Mumbai': 2.2,
    'Bengaluru': 1.5,
    'Pune': 1.0,
    'Bijapur': 0.4
}

NAME_TO_CODE = {'Bijapur': 0, 'Mumbai': 1, 'Pune': 2, 'Bengaluru': 3}

@st.cache_resource
def load_model():
    try:
        model_files = [f for f in os.listdir('.') if 'best' in f and f.endswith('.pkl')]
        model_file = model_files[0] if model_files else 'bestmodel_lasso_regression.pkl'
        
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        with open('scaler2.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model_info2.json', 'r') as f:
            info = json.load(f)
        try:
            with open('label_encoders.pkl', 'rb') as f:
                encoders = pickle.load(f)
        except:
            encoders = {}
        
        st.sidebar.success("✅ Model loaded")
        return model, scaler, info, encoders
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

model, scaler, model_info, label_encoders = load_model()

if model is None:
    st.stop()

st.title("🏠 Indian Real Estate Price Predictor")
st.markdown("### AI-Powered Property Valuation")

col1, col2, col3 = st.columns(3)

with col1:
    model_name = model_info.get('model_name', 'ML Model')
    display_name = model_name.split()[0] if ' ' in model_name else model_name
    st.metric("🤖 Model", display_name)

with col2:
    n_samples = model_info.get('n_samples', 0)
    st.metric("📊 Properties", f"{n_samples:,}")

with col3:
    st.metric("🏙️ Cities", "4")

st.markdown("---")

left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown("### 📝 Property Specifications")
    
    area = st.number_input("🏡 Area (sq ft)", 300, 15000, 1500, 100)
    
    col_bed, col_bath = st.columns(2)
    with col_bed:
        bedrooms = st.number_input("🛏️ Bedrooms", 1, 10, 3)
    with col_bath:
        bathrooms = st.number_input("🚿 Bathrooms", 1, 10, 2)
    
    col_floor, col_year = st.columns(2)
    with col_floor:
        floors = st.number_input("🏢 Floors", 1, 5, 2)
    with col_year:
        year_built = st.number_input("📅 Year Built", 1950, 2025, 2015)
    
    property_age = 2025 - year_built
    st.info(f"🕐 Property Age: {property_age} years")

with right_col:
    st.markdown("### 🎯 Location & Features")
    
    location_options = {
        'Mumbai': '🏙️ Mumbai - Most Expensive',
        'Bengaluru': '💻 Bengaluru - Premium',
        'Pune': '🎓 Pune - Moderate',
        'Bijapur': '🏘️ Bijapur - Affordable'
    }
    
    location_display = st.selectbox("📍 City", list(location_options.values()))
    location = location_display.split(' - ')[0].split(' ')[1]
    location_code = NAME_TO_CODE[location]
    multiplier = LOCATION_MULTIPLIERS[location]
    
    city_colors = {
        'Mumbai': ('🔴', 'error'),
        'Bengaluru': ('🟠', 'warning'),
        'Pune': ('🟡', 'info'),
        'Bijapur': ('🟢', 'success')
    }
    
    emoji, color = city_colors[location]
    getattr(st, color)(f"{emoji} {location} - {multiplier}x pricing factor")
    
    condition = st.selectbox("⭐ Condition", ['Excellent', 'Good', 'Fair', 'Poor'], index=1)
    garage = st.radio("🚗 Garage", ['Yes', 'No'], horizontal=True)

st.markdown("---")

if st.button("🔮 Calculate Price", type="primary", use_container_width=True):
    
    with st.spinner("🔄 Calculating..."):
        try:
            input_data = {
                'Area': area,
                'Bedrooms': bedrooms,
                'Bathrooms': bathrooms,
                'Floors': floors,
                'YearBuilt': year_built,
                'PropertyAge': property_age,
                'Location': location_code,
                'Condition': condition,
                'Garage': 1 if garage == 'Yes' else 0
            }
            
            input_df = pd.DataFrame([input_data])
            
            condition_map = {'Excellent': 4, 'Good': 3, 'Fair': 2, 'Poor': 1}
            if 'Condition' in label_encoders:
                try:
                    input_df['Condition'] = label_encoders['Condition'].transform([condition])
                except:
                    input_df['Condition'] = condition_map[condition]
            else:
                input_df['Condition'] = condition_map[condition]
            
            # Feature engineering
            input_df['Area_squared'] = input_df['Area'] ** 2
            input_df['Area_cubed'] = input_df['Area'] ** 3
            input_df['Area_sqrt'] = np.sqrt(input_df['Area'])
            input_df['Bedrooms_squared'] = input_df['Bedrooms'] ** 2
            input_df['Bedrooms_cubed'] = input_df['Bedrooms'] ** 3
            input_df['Bedrooms_sqrt'] = np.sqrt(input_df['Bedrooms'])
            input_df['Bathrooms_squared'] = input_df['Bathrooms'] ** 2
            input_df['Bathrooms_cubed'] = input_df['Bathrooms'] ** 3
            input_df['Bathrooms_sqrt'] = np.sqrt(input_df['Bathrooms'])
            input_df['Area_x_Bedrooms'] = input_df['Area'] * input_df['Bedrooms']
            input_df['Area_div_Bedrooms'] = input_df['Area'] / (input_df['Bedrooms'] + 1)
            input_df['Bedrooms_x_Bathrooms'] = input_df['Bedrooms'] * input_df['Bathrooms']
            input_df['Bedrooms_div_Bathrooms'] = input_df['Bedrooms'] / (input_df['Bathrooms'] + 1)
            
            for feat in model_info['features']:
                if feat not in input_df.columns:
                    input_df[feat] = 0
            
            input_df = input_df[model_info['features']]
            
            if model_info['model_name'] in ['Ridge Regression', 'Lasso Regression', 'ElasticNet']:
                input_scaled = scaler.transform(input_df)
                pred_base = model.predict(input_scaled)[0]
            else:
                pred_base = model.predict(input_df)[0]
            
            if model_info.get('use_log_transform', False):
                pred_base = np.expm1(pred_base)
            
            pred_final = pred_base * multiplier
            
            mae = model_info.get('test_mae', pred_final * 0.2)
            lower_bound = max(0, pred_final - mae)
            upper_bound = pred_final + mae
            
            st.balloons()
            st.success("✅ Valuation Complete!")
            
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="margin:0; font-size:1.8rem;">🏠 Estimated Market Value</h2>
                <h1 style="margin:1.5rem 0; font-size:4rem; font-weight:bold;">
                    ₹{pred_final:,.0f}
                </h1>
                <p style="font-size:2rem; margin:0.5rem 0; font-weight:600;">
                    {pred_final/100000:.2f} Lakhs
                </p>
                <p style="font-size:1.3rem; margin:1rem 0;">
                    {pred_final/10000000:.2f} Crores
                </p>
                <p style="font-size:1.1rem; margin-top:1.5rem; padding-top:1rem; 
                          border-top:2px solid rgba(255,255,255,0.3);">
                    📍 {location} | Factor: {multiplier}x
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("💰 Lower", f"₹{lower_bound:,.0f}", f"{lower_bound/100000:.1f}L")
            
            with col2:
                st.metric("🎯 Expected", f"₹{pred_final:,.0f}", f"{pred_final/100000:.1f}L")
            
            with col3:
                st.metric("📈 Upper", f"₹{upper_bound:,.0f}", f"{upper_bound/100000:.1f}L")
            
            with st.expander("🌍 Price Comparison Across Cities", expanded=True):
                st.markdown("### Same Property in Different Cities")
                
                comparison_data = []
                for rank, (city, mult) in enumerate(
                    sorted(LOCATION_MULTIPLIERS.items(), key=lambda x: x[1], reverse=True), 1
                ):
                    city_price = pred_base * mult
                    diff_pct = ((city_price - pred_final) / pred_final * 100) if pred_final else 0
                    
                    comparison_data.append({
                        'Rank': f"#{rank}",
                        'City': city,
                        'Price': f"₹{city_price:,.0f}",
                        'Lakhs': f"{city_price/100000:.2f}L",
                        'Crores': f"{city_price/10000000:.2f}Cr",
                        'Factor': f"{mult}x",
                        'vs Selected': f"{diff_pct:+.1f}%"
                    })
                
                comp_df = pd.DataFrame(comparison_data)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
                
                max_price = pred_base * 2.2
                min_price = pred_base * 0.4
                ratio = max_price / min_price if min_price > 0 else 0
                
                st.info(f"💡 Mumbai properties cost {ratio:.1f}x more than Bijapur")
            
            with st.expander("📋 Property Details"):
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.markdown("#### 🏗️ Specifications")
                    st.write(f"**Area:** {area:,} sq.ft")
                    st.write(f"**Bedrooms:** {bedrooms}")
                    st.write(f"**Bathrooms:** {bathrooms}")
                    st.write(f"**Floors:** {floors}")
                    st.write(f"**Year Built:** {year_built}")
                    st.write(f"**Age:** {property_age} years")
                    st.write(f"**Condition:** {condition}")
                    st.write(f"**Garage:** {garage}")
                
                with detail_col2:
                    st.markdown("#### 💰 Price Breakdown")
                    st.write(f"**Base Value:** ₹{pred_base:,.0f}")
                    st.write(f"**Location:** {location}")
                    st.write(f"**Factor:** {multiplier}x")
                    st.write(f"**Final:** ₹{pred_final:,.0f}")
                    st.write("---")
                    st.write(f"**Per sq.ft:** ₹{pred_final/area:,.0f}")
                    st.write(f"**Per bedroom:** ₹{pred_final/bedrooms:,.0f}")
            
            with st.expander("💡 Investment Insights"):
                st.markdown(f"#### 📍 {location} Market Analysis")
                
                insights = {
                    'Mumbai': {
                        'trend': '📈 Strong (8-12% yearly)',
                        'demand': '🔥 Very High',
                        'rental': '💰 High yields (3-4%)',
                        'liquidity': '⚡ Excellent',
                        'note': 'Premium market with strong returns. High entry cost but excellent appreciation.'
                    },
                    'Bengaluru': {
                        'trend': '📈 Good (7-10% yearly)',
                        'demand': '🔥 High',
                        'rental': '💰 Good yields (3-3.5%)',
                        'liquidity': '✅ Good',
                        'note': 'IT hub with consistent demand. Good for investment and rental income.'
                    },
                    'Pune': {
                        'trend': '📊 Moderate (6-8% yearly)',
                        'demand': '👥 Balanced',
                        'rental': '💵 Moderate (2.5-3%)',
                        'liquidity': '✅ Good',
                        'note': 'Balanced market with steady growth. Lower entry barrier than Mumbai/Bengaluru.'
                    },
                    'Bijapur': {
                        'trend': '📉 Slow (3-5% yearly)',
                        'demand': '👤 Low',
                        'rental': '💵 Low yields (2-2.5%)',
                        'liquidity': '⚠️ Limited',
                        'note': 'Best for end-use. Main advantage is affordability. Limited appreciation expected.'
                    }
                }
                
                info = insights[location]
                
                col_i1, col_i2 = st.columns(2)
                with col_i1:
                    st.write(f"**Growth:** {info['trend']}")
                    st.write(f"**Demand:** {info['demand']}")
                with col_i2:
                    st.write(f"**Rental:** {info['rental']}")
                    st.write(f"**Liquidity:** {info['liquidity']}")
                
                st.info(f"**Recommendation:** {info['note']}")
                
                st.markdown("#### 📊 Financial Metrics")
                m_col1, m_col2, m_col3 = st.columns(3)
                
                with m_col1:
                    monthly_rent = pred_final * 0.0025
                    st.metric("Monthly Rent", f"₹{monthly_rent:,.0f}")
                
                with m_col2:
                    yields = {'Mumbai': 3.5, 'Bengaluru': 3.0, 'Pune': 2.75, 'Bijapur': 2.25}
                    yield_val = yields.get(location, 3.0)
                    st.metric("Rental Yield", f"{yield_val}%")
                
                with m_col3:
                    payback = 100 / yield_val if yield_val > 0 else 0
                    st.metric("Payback", f"{payback:.0f} yrs")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            with st.expander("Debug Info"):
                st.exception(e)

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/home.png", width=80)
    
    st.markdown("## 📊 Model Info")
    
    model_name = model_info.get('model_name', 'Unknown')
    st.write(f"**Algorithm:** {model_name}")
    st.write(f"**Properties:** {model_info.get('n_samples', 0):,}")
    st.write(f"**Features:** {model_info.get('n_features', 0)}")
    
    rmse = model_info.get('test_rmse', 0)
    mae = model_info.get('test_mae', 0)
    
    if rmse > 0:
        st.write(f"**RMSE:** ₹{rmse:,.0f}")
    if mae > 0:
        st.write(f"**MAE:** ₹{mae:,.0f}")
    
    st.markdown("---")
    st.markdown("### 🏙️ City Rankings")
    
    for rank, (city, mult) in enumerate(
        sorted(LOCATION_MULTIPLIERS.items(), key=lambda x: x[1], reverse=True), 1
    ):
        emojis = ["🔴", "🟠", "🟡", "🟢"]
        st.markdown(f"{emojis[rank-1]} **{rank}. {city}** — {mult}x")
    
    st.markdown("---")
    st.info("""
    **Price Factors:**
    
    🔴 Mumbai (2.2x)
    Financial capital
    
    🟠 Bengaluru (1.5x)
    IT hub
    
    🟡 Pune (1.0x)
    Baseline
    
    🟢 Bijapur (0.4x)
    Tier-3 city
    """)
    
    st.markdown("---")
    st.caption("🏠 Built with Streamlit")
    st.caption("© 2025 v1.0")

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:gray; padding:1.5rem;">
    <p style="font-size:1.1rem; font-weight:600;">
        🏠 Indian Real Estate Valuation System
    </p>
    <p style="font-size:0.85rem;">
        AI-Powered Property Price Predictions
    </p>
    <p style="font-size:0.75rem; line-height:1.6; max-width:700px; margin:1rem auto;">
        <b>Disclaimer:</b> Predictions are estimates based on ML models and market factors. 
        Actual prices may vary. For investment decisions, consult real estate professionals.
    </p>
    <p style="font-size:0.7rem; margin-top:0.5rem;">
        © 2025 All Rights Reserved
    </p>
</div>
""", unsafe_allow_html=True)
