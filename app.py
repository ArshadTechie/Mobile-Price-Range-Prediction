import streamlit as st
import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import PowerTransformer, StandardScaler

# Loading dataset
df = pd.read_csv(r"C:\\Users\\arsha\\Downloads\\archive (1)\\train.csv")

# Feature manipulation
df['px_height'] = df['px_height'].replace(0, df['px_height'].mean())
df['sc_w'] = df['sc_w'].replace(0, df['sc_w'].mean())

# Feature Engineering
df['screen_area'] = df['sc_h'] * df['sc_w']
df['px_area'] = df['px_height'] * df['px_width']

# Feature selection
X = df[['screen_area', 'n_cores', 'int_memory', 'mobile_wt', 'px_area', 'battery_power', 'ram']]
y = df['price_range']

# Preprocessing using powertransformer
pt = PowerTransformer(copy=False)
X[['screen_area', 'px_area']] = pt.fit_transform(X[['screen_area', 'px_area']])

# Scaling data using standard scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(data=X_scaled, columns=X.columns, index=X.index)

st.image(r"C:\\Users\\arsha\\Downloads\\maxresdefault.jpg", use_column_width=True)
st.title(':phone: :blue[Mobile Price Range Predictor]')
st.subheader(':sparkles: :green[Enter accurate information to predict price range]')

# Attributes regarding system
st.subheader(':red[Enter system information]')
col1, col2, col3, col4 = st.columns(4)
with col1:
    n_cores = st.slider(':green[No. of cores]', 1, 8)
    st.write(':computer: No. of cores:', n_cores)
with col2:
    ram = st.slider(':green[Select Ram in MB]', 256, 4096)
    st.write(':ram: Ram in MB:', ram)
with col3:
    int_mem = st.slider(':green[Internal Memory]', 2, 64)
    st.write(':floppy_disk: Internal Memory:', int_mem)
with col4:
    mob_weight = st.slider(':green[Mobile Weight]', 80, 200)
    st.write(':weight_lifting_man: Mobile weight:', mob_weight)

st.subheader(':red[Enter Screen attributes]')
# Attributes regarding screen
col1, col2, col3, col4 = st.columns(4)
with col1:
    sc_height = st.slider(':green[Select height in cm]', 12, 19)
    st.write(':straight_ruler: Screen Height:', sc_height)
with col2:
    sc_wid = st.slider(':green[Select Width in cm]', 5, 18)
    st.write(':straight_ruler: Screen Width:', sc_wid)
with col3:
    px_height = st.slider(':green[Pixel resolution height]', 100, 1960)
    st.write(':bar_chart: Pixel Height is:', px_height)
with col4:
    px_width = st.slider(':green[Pixel resolution width]', 500, 2000)
    st.write(':bar_chart: Pixel width is:', px_width)

st.subheader(':red[Select Battery power in mAh]')
battery_pow = st.slider(':green[Battery Power]', 501, 2000)
st.write(':battery: Selected battery power is:', battery_pow)

# Taking input and predicting using saved model
if st.button('Predict'):
    input_data = {
        'screen_area': sc_height * sc_wid,
        'n_cores': n_cores,
        'int_memory': int_mem,
        'mobile_wt': mob_weight,
        'px_area': px_height * px_width,
        'battery_power': battery_pow,
        'ram': ram
    }

    # Create a DataFrame for the input data
    input_df = pd.DataFrame([input_data])

    # Preprocessing data
    input_df[['screen_area', 'px_area']] = pt.transform(input_df[['screen_area', 'px_area']])
    input_df_scaled = scaler.transform(input_df)

    # Load saved model
    loaded_model = pickle.load(open(r"C:\\Users\\arsha\\OneDrive\\Desktop\\mobilr_predication\\mobile_predication_prices.sav007", 'rb'))
    y_preds = loaded_model.predict(input_df_scaled)

    # Interpret prediction
    if y_preds == 0:
        st.subheader(':moneybag: Selected phone is low cost phone')
    elif y_preds == 1:
        st.subheader(':money_with_wings: Selected phone is medium cost phone')
    elif y_preds == 2:
        st.subheader(':moneybag: Selected phone is high price phone')
    else:
        st.subheader(':money_mouth_face: Selected phone is very high price phone')