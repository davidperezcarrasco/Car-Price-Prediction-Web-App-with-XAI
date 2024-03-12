import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import shap
import lightgbm as lgb
import numpy as np

st.set_page_config(
page_title="Explaniability",
page_icon="💡",
layout="wide",
initial_sidebar_state="expanded")

st.sidebar.header("🗺️ **Navigation Guide**")
st.sidebar.write("Navigate through the different tabs to learn about all the features of this app.")
st.sidebar.markdown("""
- 📢 **Welcome Tab**: Get introduced to the Vehicle Pricing use case.
- 🔍 **Data Exploration Tab**: View interactive visualizations of car data. Customize the visuals by selecting specific brands, body types, or years.
- 💰 **Price Prediction Tab**: Use our model to predict car prices. Input the car's features and get the predicted price.
- 🧠 **Model Explainability Tab**: Understand the factors influencing the predicted prices. Gain insights into the model's decision-making process.
""")
st.sidebar.markdown('---')  # Separator
st.sidebar.header('✉️ **Contact**')
st.sidebar.write('📧 david.perez18@estudiant.upf.edu')
st.sidebar.markdown('---')  # Separator
st.sidebar.header('🔗 **Source Code**')
st.sidebar.write('👉 [GitHub](https://github.com/dperezz02/Cars_Sales)')


#Load the pickle file with the model and the label encoders
def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data["model"]
le_car = data["le_car"]
le_body = data["le_body"]
le_engType = data["le_engType"]
le_drive = data["le_drive"]

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

@st.cache_data
def load_data():
    df =  pd.read_csv("car_ad_display.csv", encoding = "ISO-8859-1", sep=";").drop(columns='Unnamed: 0')

    car_map = shorten_categories(df.car.value_counts(), 10)
    df['car'] = df['car'].map(car_map)

    model_map = shorten_categories(df.model.value_counts(), 10)
    df['model'] = df['model'].map(model_map)

    df = df[df["price"] <= 100000]
    df = df[df["price"] >= 1000]
    df = df[df["mileage"] <= 600]
    df = df[df["engV"] <= 7.5]
    df = df[df["year"] >= 1975]

    return df

df_original = load_data()
df_encoded = df_original.copy()

yes_l = ['yes', 'YES', 'Yes', 'y', 'Y']
df_encoded['registration'] = np.where(df_encoded['registration'].isin(yes_l), 1, 0)

df_encoded['car'] = le_car.fit_transform(df_encoded['car'])

df_encoded['body'] = le_body.fit_transform(df_encoded['body'])

df_encoded['engType'] = le_engType.fit_transform(df_encoded['engType'])

df_encoded['drive'] = le_drive.fit_transform(df_encoded['drive'])

X = df_encoded.drop("price", axis=1)
X = X.drop(columns='model')
y = df_encoded["price"]
X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=42)

st.title('🧠 **Model Explainability**')

#display interactively some information about X_train and X_test
st.header('📊 **Data**')
st.write('The following table shows the training data and labels:')
train_data = pd.concat([X_train, y_train], axis=1)
st.dataframe(train_data)

st.write('The following table shows the test data and labels:')
test_data = pd.concat([X_test, y_test], axis=1)
st.dataframe(test_data)

# Initialize JavaScript visualization code
shap.initjs()

# Assuming that 'model' is your trained machine learning model
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

y_pred_all = model.predict(X)

# Calculate the average predicted price
average_predicted_price = y_pred_all.mean()

st.header('📈 **Average Predicted Price**')
st.write("The average predicted price for all cars is: ", "<span style='font-size: 24px; font-weight: bold;'>${:,.02f}</span>".format(average_predicted_price), unsafe_allow_html=True)

st.header('📊 **SHAP Summary Plot**')
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
fig_summary = plt.gcf()
st.pyplot(fig_summary)
plt.clf()  # Clear the current figure

st.header('📊 **SHAP Bar Plot**')
shap.plots.bar(shap_values, show=False)
plt.tight_layout()
fig_bar = plt.gcf()
st.pyplot(fig_bar)

summary = """
🔍 **Feature Relevance Summary**:
- Year and engV have the highest correlation with the target feature (price).
- Car and drive also show significant relevance to the price.
- Body and registration have less relevance in general.

📈 **Year Feature Insights**:
- Year has the highest variance in its SHAP values, indicating varying relevance in predictions.
- On average, year is the most relevant feature for price predictions.

🚫 **Registration Insight**:
- Registration has the lowest average SHAP value, suggesting a negative impact on car prices.
- Unregistered cars may have a negative effect on the predicted price.

💡 **Key Takeaways**:
- Year and engV are important factors in determining car prices.
- Car and drive also play a significant role.
- Body type and registration have less impact on prices.

"""
# Display the summary
st.header('📝 **Insights Summary**')
st.write(summary)


st.header('📊 **SHAP Scatter Plots**')
st.write('The following scatter plots show the SHAP values for the features "mileage", "engV", and "year":')
shap.plots.scatter(shap_values[:,"mileage"], show=False)
plt.tight_layout()
fig_mileage = plt.gcf()
st.pyplot(fig_mileage)
plt.clf()  # Clear the current figure

shap.plots.scatter(shap_values[:,"engV"], show=False)
plt.tight_layout()
fig_engV = plt.gcf()
st.pyplot(fig_engV)
plt.clf()  # Clear the current figure

shap.plots.scatter(shap_values[:,"year"], show=False)
plt.tight_layout()
fig_year = plt.gcf()
st.pyplot(fig_year)
plt.clf()  # Clear the current figure

st.markdown("""
- 🚗 **Year**: Newer cars (last 5 years) have higher prices. Older cars (30+ years) have smaller price differences.
- 🏎️ **Engine Volume**: Larger engines increase the car price. Engines less than 3 liters decrease the price.
- 🛣️ **Mileage**: Higher mileage often decreases the price. Cars with zero mileage have higher prices.
""")

st.header('📊 **SHAP Dependence Plots**')
st.write('The following dependence plots show the relationship between "engV" and "mileage", and "year" and "mileage" with respect to the SHAP values:')
shap.dependence_plot("engV", shap_values.values, X_test, interaction_index="mileage", show=False)
plt.tight_layout()
fig_engV_mileage = plt.gcf()
st.pyplot(fig_engV_mileage)
plt.clf()  # Clear the current figure

shap.dependence_plot("year", shap_values.values, X_test, interaction_index="mileage", show=False)
plt.tight_layout()
fig_year_mileage = plt.gcf()
st.pyplot(fig_year_mileage)
plt.clf()  # Clear the current figure

st.markdown("""
- 📅 **Year vs Mileage**: Newer cars (last 5 years) have lower mileage. Older cars (30+ years) have higher mileage, except for some collection cars with low mileage.
- 🏎️ **Engine Volume vs Mileage**: Cars with larger engines and lower mileage have higher prices. Cars with less than 3L engine volume and low mileage can have lower prices.
""")

st.header('📊 **SHAP Waterfall Plot**')
st.write('The SHAP waterfall plot shows the breakdown of the prediction for a single instance:')
shap.plots.waterfall(shap_values[0], show=False)
plt.tight_layout()
fig_waterfall = plt.gcf()
st.pyplot(fig_waterfall)
plt.clf()  # Clear the current figure

st.header('📊 **SHAP Force Plot**')
st.write('The SHAP force plot shows the impact of each feature on the prediction:')
shap.plots.force(shap_values[0], matplotlib=True)
plt.tight_layout()
fig_force = plt.gcf()
st.pyplot(fig_force)
plt.clf()  # Clear the current figure

st.header('📊 **SHAP Decision Plot**')
st.write('The SHAP decision plot shows the decision process for a single instance:')
shap.decision_plot(shap_values[0].base_values, shap_values[0].values, X_test.iloc[0], show=False)
plt.tight_layout()
fig_decision = plt.gcf()
st.pyplot(fig_decision)
plt.clf()  # Clear the current figure

st.markdown("""
✨ Congratulations! ✨
You now have a better understanding of your predictions from the Price Prediction page. 
Explore the feature relevance summary, insights, scatter plots, dependence plots, waterfall plot, force plot, and decision plot to gain valuable insights into the factors influencing car prices. 
Happy analyzing! 🚗💰
""")



st.write("""### Enter your car information to predict its price!""")

car_types = df_original['car'].unique()
car = st.selectbox('Car brand', car_types)

body_types = (
    "crossover",
    "sedan",
    "van",
    "vagon",
    "hatch",
    "other",
)

body = st.selectbox("Body", body_types)

mileage = st.slider("Mileage", 0, 600, 80)

engV = st.slider("EngV", 0.0, 7.0, 3.5)

engType_types = (
    "Gas",
    "Petrol",
    "Diesel",
    "Other",
)
engType = st.selectbox("EngType", engType_types)

registered = st.radio(
    "Is it registered?",
    ('Yes', 'No'))

year  = st.slider("Year", 1975, 2015, 2010)

drive_types = (
    "full",
    "rear",
    "front",
)
drive = st.selectbox("Drive", drive_types)

yes_l = ['yes', 'YES', 'Yes', 'y', 'Y']


ok = st.button("Calculate Price")
if ok:
    X_sample = np.array([[car, body, mileage, engV, engType, registered, year, drive ]])
    # Apply the encoder and data type corrections:
    X_sample[:, 0] = str(X_sample[:, 0][0] if X_sample[:, 0][0] in list(df_original['car'].unique()) else 'Other')
    X_sample[:, 0] = le_car.transform(X_sample[:,0])
    X_sample[:, 1] = le_body.transform(X_sample[:,1])
    X_sample[:, 4] = le_engType.transform(X_sample[:,4])
    X_sample[:, 5] = int(1 if X_sample[:, 5][0] in yes_l else 0)
    X_sample[:, 7] = le_drive.transform(X_sample[:,7])

    X_sample = np.array([[
        int(X_sample[0, 0]), 
        int(X_sample[0, 1]), 
        int(X_sample[0, 2]), 
        float(X_sample[0, 3]), 
        int(X_sample[0, 4]), 
        int(X_sample[0, 5]), 
        int(X_sample[0, 6]), 
        int(X_sample[0, 7])
    ]])
   
    price = model.predict(X_sample)
    st.subheader(f"The estimated price is **${price[0]:.2f}**")

    st.markdown("""
    🎉 Great! Now you're going to see how much each feature influenced the final predicted price of your desired car. This will help you understand which aspects of the car contribute most to its price. Let's dive in! 🚗💨
    """)

    # Define the column names
    column_names = ['car', 'body', 'mileage', 'engV', 'engType', 'registered', 'year', 'drive']

    # Create a DataFrame from X_sample with the correct column names
    X_sample_df = pd.DataFrame(X_sample, columns=column_names)

    # Now you can use X_sample_df in your SHAP plots
    shap_values2 = explainer(X_sample_df)

    st.header('📊 **SHAP Waterfall Plot**')
    st.write('The SHAP waterfall plot shows the breakdown of the prediction for a single instance:')
    shap.plots.waterfall(shap_values2[0], show=False)
    plt.tight_layout()
    fig_waterfall = plt.gcf()
    st.pyplot(fig_waterfall)
    plt.clf()  # Clear the current figure

    st.header('📊 **SHAP Force Plot**')
    st.write('The SHAP force plot shows the impact of each feature on the prediction:')
    shap.plots.force(shap_values2[0], feature_names=X_test.columns, matplotlib=True)
    plt.tight_layout()
    fig_force = plt.gcf()
    st.pyplot(fig_force)
    plt.clf()  # Clear the current figure

    st.header('📊 **SHAP Decision Plot**')
    st.write('The SHAP decision plot shows the decision process for a single instance:')
    X_sample_df = pd.DataFrame(X_sample, columns=X_test.columns)
    feature_names = list(X_test.columns)  # Convert the columns to a list
    shap.decision_plot(shap_values2[0].base_values, shap_values2[0].values, X_sample_df.iloc[0], feature_names=feature_names, show=False)
    plt.tight_layout()
    fig_decision = plt.gcf()
    st.pyplot(fig_decision)
    plt.clf()  # Clear the current figure