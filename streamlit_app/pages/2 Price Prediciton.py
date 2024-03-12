import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

st.set_page_config(
page_title="Price Prediction",
page_icon="💰",
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

st.title("🔮 Car price Predictor 🔮")
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


    # Calculate mean price for selected car, year, and body type
    mean_price_car = df_original[df_original['car'] == car]['price'].mean()
    mean_price_year = df_original[df_original['year'] == year]['price'].mean()
    mean_price_body = df_original[df_original['body'] == body]['price'].mean()

    # Create a bar chart using Streamlit
    st.subheader('Price Comparison')

    # Create a DataFrame for visualization
    comparison_data = pd.DataFrame({
        'Category': ['Predicted Price', f'Average Price for {car}', f'Average Price for {year} cars', f'Average Price for {body} body type cars'],
        'Price': [int(price[0]), int(mean_price_car), int(mean_price_year), int(mean_price_body)]
    })

    # Set colors for each bar
    colors = ['blue', 'green', 'orange', 'red']

    # Create an Altair bar chart
    chart = alt.Chart(comparison_data).mark_bar().encode(
        x=alt.X('Category:N', title=None, axis=alt.Axis(labelAngle=0)),  # Rotate x-axis labels
        y=alt.Y('Price', title='Price ($)'),
        color=alt.Color('Category', scale=alt.Scale(range=colors)),
        tooltip=['Category', 'Price']
    ).properties(
        width=alt.Step(80)  # Adjust the width based on your preference
    )

    # Add white labels inside the bars
    text = chart.mark_text(
        align='center',
        baseline='middle',
        dy=20,  # Adjust vertical position of the label
        fill='white',  # Set background color to black
        fontSize=14,
    ).encode(
        text=alt.Text('Price:Q', format='$,.2f')  # Add '$' to the labels
    )

    # Display the Altair chart with labels inside the bars in Streamlit
    st.altair_chart(chart + text, use_container_width=True)

    # Calculate the percentage differences
    percentage_difference_car = ((price[0] - mean_price_car) / mean_price_car) * 100
    percentage_difference_year = ((price[0] - mean_price_year) / mean_price_year) * 100
    percentage_difference_body = ((price[0] - mean_price_body) / mean_price_body) * 100

    # Extract scalar values
    percentage_difference_car2 = np.abs(percentage_difference_car)
    percentage_difference_year2 = np.abs(percentage_difference_year)
    percentage_difference_body2 = np.abs(percentage_difference_body)

    # Display visual text with emojis
    st.subheader("Price Comparison")
    st.markdown(f"The estimated price is **<span style='color: {'green' if percentage_difference_car > 0 else 'red'}'>{percentage_difference_car2:.0f}%</span>** {'higher' if percentage_difference_car > 0 else 'lower'} than the average price for {car} cars! 🚗💰", unsafe_allow_html=True)
    st.markdown(f"The estimated price is **<span style='color: {'green' if percentage_difference_year > 0 else 'red'}'>{percentage_difference_year2:.0f}%</span>** {'higher' if percentage_difference_year > 0 else 'lower'} than the average price for {year} cars! 📅💰", unsafe_allow_html=True)
    st.markdown(f"The estimated price is **<span style='color: {'green' if percentage_difference_body > 0 else 'red'}'>{percentage_difference_body2:.0f}%</span>** {'higher' if percentage_difference_body > 0 else 'lower'} than the average price for {body} body type cars! 🚙💰", unsafe_allow_html=True)

    # Create a bar chart using Streamlit
    st.subheader('Predicted Price by car milleage')
    mileage_values = range(0, 601)  # Generate mileage values from 0 to 600

    predicted_salaries = []  # List to store predicted salaries

    for mileage2 in mileage_values:
        X_sample2 = np.array([[car, body, mileage2, engV, engType, registered, year, drive]])
        # Apply the encoder and data type corrections:
        X_sample2[:, 0] = str(X_sample2[:, 0][0] if X_sample2[:, 0][0] in list(df_original['car'].unique()) else 'Other')
        X_sample2[:, 0] = le_car.transform(X_sample2[:,0])
        X_sample2[:, 1] = le_body.transform(X_sample2[:,1])
        X_sample2[:, 4] = le_engType.transform(X_sample2[:,4])
        X_sample2[:, 5] = int(1 if X_sample2[:, 5][0] in yes_l else 0)
        X_sample2[:, 7] = le_drive.transform(X_sample2[:,7])

        X_sample2 = np.array([[
            int(X_sample2[0, 0]), 
            int(X_sample2[0, 1]), 
            int(X_sample2[0, 2]), 
            float(X_sample2[0, 3]), 
            int(X_sample2[0, 4]), 
            int(X_sample2[0, 5]), 
            int(X_sample2[0, 6]), 
            int(X_sample2[0, 7])
        ]])

        # Predict the salary for the current mileage value
        predicted_salary = model.predict(X_sample2)[0]
        predicted_salaries.append(predicted_salary)

    # Create a linechart of mileage vs. price
    line_data = pd.DataFrame({
        'Mileage': mileage_values,
        'Predicted Price': predicted_salaries
    })

    # Create a line chart using Altair
    line_chart = alt.Chart(line_data).mark_line().encode(
        x='Mileage',
        y=alt.Y('Predicted Price', title='Predicted Price ($)')
    )

    # Display the chart in Streamlit
    st.altair_chart(line_chart, use_container_width=True)

    st.subheader('Predicted Price by car year')
    year_values = range(1975, 2016)  # Generate year values from 1975 to 2015

    predicted_prices = []  # List to store predicted prices

    for year3 in year_values:
        X_sample3 = np.array([[car, body, mileage, engV, engType, registered, year3, drive]])
        # Apply the encoder and data type corrections:
        X_sample3[:, 0] = str(X_sample3[:, 0][0] if X_sample3[:, 0][0] in list(df_original['car'].unique()) else 'Other')
        X_sample3[:, 0] = le_car.transform(X_sample3[:,0])
        X_sample3[:, 1] = le_body.transform(X_sample3[:,1])
        X_sample3[:, 4] = le_engType.transform(X_sample3[:,4])
        X_sample3[:, 5] = int(1 if X_sample3[:, 5][0] in yes_l else 0)
        X_sample3[:, 7] = le_drive.transform(X_sample3[:,7])

        X_sample3 = np.array([[
            int(X_sample3[0, 0]), 
            int(X_sample3[0, 1]), 
            int(X_sample3[0, 2]), 
            float(X_sample3[0, 3]), 
            int(X_sample3[0, 4]), 
            int(X_sample3[0, 5]), 
            int(X_sample3[0, 6]), 
            int(X_sample3[0, 7])
        ]])

        # Predict the price for the current year value
        predicted_price = model.predict(X_sample3)[0]
        predicted_prices.append(predicted_price)

    # Create a linechart of year vs. price
    line_data_year = pd.DataFrame({
        'Year': [str(year) for year in year_values],
        'Predicted Price': predicted_prices
    })

    # Create a line chart using Altair
    line_chart_year = alt.Chart(line_data_year).mark_line().encode(
        x='Year',
        y='Predicted Price'
    )

    # Display the chart in Streamlit
    st.altair_chart(line_chart_year.encode(
        x=alt.X('Year', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Predicted Price', title='Predicted Price ($)')
    ), use_container_width=True)
    


#multicar prediction
st.write("""### Enter multiple car brands to predict and compare their price!""")
st.write("⚠️ Please fill in the information about the car above and select the car brands you want to compare.")
car_types = df_original['car'].unique()
car2 = st.multiselect('Select car brands to compare the predicted prices', car_types, key='car_selector2')
ok2 = st.button("Calculate Price", key='ok2')
if ok2:
    # Predict the price for each selected car
    predicted_prices_car2 = []

    for car_brand in car2:
        X_sample_car2 = np.array([[car_brand, body, mileage, engV, engType, registered, year, drive]])
        # Apply the encoder and data type corrections:
        X_sample_car2[:, 0] = str(X_sample_car2[:, 0][0] if X_sample_car2[:, 0][0] in list(df_original['car'].unique()) else 'Other')
        X_sample_car2[:, 0] = le_car.transform(X_sample_car2[:,0])
        X_sample_car2[:, 1] = le_body.transform(X_sample_car2[:,1])
        X_sample_car2[:, 4] = le_engType.transform(X_sample_car2[:,4])
        X_sample_car2[:, 5] = int(1 if X_sample_car2[:, 5][0] in yes_l else 0)
        X_sample_car2[:, 7] = le_drive.transform(X_sample_car2[:,7])

        X_sample_car2 = np.array([[
            int(X_sample_car2[0, 0]), 
            int(X_sample_car2[0, 1]), 
            int(X_sample_car2[0, 2]), 
            float(X_sample_car2[0, 3]), 
            int(X_sample_car2[0, 4]), 
            int(X_sample_car2[0, 5]), 
            int(X_sample_car2[0, 6]), 
            int(X_sample_car2[0, 7])
        ]])

        # Predict the price for the current car brand
        predicted_price_car2 = model.predict(X_sample_car2)[0]
        predicted_prices_car2.append(predicted_price_car2)

    # Create a bar chart of car brand vs. price
    bar_data2 = pd.DataFrame({
        'Car Brand': car2,
        'Predicted Price': predicted_prices_car2
    })

    # Create a bar chart using Altair with different colors for each bar
    bar_chart2 = alt.Chart(bar_data2).mark_bar().encode(
        x=alt.X('Car Brand', axis=alt.Axis(labelAngle=0)),  # Set labelAngle to 0
        y=alt.Y('Predicted Price', title='Predicted Price ($)'),
        color=alt.Color('Car Brand', scale=alt.Scale(scheme='category20'))  # Set different colors for each bar
    )

    # Add white labels inside the bars
    text2 = bar_chart2.mark_text(
        align='center',
        baseline='middle',
        dy=20,  # Adjust vertical position of the label
        fill='white',  # Set background color to black
        fontSize=14,
    ).encode(
        text=alt.Text('Predicted Price:Q', format='$,.2f')  # Add '$' to the labels
    )

    # Display the chart in Streamlit
    st.altair_chart(bar_chart2+text2, use_container_width=True)

    st.subheader('Predicted Price by car mileage and brand')
    mileage_values = range(0, 601)  # Generate mileage values from 0 to 600

    predicted_prices_by_car = {}  # Dictionary to store predicted prices by car

    for car_brand in car2:
        predicted_prices_car = []  # List to store predicted prices for each car

        for mileage3 in mileage_values:
            X_sample3 = np.array([[car_brand, body, mileage3, engV, engType, registered, year, drive]])
            # Apply the encoder and data type corrections:
            X_sample3[:, 0] = str(X_sample3[:, 0][0] if X_sample3[:, 0][0] in list(df_original['car'].unique()) else 'Other')
            X_sample3[:, 0] = le_car.transform(X_sample3[:,0])
            X_sample3[:, 1] = le_body.transform(X_sample3[:,1])
            X_sample3[:, 4] = le_engType.transform(X_sample3[:,4])
            X_sample3[:, 5] = int(1 if X_sample3[:, 5][0] in yes_l else 0)
            X_sample3[:, 7] = le_drive.transform(X_sample3[:,7])

            X_sample3 = np.array([[
                int(X_sample3[0, 0]), 
                int(X_sample3[0, 1]), 
                int(X_sample3[0, 2]), 
                float(X_sample3[0, 3]), 
                int(X_sample3[0, 4]), 
                int(X_sample3[0, 5]), 
                int(X_sample3[0, 6]), 
                int(X_sample3[0, 7])
            ]])

            # Predict the price for the current car brand and mileage value
            predicted_price_car = model.predict(X_sample3)[0]
            predicted_prices_car.append(predicted_price_car)

        predicted_prices_by_car[car_brand] = predicted_prices_car

    # Create a line chart for each car using Altair
    line_data_by_car = pd.DataFrame({
        'Mileage': list(mileage_values) * len(car2),
        'Car Brand': [car_brand for car_brand in car2 for _ in mileage_values],
        'Predicted Price': [predicted_prices_by_car[car_brand][i] for car_brand in car2 for i in range(len(mileage_values))]
    })

    line_chart_by_car = alt.Chart(line_data_by_car).mark_line().encode(
        x='Mileage',
        y=alt.Y('Predicted Price', title='Predicted Price ($)'),
        color='Car Brand'
    )

    # Display the chart in Streamlit
    st.altair_chart(line_chart_by_car, use_container_width=True)

    st.subheader('Predicted Price by car year and brand')
    year_values = range(1975, 2016)  # Generate year values from 1975 to 2015
    predicted_prices_by_car_year = {}  # Dictionary to store predicted prices by car

    for car_brand2 in car2:
        predicted_prices_car_year = []  # List to store predicted prices for each car

        for year4 in year_values:
            X_sample4 = np.array([[car_brand2, body, mileage, engV, engType, registered, year4, drive]])
            # Apply the encoder and data type corrections:
            X_sample4[:, 0] = str(X_sample4[:, 0][0] if X_sample4[:, 0][0] in list(df_original['car'].unique()) else 'Other')
            X_sample4[:, 0] = le_car.transform(X_sample4[:,0])
            X_sample4[:, 1] = le_body.transform(X_sample4[:,1])
            X_sample4[:, 4] = le_engType.transform(X_sample4[:,4])
            X_sample4[:, 5] = int(1 if X_sample4[:, 5][0] in yes_l else 0)
            X_sample4[:, 7] = le_drive.transform(X_sample4[:,7])

            X_sample4 = np.array([[
                int(X_sample4[0, 0]), 
                int(X_sample4[0, 1]), 
                int(X_sample4[0, 2]), 
                float(X_sample4[0, 3]), 
                int(X_sample4[0, 4]), 
                int(X_sample4[0, 5]), 
                int(X_sample4[0, 6]), 
                int(X_sample4[0, 7])
            ]])

            # Predict the price for the current car brand and year value
            predicted_price_car_year = model.predict(X_sample4)[0]
            predicted_prices_car_year.append(predicted_price_car_year)

        predicted_prices_by_car_year[car_brand2] = predicted_prices_car_year

    # Create a line chart for each car using Altair
    line_data_by_car_year = pd.DataFrame({
        'Year': [str(year) for year in year_values] * len(car2),
        'Car Brand': [car_brand2 for car_brand2 in car2 for _ in year_values],
        'Predicted Price': [predicted_prices_by_car_year[car_brand2][i] for car_brand2 in car2 for i in range(len(year_values))]
    })

    line_chart_by_car_year = alt.Chart(line_data_by_car_year).mark_line().encode(
        x=alt.X('Year', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Predicted Price', title='Predicted Price ($)'),
        color='Car Brand'
    )
    # Display the chart in Streamlit
    st.altair_chart(line_chart_by_car_year, use_container_width=True)

