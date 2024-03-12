import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from mpl_toolkits.mplot3d import Axes3D
from plotly.subplots import make_subplots
import altair as alt

st.set_page_config(
page_title="Data Exploration",
page_icon="📊",
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

#The title and text
st.title("Data Exploration 📊 ")
st.write("In this tab we can see the most relevant information that we can extract through the data from the visual analytics.")

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


def plot_features_against_target(df):
    target = 'price'
    features = [x for x in df.columns if x not in ["car", "model", "registration", target]]
    n_cols = 3
    n_rows = (len(features) + 2) // n_cols
    sns.set(font_scale=2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6))
    axes = axes.flatten()
    
    # Plot each feature against the target variable in the dataframe
    for i, feature in enumerate(features):
        ax = axes[i]
        if df[feature].dtype == 'object' or df[feature].nunique() < 10:
            # For categorical data, use a boxplot or violin plot
            sns.boxplot(x=feature, y=target, data=df, ax=ax)
        else:
            # For numerical data, use a scatter plot
            sns.scatterplot(x=feature, y=target, data=df, ax=ax)
        ax.set_title(f'{feature} vs {target}')
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    # Hide any unused subplots
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes[j])
    fig.tight_layout()

    return fig


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

    df = df.dropna()  # Drop NaN values

    return df

df = load_data()

#Basic info:
st.subheader("How does the data look like?")
rows = len(df)
columns =len(df.columns)
st.write("We have:")
st.write(str(rows)+" rows")
st.write(str(columns)+" columns.")
st.dataframe(df.head(3))

#Target:
st.subheader("Distribution of the target:")
sns.set(font_scale=0.5)
sns.set_palette("icefire")
fig = plt.figure(figsize=(5, 1.5))
sns.kdeplot(x="price", data=df, fill=True)
plt.title("Prices")
st.pyplot(fig)

#Most expensive cars:
st.subheader("Top 10 Most expensive cars:")
df_priceByCar = df[['car','price']].groupby('car').mean().reset_index()
df_priceByCar = df_priceByCar.sort_values('price', ascending=False).head(10)
fig = plt.figure(figsize=(5, 2))
ax = sns.barplot(data=df_priceByCar, x="price", y="car", palette= "icefire")
ax.bar_label(ax.containers[0], fontsize=5)
st.pyplot(fig)


#Features distribution vs target:
st.subheader("Distribution of the features against the target:")
fig= plot_features_against_target(df)
st.pyplot(fig)

# Read
with open('model.pkl', 'rb') as file:
    data = pickle.load(file)
le_car = data["le_car"]
le_body = data["le_body"]
le_engType = data["le_engType"]
le_drive = data["le_drive"]

# Encode categorical variables
df_encoded = df.copy()
df_encoded['car'] = le_car.transform(df_encoded['car'])
df_encoded['body'] = le_body.transform(df_encoded['body'])
df_encoded['engType'] = le_engType.transform(df_encoded['engType'])
df_encoded['drive'] = le_drive.transform(df_encoded['drive'])

# Concatenate encoded categorical variables with numerical variables
df_concatenated = pd.concat([df_encoded[['car', 'body', 'engType', 'drive']], df.select_dtypes(include='number')], axis=1)

# Calculate correlation matrix
correlation_matrix = df_concatenated.corr()

# Plot correlation matrix using heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', annot_kws={"size": 10})
plt.title('Correlation Matrix')
st.pyplot(plt.gcf())

# Bar chart of count of cars:
st.subheader("Count of cars:")
sns.set(font_scale=1)
sns.set_palette("viridis")  # Change palette to "viridis"
fig = plt.figure(figsize=(10, 10))
sns.countplot(y='car', data=df, palette="viridis", order=df['car'].value_counts().index)
plt.xlabel("Number of cars")
plt.ylabel("Car brand")
plt.title("Count of Cars by Brand")
plt.tight_layout()
st.pyplot(fig)

# Bar chart of count of body types:
st.subheader("Count of body types:")
sns.set(font_scale=1)
sns.set_palette("viridis")  # Change palette to "viridis"
fig = plt.figure(figsize=(10, 10))
sns.countplot(y='body', data=df, palette="viridis", order=df['body'].value_counts().index)
plt.xlabel("Number of cars")
plt.ylabel("Body type")
plt.title("Count of Cars by Body Type")
plt.tight_layout()
st.pyplot(fig)


# Distribution of the target by car:
st.subheader("Distribution of the price by car:")
sns.set(font_scale=1)
sns.set_palette("coolwarm")  # Reverse the color order
# Calculate the median price for each car
median_prices = df.groupby('car')['price'].median().sort_values(ascending=False)  # Reverse the order
# Create a new column to specify the order of the cars
df['car_order'] = df['car'].map(median_prices)
fig = plt.figure(figsize=(10, 10))
sns.boxplot(x='car', y='price', data=df, order=median_prices.index, palette="coolwarm")  # Reverse the color order
plt.xticks(rotation=90)
plt.xlabel("Car Brand")
plt.ylabel("Price ($)")
st.pyplot(fig)

# Ask the user for car brands to compare:
st.subheader("Price distribution by car brands:")
selected_brands = st.multiselect("Select multiple car brands:", df['car'].unique())

if selected_brands:
    # Filter the dataframe based on the selected brands
    df_selected_brands = df[df['car'].isin(selected_brands)]
    # Calculate the median price for each selected brand
    median_prices = df_selected_brands.groupby('car')['price'].median().sort_values(ascending=False)
    # Reorder the selected brands based on median price
    df_selected_brands['car'] = pd.Categorical(df_selected_brands['car'], categories=median_prices.index, ordered=True)
    # Generate side-by-side box plots comparing the price distributions of the selected brands
    fig = plt.figure(figsize=(10, 10))
    sns.boxplot(x='car', y='price', data=df_selected_brands, order=median_prices.index, palette="Spectral")
    plt.xticks(rotation=0)  # Set rotation to 0 degrees
    plt.xlabel("Car Brand")
    plt.ylabel("Price ($)")
    plt.title("Price Distribution by Car Brands")
    st.pyplot(fig)



# Scatter plot of price vs mileage.
st.subheader("Scatter plot of price vs mileage for by brand:")
# Ask the user for a car. While no car is selected, use all data:
car_options = ['All'] + df['car'].unique().tolist()
car = st.selectbox("Select a car:", car_options)
df_car = df.copy()
if car == 'All':
    st.write("No car selected. All car brands are being used. 🚗🚙🚌🏎️")
else:
    df_car = df_car[df_car['car'] == car]
scatter_plot = alt.Chart(df_car).mark_circle().encode(
    x=alt.X('mileage', title='Mileage'),
    y=alt.Y('price', title='Price ($)'),
    tooltip=['mileage', 'price', 'year', 'car']
).interactive()

st.altair_chart(scatter_plot, use_container_width=True)

#3D Interactive Scatter plot of price vs mileage vs year:
import plotly.express as px
st.subheader("3D Interactive Scatter plot of price vs mileage vs year")
car2 = st.selectbox("Select a car:", car_options, key="car_selection")
df_car2 = df.copy()
if car2 == 'All':
    st.write("No car selected. All car brands are being used. 🚗🚙🚌🏎️")    
else:
    df_car2 = df_car2[df_car2['car'] == car2]

fig = px.scatter_3d(df_car2, x='mileage', y='price', z='year', color='year', color_continuous_scale='blackbody')
fig.update_layout(scene=dict(xaxis_title='Mileage', yaxis_title='Price ($)', zaxis_title='Year'), title='3D Interactive Scatter plot of price vs mileage vs year for '+ car2 + ' Cars')
fig.update_layout(width=800, height=800)  # Set the width and height of the plot
st.plotly_chart(fig, use_container_width=True)  # Use container width to occupy the entire layout

#Line plot of mean price by year:
st.subheader("Line plot of mean price by year and brand:")
car3 = st.selectbox("Select a car:", car_options, key="car_selection2")
df_car3 = df.copy()
if car3 == 'All':
    st.write("No car selected. All car brands are being used. 🚗🚙🚌🏎️")    
else:
    df_car3 = df_car3[df_car3['car'] == car3]
sns.set(font_scale=1)
sns.set_palette("Spectral")  # Reverse the color order
fig = plt.figure(figsize=(10, 10))
sns.lineplot(x='year', y='price', data=df, estimator='mean', ci=None)
plt.xlabel("Year")
plt.ylabel("Mean price ($)")
plt.title("Mean price by year of " + car3 + " cars")
plt.tight_layout()
st.pyplot(fig)

import plotly.graph_objects as go
st.subheader("Pie chart of count of top cars by year selected:")
year = st.slider("Year", 1975, 2015, 2010)
df_year = df.copy()
df_year = df_year[df_year['year'] == year]
top_20_cars = df_year['car'].value_counts().nlargest(20)

fig = go.Figure(data=[go.Pie(labels=top_20_cars.index, values=top_20_cars.values, hoverinfo='label+percent')])
fig.update_layout(
    title = "Number of top brands cars in " + str(year),
    width=1000,
    height=600)

st.plotly_chart(fig)


#Give the user the option to download the data:
st.subheader("Download the data:")
st.write("If you want to download the data, please click on the button below:")
if st.button('Download the data'):
    df.to_csv('car_ad_display.csv', index=False)
    st.write('The data has been downloaded successfully! ✅')
else:
    st.write('Click on the button to download the data! 📥')
