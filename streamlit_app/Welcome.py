import streamlit as st

st.set_page_config(
page_title="Welcome",
page_icon="👋",
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


#The title
st.title("Vehicle Pricing 🚗 ")

#The subheader
st.subheader("Introduction to the use case:")

#The text
st.write("We are one of the most popular car buying and selling platforms in the world. We are going to launch a new product based on a price recommender for users' vehicles. In this application you will be able to explore the data of vehicles advertised in the past, test the prediction model, and understand the model's decisions with the explainability tab.")

st.image("image.jpg", caption="Car Data Analytics & Visualization", use_column_width=True)
st.image("image2.png", caption="Some history about some of our cars", use_column_width=True)