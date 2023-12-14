import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
from gradientai import Gradient
os.environ['GRADIENT_ACCESS_TOKEN'] = st.secrets["GRADIENT_ACCESS_TOKEN"]
os.environ['GRADIENT_WORKSPACE_ID'] = st.secrets["GRADIENT_WORKSPACE_ID"]

# Load the dataset with a specified encoding
data = pd.read_csv('/content/fooddrive.csv', encoding='latin1')

# Page 1: Dashboard
def dashboard():
    st.title("Edmonton Food Drive Dashboard")
    st.image("/content/Untitled design.jpg")

    st.subheader("💡 Abstract:")
    inspiration = '''
    Food drives are crucial for addressing hunger and supporting vulnerable communities.
    They provide essential sustenance to those in need, fostering a sense of community and compassion.
    We Norquesters are super proud to be a part of the food drive.
    '''
    st.write(inspiration)

    st.subheader("👨🏻‍💻 What our Project Does?")
    what_it_does = '''
    This machine learning model mainly focuses on predicting the number of bags collected. With this, the volunteering experience and resource allocation can be optimized.
    Further, our team has also proposed an optimized version of the data collection form.
    By addressing logistical challenges, it will enhance efficiency and help vulnerable communities receive essential sustenance more effectively.
    '''
    st.write(what_it_does)

def stakeholders():
    st.image(['/content/WhatsApp Image 2023-12-13 at 14.40.27_6533e7d9.jpg', '/content/WhatsApp Image 2023-12-13 at 14.40.17_a145700a.jpg','/content/WhatsApp Image 2023-12-13 at 14.40.37_7307f472.jpg','/content/WhatsApp Image 2023-12-13 at 14.40.52_121eb9c6.jpg','/content/WhatsApp Image 2023-12-13 at 14.42.42_e23edd29.jpg'])

def data_collection():
    st.title("Data Collection")
    st.write("Please fill out the Google form to contribute to our Food Drive!")
    google_form_url = "https://forms.gle/NQX4z9WwqhJaeUFV8"
    st.markdown(f"[Data Collection Form]({google_form_url})")

def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")
    # Rename columns for clarity
    data_cleaned = data.rename(columns={
        'Timestamp': 'Date',
        'Drop Off Location': 'Location',
        'City': 'City',
        'Stake': 'Stake',
        'Route Number/Name': 'Route',
        '# of Adult Volunteers in this route': '# of Adult Volunteers',
        '# of Youth Volunteers in this route': '# of Youth Volunteers',
        '# of Donation Bags Collected/Route': 'Donation Bags Collected',
        'Time to Complete (in minutes) pick up of bags /route': 'Time to Complete (min)',
        'Did you complete more than 1 route?': 'Completed More Than One Route',
        'Number of routes completed': 'Routes Completed',
        '# of Doors in Route': 'Doors in Route'
    })

    # Visualize the distribution of numerical features using Plotly
    fig = px.histogram(data_cleaned, x='# of Adult Volunteers', nbins=20, labels={'# of Adult Volunteers': 'Adult Volunteers'})
    st.plotly_chart(fig)

    fig = px.histogram(data_cleaned, x='# of Youth Volunteers', nbins=20, labels={'# of Youth Volunteers': 'Youth Volunteers'})
    st.plotly_chart(fig)

    fig = px.histogram(data_cleaned, x='Donation Bags Collected', nbins=20, labels={'Donation Bags Collected': 'Donation Bags Collected'})
    st.plotly_chart(fig)

    fig = px.histogram(data_cleaned, x='Time to Complete (min)', nbins=20, labels={'Time to Complete (min)': 'Time to Complete'})
    st.plotly_chart(fig)


def machine_learning_modeling():
    st.title("Machine Learning Modeling")
    st.write("Enter the details to predict donation bags:")

    # Input fields for user to enter data
    routes_completed = st.slider("Routes Completed", 1, 10, 5)
    time_spent = st.slider("Time Spent (minutes)", 10, 300, 60)
    adult_volunteers = st.slider("Number of Adult Volunteers", 1, 50, 10)
    doors_in_route = st.slider("Number of Doors in Route", 10, 500, 100)


    # Predict button
    if st.button("Predict"):
        # Load the trained model
        model = joblib.load('/content/random_forest_regressor_model.pkl')

        # Prepare input data for prediction
        input_data = [[routes_completed, time_spent, adult_volunteers, doors_in_route]]

        # Make prediction
        prediction = model.predict(input_data)

        # Display the prediction
        st.success(f"Predicted Donation Bags: {prediction[0]}")
def chatbot():
    st.title("Interactive Food Drive Assistant")
    st.write("Ask a question about the Food Drive!")

    with Gradient() as gradient:
        base_model = gradient.get_base_model(base_model_slug="nous-hermes2")
        new_model_adapter = base_model.create_model_adapter(name="interactive_food_drive_model")

        user_input = st.text_input("Ask your question:")
        if user_input and user_input.lower() not in ['quit', 'exit']:
            sample_query = f"### Instruction: {user_input} \n\n### Response:"
            st.markdown(f"Asking: {sample_query}")

            # before fine-tuning
            completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
            st.markdown(f"Generated: {completion}")

        # Delete the model adapter after generating the response
        new_model_adapter.delete()
geodata = pd.read_csv("/content/ADDRESS ONLY Property_Assessment_Data__Current_Calendar_Year_ - Property_Assessment_Data__Current_.csv")        
def neighbourhood_mapping():
    st.title("Neighbourhood Mapping")

    # Get user input for neighborhood
    user_neighbourhood = st.text_input("Enter the neighborhood:")

    # Check if user provided input
    if user_neighbourhood:
        # Filter the dataset based on the user input
        filtered_data = geodata[geodata['Neighbourhood'] == user_neighbourhood]

        # Check if the filtered data is empty, if so, return a message indicating no data found
        if filtered_data.empty:
            st.write("No data found for the specified neighborhood.")
        else:
            # Create the map using the filtered data
            fig = px.scatter_mapbox(filtered_data,
                                    lat='Latitude',
                                    lon='Longitude',
                                    hover_name='Neighbourhood',
                                    zoom=12)

            # Update map layout to use OpenStreetMap style
            fig.update_layout(mapbox_style='open-street-map')

            # Show the map
            st.plotly_chart(fig)
    else:
        st.write("Please enter a neighborhood to generate the map.")



# Main App Logic
def main():
    st.sidebar.title("Food Drive App")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "Stakeholders","Data Collection","EDA", "ML Modeling", "Neighbourhood Mapping","Chatbot"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page=="Stakeholders":
        stakeholders()
    elif app_page == "EDA":
        exploratory_data_analysis()
    elif app_page == "ML Modeling":
        machine_learning_modeling()
    elif app_page == "Neighbourhood Mapping":
        neighbourhood_mapping()
    elif app_page == "Data Collection":
        data_collection()
    elif app_page == "Chatbot":
        chatbot()

if __name__ == "__main__":
    main()



