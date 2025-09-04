import streamlit as st
import os
from ibm_watson_machine_learning.foundation_models import infer
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

# You'll need to set your API key and project ID as environment variables
# Alternatively, you can hardcode them for a simple demo, but this is not recommended for production.
# os.environ['IBM_API_KEY'] = 'YOUR_API_KEY'
# os.environ['PROJECT_ID'] = 'YOUR_PROJECT_ID'

# Set up IBM Watsonx.ai client
def get_model_inference_instance(model_id):
    project_id = os.getenv("PROJECT_ID")
    if not project_id:
        st.error("Please set the PROJECT_ID environment variable.")
        st.stop()
        
    model_params = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MAX_NEW_TOKENS: 500,
        GenParams.TEMPERATURE: 0.1,
    }
    
    # Use the Granite model as specified
    model = infer.ModelInference(
        model_id=model_id,
        params=model_params,
        project_id=project_id,
        credentials={
            "api_key": os.getenv("IBM_API_KEY"),
            "url": "https://us-south.ml.cloud.ibm.com",
        }
    )
    return model

# Function to generate a response from the Granite model
def generate_response(prompt, model_instance, user_demographic):
    # Adjust tone and complexity based on demographic
    if user_demographic == "Student":
        tone_modifier = "in simple terms, using analogies that a student would understand."
    else:
        tone_modifier = "with a professional tone, providing detailed, actionable insights."

    # Construct the full prompt
    full_prompt = f"You are a personal finance expert. {tone_modifier} Please provide a detailed financial guidance for the following query: {prompt}"

    # Generate a response using the Granite model
    result = model_instance.generate_text(prompt=full_prompt)
    return result

# Streamlit UI
st.title("Personal Finance Chatbot 💰")

# Sidebar for demographic selection
with st.sidebar:
    st.subheader("User Profile")
    demographic = st.radio("I am a:", ("Student", "Professional"))

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your finances..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Use the Granite 13B model
            granite_model = get_model_inference_instance(model_id=ModelTypes.GRANITE_13B_CHAT_V2)
            response = generate_response(prompt, granite_model, demographic)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})