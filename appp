import streamlit as st
import anthropic
import base64
import os
import httpx

# Load environment variables
CLAUDE_API_KEY = st.secrets["CLAUDE_API_KEY"]

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

st.title("Claude Image Token Counter")
st.write("Upload an image to see how many input tokens Claude 3.5 Haiku uses.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image preview
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Convert to base64
    image_bytes = uploaded_file.read()
    image_data = base64.standard_b64encode(image_bytes).decode("utf-8")

    # Detect MIME type
    mime_type = "image/jpeg"
    if uploaded_file.type == "image/png":
        mime_type = "image/png"

    # Call Claude token counter
    try:
        response = client.messages.count_tokens(
            model="claude-3-5-haiku-20241022",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe this image"
                        }
                    ],
                }
            ],
        )

        st.success(f"Estimated input tokens: {response.input_tokens}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
