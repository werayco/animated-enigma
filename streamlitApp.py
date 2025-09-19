import base64
import streamlit as st
from anthropic import Anthropic
import google.generativeai as genai
import os
import json
import tempfile
from PIL import Image
import re
from concurrent.futures import ThreadPoolExecutor

CLAUDE_API_KEY = st.secrets["CLAUDE_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = Anthropic(api_key=CLAUDE_API_KEY)

client = Anthropic(api_key=CLAUDE_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")


class ModelTest:
    @staticmethod
    def claudeModel(image_path, modelName):
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        message = client.messages.create(
            model=modelName, 
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": """Analyze the food in this image and provide a detailed JSON response.
                Identify the following:
                1.  food_items: A list of all food items present.
                2.  estimated_quantities: Estimated portion size for each food item (e.g., "one serving," "a slice," "a cup").
                3.  cooking_method: The apparent cooking method (e.g., "fried," "boiled," "baked"). If not clear, state "not apparent."
                4.  cuisine_type: The likely cuisine (e.g., "Italian," "Mexican," "Indian"). If not clear, state "not apparent."
                5.  confidence_score: A high-level confidence score (from 0.0 to 1.0) for the overall analysis.
                
                Format the response as a single, valid JSON object.
                ```json
        {
        "food_items": [
            {
            "item": "",
            "estimated_quantities": "",
            "cooking_method": ",
            "cuisine_type": ""
            },...
        ],
        "overall_cuisine_type": "",
        "overall_cooking_method": "",
        "confidence_score": 
        }
        ```
                """,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_base64,
                            },
                        }
                    ],
                },
            ],
        )

        parsedResponse = ModelTest.parse_response1(message.content[0].text)
        return parsedResponse

    @staticmethod
    def detect_food_with_gemini(image_path):
        if not os.path.exists(image_path):
            return json.dumps({"error": f"Error: Image file not found at {image_path}"})

        try:
            image_part = Image.open(image_path)
        except Exception as e:
            return json.dumps({"error": f"Could not open image file: {e}"})

        prompt_parts = [
            image_part,
            """Analyze the food in this image and provide a detailed JSON response.
            Identify the following:
            1.  food_items: A list of all food items present.
            2.  estimated_quantities: Estimated portion size for each food item (e.g., "one serving," "a slice," "a cup").
            3.  cooking_method: The apparent cooking method (e.g., "fried," "boiled," "baked"). If not clear, state "not apparent."
            4.  cuisine_type: The likely cuisine (e.g., "Italian," "Mexican," "Indian"). If not clear, state "not apparent."
            5.  confidence_score: A high-level confidence score (from 0.0 to 1.0) for the overall analysis.
            
            Format the response as a single, valid JSON object.
                ```json
            {
            "food_items": [
                {
                "item": "",
                "estimated_quantities": "",
                "cooking_method": ",
                "cuisine_type": ""
                },...
            ],
            "overall_cuisine_type": "",
            "overall_cooking_method": "",
            "confidence_score": 
            }
            ```""",
        ]
        try:
            response = model.generate_content(prompt_parts)
            parsedResponse = ModelTest.parse_response1(response.text)
            return parsedResponse
        except Exception as e:
            return json.dumps({"error": f"An API error occurred: {e}"})
        
    @staticmethod
    def parse_response1(text):
        try:
            text = text.strip()
            if text.startswith("{") and text.endswith("}"):
                return json.loads(text)
        except:
            pass
        try:
            pattern = r"```(?:json)?\s*(.*?)```"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return json.loads(match.group(1).strip())
            pattern = r"(\{.*\})"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return json.loads(match.group(1).strip())

            return {"email_category": "unknown", "confidence_level": 0}
        except Exception as e:
            return {"email_category": "unknown", "confidence_level": 0}
        
    @staticmethod
    def threadPoolExecutorModels(image_path, modelName):
        with ThreadPoolExecutor(max_workers=3) as workers:
            claudeFuture = workers.submit(ModelTest.claudeModel, image_path, modelName)
            geminiFuture = workers.submit(ModelTest.detect_food_with_gemini, image_path)

        claudeResult: dict = claudeFuture.result()
        geminiResult: dict = geminiFuture.result()
        return claudeResult, geminiResult
import pandas as pd

st.title("AI Food Analyzer")
st.markdown("Upload a food image and get detailed analysis from Claude and Gemini models")

image = st.file_uploader(
    "Choose a food image...", 
    type=['png', 'jpg', 'jpeg'],
    help="Upload an image of food for AI analysis"
)

claudeModelName = st.selectbox(
    "Choose your Claude model",
    ["claude-3-haiku-20240307", "claude-3-5-haiku-20241022"],
    index=0,
    help="Pick between Claude 3 Haiku (2024-03-07) and Claude 3.5 Haiku (2024-10-22)"
)

if image is not None and st.button("Analyze Food", type="primary"):
    st.subheader("Analysis Results")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        img = Image.open(image)
        img.save(tmp_file.name)
        temp_path = tmp_file.name
        
        with st.spinner("Analyzing image..."):
            claude_result, gemini_result = ModelTest.threadPoolExecutorModels(temp_path, claudeModelName)
    
    os.unlink(temp_path)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Claude Results")
        if "error" in claude_result:
            st.error(f"Error: {claude_result['error']}")
        else:
            claude_data = []
            claude_data.append(["Overall Cuisine", claude_result.get("overall_cuisine_type", "N/A")])
            claude_data.append(["Overall Cooking Method", claude_result.get("overall_cooking_method", "N/A")])
            claude_data.append(["Confidence Score", f"{claude_result.get('confidence_score', 0):.2f}"])
            
            # Add food items
            if "food_items" in claude_result:
                for i, item in enumerate(claude_result["food_items"], 1):
                    claude_data.append([f"Food Item {i}", item.get("item", "N/A")])
                    claude_data.append([f"Quantity {i}", item.get("estimated_quantities", "N/A")])
            
            claude_df = pd.DataFrame(claude_data, columns=["Attribute", "Value"])
            st.dataframe(claude_df, hide_index=True)
    
    with col2:
        st.subheader("Gemini Results")
        if "error" in gemini_result:
            st.error(f"Error: {gemini_result['error']}")
        else:
            gemini_data = []
            gemini_data.append(["Overall Cuisine", gemini_result.get("overall_cuisine_type", "N/A")])
            gemini_data.append(["Overall Cooking Method", gemini_result.get("overall_cooking_method", "N/A")])
            gemini_data.append(["Confidence Score", f"{gemini_result.get('confidence_score', 0):.2f}"])
            
            if "food_items" in gemini_result:
                for i, item in enumerate(gemini_result["food_items"], 1):
                    gemini_data.append([f"Food Item {i}", item.get("item", "N/A")])
                    gemini_data.append([f"Quantity {i}", item.get("estimated_quantities", "N/A")])
            
            gemini_df = pd.DataFrame(gemini_data, columns=["Attribute", "Value"])
            st.dataframe(gemini_df, hide_index=True)
    
    st.subheader("Side-by-Side Comparison")
    
    if "error" not in claude_result and "error" not in gemini_result:
        comparison_data = []
        
        comparison_data.append([
            "Overall Cuisine",
            claude_result.get("overall_cuisine_type", "N/A"),
            gemini_result.get("overall_cuisine_type", "N/A")
        ])
        comparison_data.append([
            "Overall Cooking Method",
            claude_result.get("overall_cooking_method", "N/A"),
            gemini_result.get("overall_cooking_method", "N/A")
        ])
        comparison_data.append([
            f"{claude_result.get('confidence_score', 0):.2f}",
            f"{gemini_result.get('confidence_score', 0):.2f}"
        ])
        
        claude_items = claude_result.get("food_items", [])
        gemini_items = gemini_result.get("food_items", [])
        max_items = max(len(claude_items), len(gemini_items))
        
        for i in range(max_items):
            claude_item = claude_items[i] if i < len(claude_items) else {}
            gemini_item = gemini_items[i] if i < len(gemini_items) else {}
            
            comparison_data.append([
                f"Food Item {i+1}",
                claude_item.get("item", "N/A"),
                gemini_item.get("item", "N/A")
            ])
            comparison_data.append([
                f"Quantity {i+1}",
                claude_item.get("estimated_quantities", "N/A"),
                gemini_item.get("estimated_quantities", "N/A")
            ])
        
        comparison_df = pd.DataFrame(comparison_data, columns=["Attribute", "Claude", "Gemini"])
        st.dataframe(comparison_df, hide_index=True)
