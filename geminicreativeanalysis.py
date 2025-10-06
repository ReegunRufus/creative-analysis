import os
import json
import requests
from io import BytesIO
from google import genai
from google.genai import types

# Download image as bytes
def download_image_as_bytes(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"[ERROR] Error downloading image: {e}")
        return None

def lambda_handler(event, context):
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Content-Type": "application/json"
    }

    try:
        print("[DEBUG] Incoming event:", json.dumps(event))

        # Handle CORS preflight
        if event.get("httpMethod") == "OPTIONS":
            return {"statusCode": 200, "headers": headers, "body": json.dumps({"message": "CORS preflight OK"})}

        # Initialize Gemini client
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise Exception("Missing GEMINI_API_KEY environment variable")
        client = genai.Client(api_key=api_key)
        print("[DEBUG] genai.Client initialized successfully")

        # Parse body
        body = event.get("body")
        if body and isinstance(body, str):
            body = json.loads(body)
        elif not body:
            body = event

        image_url = body.get("image_url")
        brief = body.get("brief")
        brand_name = body.get("brand_name")
        brand_description = body.get("brand_description")

        if not image_url or not brief or not brand_name or not brand_description:
            return {
                "statusCode": 400,
                "headers": headers,
                "body": json.dumps({
                    "error": "Missing required fields: image_url, brief, brand_name, brand_description"
                })
            }

        # Handle image: URL or base64
        if image_url.startswith("data:image/"):
            base64_data = image_url.split(",")[1]
            image_bytes = BytesIO(b64decode(base64_data)).read()
        else:
            image_bytes = download_image_as_bytes(image_url)
            if not image_bytes:
                return {"statusCode": 500, "headers": headers, "body": json.dumps({"error": "Failed to download image"})}

        # Wrap image in types.Part for Gemini
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        print("[DEBUG] types.Part created successfully")

        # Build prompts
        system_prompt = (
            "You are a creative strategist with deep expertise in performance marketing creatives. "
            "You are from India and understand the consumer personas and market dynamics in India. "
            "Evaluate the attached creative from a performance marketing lens. "
            "Provide: an overall score, copy analysis, visual analysis, targeting/persona insights, "
            "performance prediction, and improvement areas. "
            "Keep it concise, in bullet points, and highlight key takeaways in **bold**."
        )
        user_prompt = f"Brief: {brief}\nBrand Name: {brand_name}\nBrand Description: {brand_description}"

        # Call Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[system_prompt, user_prompt, image_part]
        )
        print("[DEBUG] Response received from Gemini")

        summary = response.text.strip()
        if summary.lower() in ["ok", "fine", "good", "looks fine"]:
            summary = "The image looks aligned and acceptable."

        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps({
                "summary": summary,
                "brand": brand_name,
                "brief": brief
            })
        }

    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        return {"statusCode": 500, "headers": headers, "body": json.dumps({"error": str(e)})}
