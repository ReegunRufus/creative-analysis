import os
import json
import requests
from io import BytesIO
from base64 import b64encode, b64decode
from openai import OpenAI

# Convert BytesIO to base64 URI
def image_bytes_to_base64_uri(image_bytes_io):
    base64_str = b64encode(image_bytes_io.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_str}"

def download_image_as_bytes(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return BytesIO(response.content)
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
        if event.get("httpMethod") == "OPTIONS":
            return {"statusCode": 200, "headers": headers, "body": json.dumps({"message": "CORS preflight OK"})}

        # Initialize OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise Exception("Missing OPENAI_API_KEY environment variable")
        client = OpenAI(api_key=api_key)

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
            return {"statusCode": 400, "headers": headers, "body": json.dumps({"error": "Missing required fields"})}

        # Handle image
        if image_url.startswith("data:image/"):
            image_b64_uri = image_url
        else:
            image_bytes = download_image_as_bytes(image_url)
            if not image_bytes:
                return {"statusCode": 500, "headers": headers, "body": json.dumps({"error": "Failed to download image"})}
            image_b64_uri = image_bytes_to_base64_uri(image_bytes)

        # Prompts
        system_prompt = (
            "You are a creative strategist with deep expertise in performance marketing creatives. "
            "You are from India and understand the consumer personas and market dynamics in India. "
            "Evaluate the attached creative from a performance marketing lens. "
            "Provide: an overall score, copy analysis, visual analysis, targeting/persona insights, "
            "performance prediction, and improvement areas. "
            "Keep it concise, in bullet points, and highlight key takeaways in **bold**."
        )

        user_prompt = f"Brief: {brief}\nBrand Name: {brand_name}\nBrand Description: {brand_description}"

        # Call OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_b64_uri}}
                ]}
            ],
            max_tokens=400,
            temperature=0.7
        )

        summary = response.choices[0].message.content.strip()

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
        return {"statusCode": 500, "headers": headers, "body": json.dumps({"error": str(e)})}
