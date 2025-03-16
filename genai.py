import google.generativeai as genai

def chat_with_gemini(user_input, api_token):
    genai.configure(api_key=api_token)
    allowed_topics = [
        "T-shirts", "fashion", "clothing", "outfits", "style", "fabric", "brands", "size",
        "colors", "recommended T-shirts", "website features", "shopping", "scraped T-shirts"
    ]

    if user_input.lower() in ["hi", "hello"]:
        return "Welcome to our Website. Feel free to ask anything about fashion or this website."

    if any(topic in user_input.lower() for topic in allowed_topics):
        model = genai.GenerativeModel("gemini-2.0-flash-001")
        response = model.generate_content(user_input)
        return response.text if response else "Sorry, I couldn't process that."

    return "Sorry, I couldn't process that. Please ask about fashion or this website."