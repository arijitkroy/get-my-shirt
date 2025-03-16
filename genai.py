from streamlit import secrets
import google.generativeai as genai

gen_conf = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(model_name=secrets["MODEL_ID"], generation_config=gen_conf)

def chat_with_gemini_stream(user_input, api_token):
    genai.configure(api_key=api_token)

    if user_input.lower() in ["hi", "hello"]:
        yield "Welcome to our Website. Feel free to ask anything about fashion or this website."
        return

    if user_input:
        response = model.generate_content(user_input, stream=True)
        for chunk in response:
            yield chunk.text
        return

    yield "Sorry, I couldn't process that. Please ask about fashion or this website."