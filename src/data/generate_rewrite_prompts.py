import os
from openai import OpenAI
import json

def get_chat_completion(client):
    return client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "You are a highly creative prompt engineer and you are going to generate a rewriting prompt that indicates a style, tone, mood, character, era, ... for rewriting a passage. Try being as specific as you can.",
            }
        ],
        model="gpt-3.5-turbo",
    )

def main():
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    responses = []
    for _ in range(100):
        response = get_chat_completion(client)
        responses.append(response.choices[0].message.content)
    
    with open('data/raw/responses.json', 'a') as f:
        json.dump(responses, f)

if __name__ == "__main__":
    main()
