from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI()
response=client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role" :"system","content":"You are a helpful health assistant"},
        {"role": "user", "content":"what are the symptoms of type 2 diabetes"}
    ]
)

print(response.choices[0].message.content)