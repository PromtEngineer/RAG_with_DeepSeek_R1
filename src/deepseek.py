import os
import openai

client = openai.OpenAI(
    api_key="1257bdf0-e39f-4bb8-a0c4-fb7df735a8cb", #os.environ.get("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)

response = client.chat.completions.create(
    model="DeepSeek-R1-Distill-Llama-70B",
    messages=[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"What is the one most unique thing about human experience?"}],
    temperature=0.1,
    top_p=0.1
)

print(response.choices[0].message.content)