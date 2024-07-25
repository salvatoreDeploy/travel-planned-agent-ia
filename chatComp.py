from openai import OpenAI # type: ignore

client = OpenAI()

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpfull assistant."},
    {"role": "user", "content": "Vou participar da CCXP em Dezembro de 2024. Quais paineis posso ver?"}
  ]
)

print(response.choices[0].message.content)