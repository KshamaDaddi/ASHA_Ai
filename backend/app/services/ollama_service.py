import ollama

def ask_ai(prompt):

    system_prompt = """
    You are ASHA AI.

    You are an offline healthcare assistant.

    Give simple healthcare guidance.

    Recommend visiting doctor for serious symptoms.
    """

    response = ollama.chat(
        model="phi3",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]