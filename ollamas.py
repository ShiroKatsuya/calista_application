import ollama

def ollama_full_text(prompt, model="gemma3:1b"):
    """
    Get a short text completion (limited to 14 tokens) from Ollama.

    Args:
        prompt (str): The prompt to send to Ollama.
        model (str): The model name to use (default: "gemma3:1b").

    Returns:
        str: The short generated text.
    """

    context = (
        "Answer the following question in a way that's friendly and relatable, not too formal. "
        "Keep it short and informative, but make sure your response feels genuine and connected to the user."
    )
    
    simple = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": (
                f"{context}\n\n{prompt}\n\n"
                "Please give a concise, helpful answer that feels natural and engaging."
            )
        }],
        options={"num_predict": 40}
    )
    return simple["message"]["content"]

# from example import text_to_speech




# # Print the full response
# response = ollama_full_text(prompt="hello what are you doing now?")
# print(response)

# # Fix: Clean the response more robustly
# import re
# # Remove asterisks and excessive newlines, and strip leading/trailing whitespace
# cleaned_response = re.sub(r'\*+', '', response)
# cleaned_response = re.sub(r'\n\s*\n', '\n', cleaned_response)
# cleaned_response = cleaned_response.strip()

# output_file = "output.wav"
# text_to_speech(cleaned_response, voice="tara", output_file=output_file)

