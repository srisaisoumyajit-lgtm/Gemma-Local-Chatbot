from llama_cpp import Llama

MODEL_PATH = r"D:\padhai likhai\Programming\gemma_project\models\gemma-4-E4B-it-Q4_K_M.gguf"

print("Loading Gemma...")

llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=False
)

print("Gemma loaded!")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": user_input
            }
        ],
        temperature=0.7,
        max_tokens=512
    )

    answer = response["choices"][0]["message"]["content"]

    print("Gemma:", answer)
    print()