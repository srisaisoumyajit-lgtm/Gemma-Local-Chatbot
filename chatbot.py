from llama_cpp import Llama

# --------------------------------------------------
# Model Configuration
# --------------------------------------------------

MODEL_PATH = r"D:\padhai likhai\Programming\gemma_project\models\gemma-4-E4B-it-Q4_K_M.gguf"

print("\nLoading Gemma... Please wait.\n")

llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=False
)

print("Gemma loaded successfully!")
print("Type /help to see available commands.\n")


# --------------------------------------------------
# System Prompt + Conversation Memory
# --------------------------------------------------

SYSTEM_PROMPT = """
You are Gemma, a helpful, friendly, and intelligent AI assistant.

Answer questions clearly and accurately.
When explaining programming concepts, provide examples when useful.
Keep responses understandable and well structured.
"""

conversation = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]


# --------------------------------------------------
# Chat Commands
# --------------------------------------------------

def show_help():
    print("""
Available Commands:

/help   - Show available commands
/clear  - Clear conversation memory
/exit   - Exit the chatbot
""")


# --------------------------------------------------
# Main Chat Loop
# --------------------------------------------------

while True:

    try:
        user_input = input("You: ").strip()

        # Ignore empty messages
        if not user_input:
            continue

        # Exit chatbot
        if user_input.lower() in ["/exit", "exit"]:
            print("\nGoodbye! 👋\n")
            break

        # Show help
        if user_input.lower() == "/help":
            show_help()
            continue

        # Clear conversation memory
        if user_input.lower() == "/clear":

            conversation = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }
            ]

            print("\nConversation memory cleared!\n")
            continue

        # Add user message to conversation
        conversation.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        print("\nGemma: ", end="", flush=True)

        # Generate response
        response = llm.create_chat_completion(
            messages=conversation,
            temperature=0.7,
            max_tokens=512
        )

        answer = response["choices"][0]["message"]["content"]

        # Print response
        print(answer)
        print()

        # Save assistant response to conversation memory
        conversation.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋\n")
        break

    except Exception as error:
        print(f"\nError: {error}\n")