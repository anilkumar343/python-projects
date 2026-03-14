while True:

    user = input("You: ")

    if user.lower() == "hi":
        print("Bot: Hello!")

    elif user.lower() == "how are you":
        print("Bot: I am fine")

    elif user.lower() == "bye":
        print("Bot: Goodbye")
        break

    else:
        print("Bot: I don't understand")
