import chainlit as cl
from rag_pipeline import ask_question


@cl.on_chat_start
async def start():
    cl.user_session.set("chat_history", [])

    await cl.Message(
        content=(
            "Welcome to the iPhone User Guide Assistant!\n\n"
            "Ask me anything about the iPhone User Guide."
        )
    ).send()


@cl.on_message
async def main(message: cl.Message):
    question = message.content

    chat_history = cl.user_session.get("chat_history", [])

    try:
        answer, sources = ask_question(question, chat_history)

        if sources:
            pages = ", ".join(str(page) for page in sources)
            answer += f"\n\n**Source pages:** {pages}"

        await cl.Message(content=answer).send()

        chat_history.append(("user", question))
        chat_history.append(("assistant", answer))

        cl.user_session.set("chat_history", chat_history)

    except Exception as e:
        await cl.Message(
            content=f"An error occurred: {str(e)}"
        ).send()