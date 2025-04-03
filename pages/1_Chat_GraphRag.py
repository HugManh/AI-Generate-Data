import streamlit as st
import random
import time
import subprocess
import os

st.title("Dino Chatbot")


def construct_cli_args(query, selected_folder="20250401-163203", query_type="global"):
    artifacts_folder = os.path.join(
        "./indexing/output", selected_folder, "artifacts")
    if not os.path.exists(artifacts_folder):
        raise ValueError(f"Artifacts folder not found in {artifacts_folder}")

    base_args = [
        "python", "-m", "graphrag.query",
        "--data", artifacts_folder,
        "--method", query_type,
    ]

    preset = "Default Search"
    if preset.startswith("Default"):
        base_args.extend(
            ["--community_level", "2", "--response_type", "Multiple Paragraphs"])

    base_args.append(query)

    return base_args


def run_graphrag_query(cli_args):
    try:
        command = ' '.join(cli_args)
        print(f"Executing command: {command}")
        result = subprocess.run(
            cli_args, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running GraphRAG query: {e}")
        print(f"Command output (stdout): {e.stdout}")
        print(f"Command output (stderr): {e.stderr}")
        raise RuntimeError(f"GraphRAG query failed: {e.stderr}")


def send_message(query):
    cli_args = construct_cli_args(query)
    result = run_graphrag_query(cli_args)
    return result

# Streamed response emulator


def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Say something"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = send_message(prompt)
    # response = response_generator()
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write(response)
    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response})
