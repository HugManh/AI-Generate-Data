import streamlit as st
import random
import time
import subprocess
import os
from langdetect import detect
from deep_translator import GoogleTranslator
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# import altair as alt

#######################
# Page configuration
st.set_page_config(
    page_title="US Population Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

# alt.themes.enable("quartz")


def translate_auto(text):
    lang = detect(text)
    text = GoogleTranslator(source="auto", target="vi").translate(text)
    return f"[{lang}] {text}"


def list_output_folders(root_dir):
    output_dir = os.path.join(root_dir, "output")
    print(f"++++++ output_dir {output_dir}")
    folders = [f for f in os.listdir(
        output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    return sorted(folders, reverse=True)


def construct_cli_args(query, selected_folder, query_type):
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
    cli_args = construct_cli_args(query, selected_folder, query_type)
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


st.title("Dino Chatbot")
chatbot, options = st.columns([3, 1])


with options.container(height=600):
    selected_folder = st.selectbox(
        "Select Index Folder to Chat With",
        list_output_folders("./indexing"),
    )
    query_type = st.radio(
        "Query Type",
        ["global", "local"],
        horizontal=True,
    )

with chatbot:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Container for chat history
    chat_container = st.container(height=500)
    with chat_container:
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Say something"):
        # Display user message in chat message container
        user = chat_container.chat_message("user")
        user.write(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        assistant = chat_container.chat_message("assistant")
        with assistant:
            with st.spinner("Wait for it...", show_time=True):
                anwser = send_message(prompt)
                assistant.write(anwser)
        # response = response_generator()
        # Display assistant response in chat message container
        # response = st.write(translate_auto(anwser))
        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": anwser})
