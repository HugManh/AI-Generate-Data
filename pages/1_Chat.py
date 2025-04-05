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
import re
from typing import List

#######################
# Page configuration
st.set_page_config(
    page_title="Dino.AI",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")


def translate_auto(text):
    lang = detect(text)
    text = GoogleTranslator(source="auto", target="vi").translate(text)
    return f"[{lang}] {text}"


def get_preset_args(preset: str) -> List[str]:
    preset_args = {
        "Default Global Search": ["--community_level", "2", "--response_type", "Multiple Paragraphs"],
        "Default Local Search": ["--community_level", "2", "--response_type", "Multiple Paragraphs"],
        "Detailed Global Analysis": ["--community_level", "3", "--response_type", "Multi-Page Report"],
        "Detailed Local Analysis": ["--community_level", "3", "--response_type", "Multi-Page Report"],
        "Quick Global Summary": ["--community_level", "1", "--response_type", "Single Paragraph"],
        "Quick Local Summary": ["--community_level", "1", "--response_type", "Single Paragraph"],
        "Global Bullet Points": ["--community_level", "2", "--response_type", "List of 3-7 Points"],
        "Local Bullet Points": ["--community_level", "2", "--response_type", "List of 3-7 Points"],
        "Comprehensive Global Report": ["--community_level", "4", "--response_type", "Multi-Page Report"],
        "Comprehensive Local Report": ["--community_level", "4", "--response_type", "Multi-Page Report"],
        "High-Level Global Overview": ["--community_level", "1", "--response_type", "Single Page"],
        "High-Level Local Overview": ["--community_level", "1", "--response_type", "Single Page"],
        "Focused Global Insight": ["--community_level", "3", "--response_type", "Single Paragraph"],
        "Focused Local Insight": ["--community_level", "3", "--response_type", "Single Paragraph"],
    }
    return preset_args.get(preset, [])


def construct_cli_args(query, selected_folder, query_type, preset):
    artifacts_folder = os.path.join(
        "./indexing/output", selected_folder, "artifacts")
    if not os.path.exists(artifacts_folder):
        raise ValueError(f"Artifacts folder not found in {artifacts_folder}")

    base_args = [
        "python", "-m", "graphrag.query",
        "--data", artifacts_folder,
        "--method", query_type,
    ]

    # Apply preset configurations
    if preset.startswith("Default"):
        base_args.extend(
            ["--community_level", "2", "--response_type", "Multiple Paragraphs"])
    elif preset.startswith("Detailed"):
        base_args.extend(
            ["--community_level", "4", "--response_type", "Multi-Page Report"])
    elif preset.startswith("Quick"):
        base_args.extend(
            ["--community_level", "1", "--response_type", "Single Paragraph"])
    elif preset.startswith("Bullet"):
        base_args.extend(
            ["--community_level", "2", "--response_type", "List of 3-7 Points"])
    elif preset.startswith("Comprehensive"):
        base_args.extend(
            ["--community_level", "5", "--response_type", "Multi-Page Report"])
    elif preset.startswith("High-Level"):
        base_args.extend(
            ["--community_level", "1", "--response_type", "Single Page"])
    elif preset.startswith("Focused"):
        base_args.extend(
            ["--community_level", "3", "--response_type", "Multiple Paragraphs"])
    # elif preset == "Custom Query":
    #     base_args.extend([
    #         "--community_level", str(community_level),
    #         "--response_type", f'"{response_type}"',
    #     ])
    #     if custom_cli_args:
    #         base_args.extend(custom_cli_args.split())

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


# def send_message(query):
#     cli_args = construct_cli_args(
#         query, selected_folder, query_type, preset_dropdown)
#     result = run_graphrag_query(cli_args)
#     return result

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


def delete_chat():
    st.session_state.messages = []
    print("Chat history cleared!")


# st.title("Dino Chatbot")
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

tab1, tab2 = st.tabs(["Conversations", "Dog"])

# Container for chat history
chat_container = st.container()
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
            # anwser = send_message(prompt)
            anwser = response_generator()
            assistant.write(anwser)
    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": anwser})
