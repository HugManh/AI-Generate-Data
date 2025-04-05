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


def list_output_folders(root_dir):
    output_dir = os.path.join(root_dir, "output")
    print(f"++++++ output_dir {output_dir}")
    folders = [f for f in os.listdir(
        output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    return sorted(folders, reverse=True)


with st.container(height=500):
    selected_folder = st.selectbox(
        "Select Index Folder to Chat With",
        list_output_folders("./indexing"),
    )
    query_type = st.radio(
        "Query Type",
        ["local", "global"],
        horizontal=True,
    )
    preset_dropdown = st.selectbox(
        "Preset Query Options",
        (
            "Default Local Search",
            "Default Global Search",
            "Detailed Global Analysis",
            "Detailed Local Analysis",
            "Quick Global Summary",
            "Quick Local Summary",
            "Global Bullet Points",
            "Local Bullet Points",
            "Comprehensive Global Report",
            "Comprehensive Local Report",
            "High-Level Global Overview",
            "High-Level Local Overview",
            "Focused Global Insight",
            "Focused Local Insight",
            "Custom Query"
        ),
    )
