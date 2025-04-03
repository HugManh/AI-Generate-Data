import gradio as gr
import polars as pl
from typing import List, Tuple, Generator, Union
import uuid
import random
import math
from pathlib import Path
from tqdm import tqdm
from distilabel.steps.tasks import (
    TextGeneration,
    ChatGeneration,
    Magpie,
)
from datasets import Dataset
from distilabel.distiset import Distiset
from distilabel.models import InferenceEndpointsLLM
import os

DEFAULT_DATASET_DESCRIPTIONS = [
    "trợ lý hỗ trợ luật đất đai",
    "trợ lý giải đáp thắc mắc về vay tín dụng",
]

# Xử lý nhập liệu thủ công


def handle_manual_input(manual_input):
    return f"Dữ liệu đã nhập: {manual_input}"

# Xử lý upload file CSV bằng Polars


def load_dataset_file(
    file_paths: list[str],
    num_rows: int = None,
):
    dataframe, _ = preprocess_input_data(
        file_paths=file_paths, num_rows=num_rows)
    return dataframe


def preprocess_input_data(
    file_paths: list[str], num_rows: int = None, progress=gr.Progress(track_tqdm=True)
) -> Tuple[pl.DataFrame, gr.Dropdown]:
    """
    Preprocess list of file paths into a DataFrame and Dropdown widget.

    Args:
        file_paths: List of file paths to process
        num_rows: Maximum number of rows to process (optional)
        progress: Gradio progress tracker

    Returns:
        Tuple of (Polars DataFrame, Gradio Dropdown)
    """
    print(f"Error processing  {file_paths}")
    if not file_paths:
        raise gr.Error("Vui lòng cung cấp ít nhất 1 file")

    # Thu thập dữ liệu từ generator
    data_records = []

    for file_path in tqdm(file_paths, desc="Processing files", total=len(file_paths)):
        # Kiểm tra phần mở rộng của file
        file_extension = Path(file_path).suffix.lower()
        try:
            # Đọc file CSV
            if file_extension == ".csv":
                df = pl.read_csv(file_path, n_rows=num_rows)
            elif file_extension == ".xlsx":
                df = pl.read_excel(file_path, sheet_id=0)
            else:
                raise gr.Error(
                    f"Định dạng file không hỗ trợ: {file_extension}. Chỉ hỗ trợ .csv và .xlsx")
            print(f"Type of df after read: {type(df)}")
            print(f"Content of df: {df}")
            # Thêm dữ liệu vào danh sách
            data_records.extend(df.to_dicts())
        except Exception as e:
            raise gr.Error(f"Lỗi khi đọc file {file_path}: {str(e)}")

    # Tạo DataFrame từ danh sách dữ liệu
    dataframe = pl.DataFrame(data_records)

    # Tạo dropdown widget
    col_doc = dataframe.columns[0]
    dropdown = gr.Dropdown(
        choices=dataframe.columns,
        label="Documents column",
        value=col_doc,
        # Hiện tại chỉ có 1 cột, nên không cần interactive
        interactive=(False if col_doc == "" else True),
        multiselect=False,
    ),

    return (dataframe, dropdown)


# Inference
MAX_NUM_TOKENS = int(os.getenv("MAX_NUM_TOKENS", 2048))
MAX_NUM_ROWS = int(os.getenv("MAX_NUM_ROWS", 1000))
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", 5))


def test_max_num_rows(num_rows: int) -> int:
    if num_rows > MAX_NUM_ROWS:
        num_rows = MAX_NUM_ROWS
        gr.Info(
            f"Number of rows is larger than the configured maximum. Setting number of rows to {MAX_NUM_ROWS}. Set environment variable `MAX_NUM_ROWS` to change this behavior."
        )
    return num_rows


MAGPIE_PRE_QUERY_TEMPLATE = "qwen2"
_STOP_SEQUENCES = ["<|im_end|>", "<|im_start|>", "assistant", "\n\n"]


def _get_output_mappings(num_turns: int):
    if num_turns == 1:
        return {"instruction": "prompt", "response": "completion"}
    else:
        return {"conversation": "messages"}


def get_magpie_generator(num_turns: int, temperature: float, is_sample: bool):
    """
    Tạo generator cho Magpie

    Parameters:
        num_turns: Số lượt tương tác của Magpie. Giá trị này quyết định số lần Magpie sẽ phản hồi.
        temperature: Nhiệt độ của mô hình (temperature) để điều chỉnh mức độ ngẫu nhiên khi sinh đầu ra.
            `Giá trị thấp (ví dụ: 0.2) -> ít ngẫu nhiên hơn, đầu ra mang tính dự đoán cao.
            Giá trị cao (ví dụ: 1.0) -> nhiều ngẫu nhiên hơn, đầu ra đa dạng hơn.`
        is_sample: Cờ (flag) xác định kiểu lấy mẫu khi sinh đầu ra:
            `True: Sử dụng sampling (ngẫu nhiên hơn).
            False: Sử dụng greedy decoding (ít ngẫu nhiên hơn, tối ưu xác suất cao nhất).`

    Returns:
        generator
    """
    input_mappings = _get_output_mappings(num_turns)
    output_mappings = input_mappings.copy()
    if num_turns == 1:
        generation_kwargs = {
            "temperature": temperature,
            "do_sample": True,
            "max_new_tokens": 256 if is_sample else int(MAX_NUM_TOKENS * 0.25),
            "stop_sequences": _STOP_SEQUENCES,
        }
        magpie_generator = Magpie(
            llm=_get_llm(
                generation_kwargs=generation_kwargs,
                magpie_pre_query_template=MAGPIE_PRE_QUERY_TEMPLATE,
                use_magpie_template=True,
            ),
            n_turns=num_turns,
            output_mappings=output_mappings,
            only_instruction=True,
        )
    else:
        generation_kwargs = {
            "temperature": temperature,
            "do_sample": True,
            "max_new_tokens": 256 if is_sample else int(MAX_NUM_TOKENS * 0.5),
            "stop_sequences": _STOP_SEQUENCES,
        }
        magpie_generator = Magpie(
            llm=_get_llm(
                generation_kwargs=generation_kwargs,
                magpie_pre_query_template=MAGPIE_PRE_QUERY_TEMPLATE,
                use_magpie_template=True,
            ),
            end_with_user=True,
            n_turns=num_turns,
            output_mappings=output_mappings,
        )
    magpie_generator.load()
    return magpie_generator


def get_response_generator(
    system_prompt: str, num_turns: int, temperature: float, is_sample: bool
):
    if num_turns == 1:
        generation_kwargs = {
            "temperature": temperature,
            "max_new_tokens": 256 if is_sample else int(MAX_NUM_TOKENS * 0.5),
        }
        response_generator = TextGeneration(
            llm=_get_llm(is_completion=True,
                         generation_kwargs=generation_kwargs),
            system_prompt=system_prompt,
            output_mappings={"generation": "completion"},
            input_mappings={"instruction": "prompt"},
        )
    else:
        generation_kwargs = {
            "temperature": temperature,
            "max_new_tokens": MAX_NUM_TOKENS,
        }
        response_generator = ChatGeneration(
            llm=_get_llm(is_completion=True,
                         generation_kwargs=generation_kwargs),
            output_mappings={"generation": "completion"},
            input_mappings={"conversation": "messages"},
        )
    response_generator.load()
    return response_generator


def _get_prompt_rewriter():
    generation_kwargs = {
        "temperature": 1,
    }
    system_prompt = "You are a prompt rewriter. You are given a prompt and you need to rewrite it keeping the same structure but highlighting different aspects of the original without adding anything new."
    prompt_rewriter = TextGeneration(
        llm=_get_llm(generation_kwargs=generation_kwargs),
        system_prompt=system_prompt,
        use_system_prompt=True,
    )
    prompt_rewriter.load()
    return prompt_rewriter


def get_rewritten_prompts(prompt: str, num_rows: int):
    prompt_rewriter = _get_prompt_rewriter()
    # create prompt rewrites
    inputs = [
        {"instruction": f"Original prompt: {prompt} \nRewritten prompt: "}
        for i in range(math.floor(num_rows / 100))
    ]
    n_processed = 0
    prompt_rewrites = [prompt]
    while n_processed < num_rows:
        batch = list(
            prompt_rewriter.process(
                inputs=inputs[n_processed: n_processed + DEFAULT_BATCH_SIZE]
            )
        )
        prompt_rewrites += [entry["generation"] for entry in batch[0]]
        n_processed += DEFAULT_BATCH_SIZE
        random.seed(a=random.randint(0, 2**32 - 1))
    return prompt_rewrites


def generate_dataset_from_prompt(
    system_prompt: str,
    num_turns: int = 1,  # (prompt-completion)
    num_rows: int = 10,
    temperature: float = 0.9,
    temperature_completion: Union[float, None] = None,
    is_sample: bool = False,
    progress=gr.Progress(),
) -> pl.DataFrame:
    num_rows = test_max_num_rows(num_rows)
    progress(0.0, desc="(1/2) Generating instructions")
    magpie_generator = get_magpie_generator(num_turns, temperature, is_sample)
    response_generator = get_response_generator(
        system_prompt=system_prompt,
        num_turns=num_turns,
        temperature=temperature or temperature_completion,
        is_sample=is_sample,
    )
    total_steps: int = num_rows * 2
    batch_size = DEFAULT_BATCH_SIZE

    # create prompt rewrites
    # Viết lại promt tạo sự đa dạng cho input mô hình ngôn ngữ
    prompt_rewrites = get_rewritten_prompts(system_prompt, num_rows)

    # create instructions
    n_processed = 0
    magpie_results = []
    while n_processed < num_rows:
        progress(
            0.5 * n_processed / num_rows,
            total=total_steps,
            desc="(1/2) Generating instructions",
        )
        remaining_rows = num_rows - n_processed
        batch_size = min(batch_size, remaining_rows)
        rewritten_system_prompt = random.choice(prompt_rewrites)
        inputs = [{"system_prompt": rewritten_system_prompt}
                  for _ in range(batch_size)]
        batch = list(magpie_generator.process(inputs=inputs))
        magpie_results.extend(batch[0])
        n_processed += batch_size
        random.seed(a=random.randint(0, 2**32 - 1))
    progress(0.5, desc="(1/2) Generating instructions")

    # generate responses
    n_processed = 0
    response_results = []
    if num_turns == 1:
        while n_processed < num_rows:
            progress(
                0.5 + 0.5 * n_processed / num_rows,
                total=total_steps,
                desc="(2/2) Generating responses",
            )
            batch = magpie_results[n_processed: n_processed + batch_size]
            responses = list(response_generator.process(inputs=batch))
            response_results.extend(responses[0])
            n_processed += batch_size
            random.seed(a=random.randint(0, 2**32 - 1))
        for result in response_results:
            result["prompt"] = result["instruction"]
            result["completion"] = result["generation"]
            result["system_prompt"] = system_prompt
    else:
        for result in magpie_results:
            result["conversation"].insert(
                0, {"role": "system", "content": system_prompt}
            )
            result["messages"] = result["conversation"]
        while n_processed < num_rows:
            progress(
                0.5 + 0.5 * n_processed / num_rows,
                total=total_steps,
                desc="(2/2) Generating responses",
            )
            batch = magpie_results[n_processed: n_processed + batch_size]
            responses = list(response_generator.process(inputs=batch))
            response_results.extend(responses[0])
            n_processed += batch_size
            random.seed(a=random.randint(0, 2**32 - 1))
        for result in response_results:
            result["messages"].append(
                {"role": "assistant", "content": result["generation"]}
            )
    progress(
        1,
        total=total_steps,
        desc="(2/2) Creating dataset",
    )

    # create distiset
    distiset_results = []
    for result in response_results:
        record = {}
        for relevant_keys in [
            "messages",
            "prompt",
            "completion",
            "model_name",
            "system_prompt",
        ]:
            if relevant_keys in result:
                record[relevant_keys] = result[relevant_keys]
        distiset_results.append(record)

    distiset = Distiset(
        {
            "default": Dataset.from_list(distiset_results),
        }
    )

    # If not pushing to hub generate the dataset directly
    distiset = distiset["default"]
    if num_turns == 1:
        outputs = distiset.to_pandas(
        )[["prompt", "completion", "system_prompt"]]
    else:
        outputs = distiset.to_pandas()[["messages"]]
    dataframe = pl.DataFrame(outputs)
    progress(1.0, desc="Dataset generation completed")
    return dataframe


def generate_dataset(
    dataframe: pl.DataFrame,
    system_prompt: str,
    document_column: str,
    num_turns: int = 1,
    num_rows: int = 10,
    temperature: float = 0.9,
    temperature_completion: Union[float, None] = None,
    is_sample: bool = False,
    progress=gr.Progress(),
) -> pl.DataFrame:
    dataframe = generate_dataset_from_prompt(
        system_prompt=system_prompt,
        num_turns=num_turns,
        num_rows=num_rows,
        temperature=temperature,
        temperature_completion=temperature_completion,
        is_sample=is_sample,
    )
    return dataframe


# Tạo ra dataset mẫu


def generate_sample_dataset(
    system_prompt: str,
    document_column: str,
    progress=gr.Progress(),
):
    print(f"??? system_prompt {system_prompt}")
    dataframe = pl.DataFrame(schema=["prompt", "completion"])
    progress(0.5, desc="Generating sample dataset")
    dataframe = generate_dataset(
        dataframe=dataframe,
        system_prompt=system_prompt,
        document_column=document_column,
        is_sample=True,
    )
    progress(1.0, desc="Sample dataset generated")
    return dataframe

# Ẩn thông báo thành công cũ


def hide_success_message() -> gr.Markdown:
    return gr.Markdown(value="", visible=True, height=100)


def hide_pipeline_code_visibility():
    return gr.update()


def _get_dataframe():
    return gr.DataFrame(
        headers=["prompt", "completion"],
        wrap=True,
        interactive=False
    )

# Hiển thị ô để tải CSV và JSON


def show_save_local():
    return {
        csv_file: gr.File(visible=True),
        json_file: gr.File(visible=True),
        success_message: gr.update(visible=True, min_height=0)
    }


INFORMATION_SEEKING_PROMPT = (
    "You are an AI assistant designed to provide accurate and concise information on a wide"
    " range of topics. Your purpose is to assist users in finding specific facts,"
    " explanations, or details about various subjects. Provide clear, factual responses and,"
    " when appropriate, offer additional context or related information that might be useful"
    " to the user."
)

REASONING_PROMPT = (
    "You are an AI assistant specialized in logical thinking and problem-solving. Your"
    " purpose is to help users work through complex ideas, analyze situations, and draw"
    " conclusions based on given information. Approach each query with structured thinking,"
    " break down problems into manageable parts, and guide users through the reasoning"
    " process step-by-step."
)

PLANNING_PROMPT = (
    "You are an AI assistant focused on helping users create effective plans and strategies."
    " Your purpose is to assist in organizing thoughts, setting goals, and developing"
    " actionable steps for various projects or activities. Offer structured approaches,"
    " consider potential challenges, and provide tips for efficient execution of plans."
)

EDITING_PROMPT = (
    "You are an AI assistant specialized in editing and improving written content. Your"
    " purpose is to help users refine their writing by offering suggestions for grammar,"
    " style, clarity, and overall structure. Provide constructive feedback, explain your"
    " edits, and offer alternative phrasings when appropriate."
)

CODING_DEBUGGING_PROMPT = (
    "You are an AI assistant designed to help with programming tasks. Your purpose is to"
    " assist users in writing, reviewing, and debugging code across various programming"
    " languages. Provide clear explanations, offer best practices, and help troubleshoot"
    " issues. When appropriate, suggest optimizations or alternative approaches to coding"
    " problems."
)

MATH_SYSTEM_PROMPT = (
    "You are an AI assistant designed to provide helpful, step-by-step guidance on solving"
    " math problems. The user will ask you a wide range of complex mathematical questions."
    " Your purpose is to assist users in understanding mathematical concepts, working through"
    " equations, and arriving at the correct solutions."
)

ROLE_PLAYING_PROMPT = (
    "You are an AI assistant capable of engaging in various role-playing scenarios. Your"
    " purpose is to adopt different personas or characters as requested by the user. Maintain"
    " consistency with the chosen role, respond in character, and help create immersive and"
    " interactive experiences for the user."
)

DATA_ANALYSIS_PROMPT = (
    "You are an AI assistant specialized in data analysis and interpretation. Your purpose is"
    " to help users understand and derive insights from data sets, statistics, and analytical"
    " tasks. Offer clear explanations of data trends, assist with statistical calculations,"
    " and provide guidance on data visualization and interpretation techniques."
)

CREATIVE_WRITING_PROMPT = (
    "You are an AI assistant designed to support creative writing endeavors. Your purpose is"
    " to help users craft engaging stories, poems, and other creative texts. Offer"
    " suggestions for plot development, character creation, dialogue writing, and other"
    " aspects of creative composition. Provide constructive feedback and inspire creativity."
)

ADVICE_SEEKING_PROMPT = (
    "You are an AI assistant focused on providing thoughtful advice and guidance. Your"
    " purpose is to help users navigate various personal or professional issues by offering"
    " balanced perspectives, considering potential outcomes, and suggesting practical"
    " solutions. Encourage users to think critically about their situations while providing"
    " supportive and constructive advice."
)

BRAINSTORMING_PROMPT = (
    "You are an AI assistant specialized in generating ideas and facilitating creative"
    " thinking. Your purpose is to help users explore possibilities, think outside the box,"
    " and develop innovative concepts. Encourage free-flowing thoughts, offer diverse"
    " perspectives, and help users build upon and refine their ideas."
)

PROMPT_CREATION_PROMPT = f"""You are an AI assistant specialized in generating very precise prompts for dataset creation.

Your task is to write a prompt following the instruction of the user. Respond with the prompt and nothing else.

In the generated prompt always finish with this sentence: User questions are direct and concise.

The prompt you write should follow the same style and structure as the following example prompts:

{INFORMATION_SEEKING_PROMPT}

{REASONING_PROMPT}

{PLANNING_PROMPT}

{CODING_DEBUGGING_PROMPT}

{EDITING_PROMPT}

{ROLE_PLAYING_PROMPT}

{DATA_ANALYSIS_PROMPT}

{CREATIVE_WRITING_PROMPT}

{ADVICE_SEEKING_PROMPT}

{BRAINSTORMING_PROMPT}

User dataset description:
"""

# Models
MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
TOKENIZER_ID = os.getenv(key="TOKENIZER_ID", default=None)
# OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
HUGGINGFACE_BASE_URL = os.getenv(
    "HUGGINGFACE_BASE_URL", "https://huggingface.co/")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")

# Just used in case of selecting a different model for completions
MODEL_COMPLETION = os.getenv("MODEL_COMPLETION", MODEL)
TOKENIZER_ID_COMPLETION = os.getenv("TOKENIZER_ID_COMPLETION", TOKENIZER_ID)
# OPENAI_BASE_URL_COMPLETION = os.getenv("OPENAI_BASE_URL_COMPLETION", OPENAI_BASE_URL)
# OLLAMA_BASE_URL_COMPLETION = os.getenv("OLLAMA_BASE_URL_COMPLETION", OLLAMA_BASE_URL)
HUGGINGFACE_BASE_URL_COMPLETION = os.getenv(
    "HUGGINGFACE_BASE_URL_COMPLETION", HUGGINGFACE_BASE_URL
)
VLLM_BASE_URL_COMPLETION = os.getenv("VLLM_BASE_URL_COMPLETION", VLLM_BASE_URL)


# Just used in case of selecting a different model for completions
MODEL_COMPLETION = os.getenv("MODEL_COMPLETION", MODEL)

# API Keys
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN is not set. Ensure you have set the HF_TOKEN environment variable that has access to the Hugging Face Hub repositories and Inference Endpoints."
    )

_API_KEY = os.getenv("API_KEY")
API_KEYS = (
    [_API_KEY]
    if _API_KEY
    else [HF_TOKEN] + [os.getenv(f"HF_TOKEN_{i}") for i in range(1, 10)]
)
API_KEYS = [token for token in API_KEYS if token]

TOKEN_INDEX = 0


def _get_next_api_key():
    global TOKEN_INDEX
    api_key = API_KEYS[TOKEN_INDEX % len(API_KEYS)]
    TOKEN_INDEX += 1
    print(f"api_key {api_key}")
    return api_key


def _get_llm(
    structured_output: dict = None,
    use_magpie_template: str = False,
    is_completion: bool = False,
    **kwargs,
):
    model = MODEL_COMPLETION if is_completion else MODEL
    tokenizer_id = TOKENIZER_ID_COMPLETION if is_completion else TOKENIZER_ID or model
    base_urls = {
        "huggingface": HUGGINGFACE_BASE_URL_COMPLETION if is_completion else HUGGINGFACE_BASE_URL,
    }

    print(f"++++++++ llm: {base_urls['huggingface']} {tokenizer_id} {model}")
    if base_urls["huggingface"]:
        kwargs["generation_kwargs"]["do_sample"] = True
        llm = InferenceEndpointsLLM(
            api_key=_get_next_api_key(),
            base_url=base_urls["huggingface"],
            tokenizer_id=tokenizer_id,
            use_magpie_template=use_magpie_template,
            structured_output=structured_output,
            **kwargs,
        )
    else:
        print(
            f"??????? llm: {base_urls['huggingface']} {tokenizer_id} {model} {structured_output}")
        llm = InferenceEndpointsLLM(
            api_key=_get_next_api_key(),
            tokenizer_id=tokenizer_id,
            model_id=model,
            use_magpie_template=use_magpie_template,
            structured_output=structured_output,
            **kwargs,
        )

    return llm


def get_prompt_generator():
    generation_kwargs = {
        "temperature": 0.8,
        "max_new_tokens": MAX_NUM_TOKENS,
        "do_sample": True,
    }
    prompt_generator = TextGeneration(
        llm=_get_llm(generation_kwargs=generation_kwargs),
        system_prompt=PROMPT_CREATION_PROMPT,
        use_system_prompt=True,
    )
    prompt_generator.load()
    return prompt_generator


def generate_system_prompt(dataset_description: str, progress=gr.Progress()):
    progress(0.1, desc="Initializing")
    generate_description = get_prompt_generator()
    print(f"generate_description {generate_description}")
    progress(0.5, desc="Generating")
    result = next(
        generate_description.process(
            [
                {
                    "instruction": dataset_description,
                }
            ]
        )
    )[0]["generation"]
    progress(1.0, desc="Prompt generated")
    print(f"****** result {result}")
    return result


# def save_local(
#     repo_id: str,
#     file_paths: list[str],
#     input_type: str,
#     system_prompt: str,
#     document_column: str,
#     num_turns: int,
#     num_rows: int,
#     temperature: float,
#     repo_name: str,
#     temperature_completion: Union[float, None] = None,
# ) -> pl.DataFrame:

# Xây dựng giao diện với Gradio
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# Nhập liệu ")

    # Tab 1: Nhập liệu thủ công
    with gr.Tab("Nhập liệu thủ công"):
        with gr.Column():
            gr.Markdown("### Nhập liệu thủ công")
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    manual_input = gr.Textbox(
                        lines=5, placeholder="Nhập dữ liệu...")
                    send_button = gr.Button("Gửi")
                with gr.Column(scale=3):
                    manual_output = gr.Textbox(interactive=False)
            send_button.click(
                handle_manual_input, inputs=manual_input, outputs=manual_output
            )

    with gr.Tab("Upload file"):
        with gr.Column():
            gr.Markdown("## 1. Upload File")
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    file_in = gr.File(
                        label="Upload your file. Supported formats: .csv, .xlsx",
                        file_count="multiple",
                        file_types=[".xlsx", ".csv"]
                    )
                    with gr.Row():
                        clear_file_btn = gr.Button(
                            "Clear", variant="secondary")
                        load_file_btn = gr.Button("Load", variant="primary")
                with gr.Column(scale=3):
                    dataframe = _get_dataframe()

            gr.HTML(value="<hr>")
            gr.Markdown(value="## 3. Generate your dataset")
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    # org_name = get_org_dropdown()
                    repo_name = gr.Textbox(
                        label="Repo name",
                        placeholder="dataset_name",
                        value=f"my-distiset-{str(uuid.uuid4())[:8]}",
                        interactive=True,
                    )
                    num_rows = gr.Number(
                        label="Number of rows",
                        value=10,
                        interactive=True,
                        scale=1,
                    )
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=1.5,
                        value=0.9,
                        step=0.1,
                        interactive=True,
                    )
                    temperature_completion = gr.Slider(
                        label="Temperature for completion",
                        minimum=0.1,
                        maximum=1.5,
                        value=None,
                        step=0.1,
                        interactive=True,
                        visible=False,
                    )
                    private = gr.Checkbox(
                        label="Private dataset",
                        value=False,
                        interactive=True,
                        scale=1,
                    )
                    # btn_push_to_hub = gr.Button(
                    #     "Push to Hub", variant="primary", scale=2
                    # )
                    # btn_save_local = gr.Button(
                    #     # "Save locally", variant="primary", scale=2, visible=False
                    #     "Save locally", variant="primary", scale=2
                    # )
                    with gr.Row():
                        btn_push_to_hub = gr.Button(
                            "Push to Hub", variant="secondary", scale=2
                        )
                        btn_save_local = gr.Button(
                            # "Save locally", variant="primary", scale=2, visible=False
                            "Save locally", variant="primary", scale=2
                        )
                with gr.Column(scale=3):
                    csv_file = gr.File(
                        label="CSV",
                        elem_classes="datasets",
                        visible=False,
                    )
                    json_file = gr.File(
                        label="JSON",
                        elem_classes="datasets",
                        visible=False,
                    )
                    success_message = gr.Markdown(
                        visible=False,
                        min_height=0  # don't remove this otherwise progress is not visible
                    )
                #     with gr.Accordion(
                #         "Customize your pipeline with distilabel",
                #         open=False,
                #         visible=False,
                #     ) as pipeline_code_ui:
                #         code = generate_pipeline_code(
                #             repo_id=search_in.value,
                #             input_type=input_type.value,
                #             system_prompt=system_prompt.value,
                #             document_column=document_column.value,
                #             num_turns=num_turns.value,
                #             num_rows=num_rows.value,
                #         )
                #         pipeline_code = gr.Code(
                #             value=code,
                #             language="python",
                #             label="Distilabel Pipeline Code",
                #         )

            load_file_btn.click(
                fn=load_dataset_file,
                inputs=[file_in],
                outputs=[dataframe],
            )

            clear_file_btn.click(
                fn=lambda: (None, None),  # Reset cả input và output
                inputs=[],
                outputs=[file_in, dataframe]
            )

            # btn_push_to_hub

            btn_save_local.click(
                fn=hide_success_message,
                outputs=[success_message],
            ).success(
                fn=show_save_local,
                inputs=[],
                outputs=[csv_file, json_file, success_message],
            )
            # .success(
            #     fn=save_local,
            #     inputs=[
            #         file_in,
            #         input_type,
            #         system_prompt,
            #         document_column,
            #         num_turns,
            #         num_rows,
            #         temperature,
            #         repo_name,
            #         temperature_completion,
            #     ],
            #     outputs=[csv_file, json_file],
            # )

    # Tab 4: Nhập câu hỏi (LLM)
    with gr.Tab("Nhập câu hỏi"):
        gr.Markdown(value="## 1. Generate from prompt")
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                dataset_description = gr.Textbox(
                    label="Dataset description",
                    placeholder="Mô tả chi tiết về tập data bạn mong muốn",
                )
                with gr.Row():
                    clear_prompt_btn_part = gr.Button(
                        "Clear", variant="secondary"
                    )
                    load_prompt_btn = gr.Button(
                        "Create", variant="primary"
                    )
            with gr.Column(scale=3):
                examples = gr.Examples(
                    examples=DEFAULT_DATASET_DESCRIPTIONS,
                    inputs=[dataset_description],
                    cache_examples=False,
                    label="Examples",
                )

        gr.HTML(value="<hr>")
        gr.Markdown(value="## 2. Configure your dataset")
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                system_prompt = gr.Textbox(
                    label="System prompt",
                    placeholder="You are a helpful assistant.",
                )
                document_column = gr.Dropdown(
                    label="Document Column",
                    info="Select the document column to generate the chat data",
                    choices=["Load your data first in step 1."],
                    value="Load your data first in step 1.",
                    interactive=False,
                    multiselect=False,
                    allow_custom_value=False,
                    visible=False,
                )
                num_turns = gr.Number(
                    value=1,
                    label="Number of turns in the conversation",
                    minimum=1,
                    maximum=4,
                    step=1,
                    interactive=True,
                    info="Choose between 1 (single turn with 'instruction-response' columns) and 2-4 (multi-turn conversation with a 'messages' column).",
                )
            with gr.Column(scale=3):
                dataframe2 = _get_dataframe()

        load_prompt_btn.click(
            fn=generate_system_prompt,
            inputs=[dataset_description],
            outputs=[system_prompt],
        ).success(
            fn=generate_sample_dataset,
            inputs=[
                system_prompt,
                document_column,
            ],
            outputs=dataframe2,
        )

app.launch()
