import gradio as gr
import polars as pl
from typing import List, Tuple, Generator
import uuid
from tqdm import tqdm
from multiprocessing import Pool
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition

# Xử lý nhập liệu thủ công


def handle_manual_input(manual_input):
    return f"Dữ liệu đã nhập: {manual_input}"

# Xử lý upload file Excel


def handle_excel_upload(file):
    df = pl.read_excel(file.name, sheet_id=0)
    return df.head().to_pandas().to_html()

# Xử lý upload file CSV bằng Polars


def load_dataset_file(
    file_paths: list[str],
    num_rows: int = 10,
):
    # df = pl.read_csv(file.name)
    # # return df.head().to_pandas()
    # return df.to_pandas()
    return preprocess_input_data(file_paths=file_paths, num_rows=num_rows)


def process_single_file(file_path: str) -> Generator[Tuple[str, str], None, None]:
    """Xử lý một file và trả về generator của (filename, chunk)."""
    try:
        partitioned_file = partition(filename=file_path)
        for chunk in chunk_by_title(partitioned_file):
            yield (file_path, str(chunk))
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def preprocess_input_data(
    file_paths: List[str], num_rows: int = None, progress=gr.Progress(track_tqdm=True)
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
    if not file_paths:
        raise gr.Error("Vui lòng cung cấp ít nhất 1 file")

    # Sử dụng multiprocessing để xử lý song song
    def data_generator():
        # total_chunks = 0
        # for file_path in file_paths:
        #     for record in process_single_file(file_path):
        #         print(f"Yielding record: {record}")
        #         if num_rows is not None and total_chunks >= num_rows:
        #             return
        #         total_chunks += 1
        #         yield record
        return

    # Stream dữ liệu vào DataFrame
    dataframe = pl.DataFrame(
        tqdm(data_generator(), desc="Processing files", total=num_rows or None),
        schema=["filename", "chunks"]
    )

    # Tạo dropdown widget
    col_doc = "chunks"
    dropdown = gr.Dropdown(
        choices=["chunks"],
        label="Documents column",
        value=col_doc,
        # Hiện tại chỉ có 1 cột, nên không cần interactive
        interactive=(False if col_doc == "" else True),
        multiselect=False,
    ),

    return (dataframe, dropdown)

# Tạo ra dataset mẫu


def generate_sample_dataset():
    return

# Ẩn thông báo thành công cũ


def hide_success_message() -> gr.Markdown:
    return gr.Markdown(value="", visible=True, height=100)


def hide_pipeline_code_visibility():
    return gr.update()

# Hiển thị ô để tải CSV và JSON


def show_save_local():
    return {
        csv_file: gr.File(visible=True),
        json_file: gr.File(visible=True),
        success_message: gr.update(visible=True, min_height=0)
    }


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
                        file_types=[".xlsx", ".csv"], label="Upload Excel (.xlsx) or CSV (.csv)")
                    with gr.Row():
                        clear_file_btn = gr.Button("Clear", variant="secondary")
                        load_file_btn = gr.Button("Load", variant="primary")
                with gr.Column(scale=3):
                    dataframe = gr.DataFrame(
                        headers=["prompt", "completion"],
                        wrap=True,
                        interactive=False
                    )

            gr.HTML(value="<hr>")
            gr.Markdown(value="## 2. Generate your dataset")
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
        gr.Markdown("### Nhập câu hỏi (LLM)")
        question_input = gr.Textbox(placeholder="Nhập câu hỏi...")
        llm_output = gr.Textbox(interactive=False)
        gr.Button("Gửi").click(
            lambda x: f"Bạn đã hỏi: {x}", inputs=question_input, outputs=llm_output
        )

app.launch()
