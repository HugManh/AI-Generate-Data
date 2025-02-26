import gradio as gr
import polars as pl
import uuid
import tqdm
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition

# Xử lý nhập liệu thủ công
def handle_manual_input(manual_input):
    return f"Dữ liệu đã nhập: {manual_input}"

# Xử lý upload file Excel
def handle_excel_upload(file):
    df = pl.read_excel(file.name, sheet_id=0)
    return df.head().to_pandas().to_html()  # Chuyển sang Pandas để render HTML

# Xử lý upload file CSV bằng Polars
def load_dataset_file(
    file_paths: list[str],
    num_rows: int = 10,
):
    # df = pl.read_csv(file.name)
    # # return df.head().to_pandas()  # Chuyển sang Pandas để render HTML
    # return df.to_pandas()  # Chuyển sang Pandas để render HTML
    return preprocess_input_data(file_paths=file_paths, num_rows=num_rows)


def preprocess_input_data(
    file_paths: list[str], num_rows: int, progress=gr.Progress(track_tqdm=True)
):
    if not file_paths:
        raise gr.Error("Vui lòng cung cấp ít nhất 1 file")

    data = {}
    total_chunks = 0

    for file_path in tqdm(file_paths, desc="Processing files", total=len(file_paths)):
        partitioned_file = partition(filename=file_path)
        chunks = [str(chunk) for chunk in chunk_by_title(partitioned_file)]
        data[file_path] = chunks
        total_chunks += len(chunks)
        if total_chunks >= num_rows:
            break

    dataframe = pl.DataFrame.from_records(
        [(k, v) for k, values in data.items() for v in values],
        columns=["filename", "chunks"],
    )
    col_doc = "chunks"

    return (
        dataframe,
        gr.Dropdown(
            choices=["chunks"],
            label="Documents column",
            value=col_doc,
            interactive=(False if col_doc == "" else True),
            multiselect=False,
        ),
    )

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

    # with gr.Tab("Upload file"):
    #     gr.Markdown("## Select your input")
    #     with gr.Row(equal_height=False):
    #         # Tab 2: Upload file Excel
    #         with gr.Tab("Upload Excel"):
    #             with gr.Column():
    #                 gr.Markdown("### Upload file Excel (Polars)")
    #                 with gr.Row(equal_height=False):
    #                     with gr.Column(scale=2):
    #                         excel_input = gr.File(file_types=[".xlsx"])
    #                         show_button = gr.Button("Xem nội dung")
    #                     with gr.Column(scale=3):
    #                         excel_output = gr.DataFrame(
    #                             interactive=False, label="Nội dung CSV")
    #                 show_button.click(
    #                     handle_excel_upload, inputs=excel_input, outputs=excel_output
    #                 )

    #         # Tab 3: Upload file CSV
    #         with gr.Tab("Upload CSV"):
    #             with gr.Column():
    #                 gr.Markdown("## 1. Upload file CSV (Polars)")
    #                 with gr.Row(equal_height=False):
    #                     with gr.Column(scale=2):
    #                         file_in = gr.File(file_types=[".csv"])
    #                         with gr.Row():
    #                             clear_file_btn = gr.Button(
    #                                 "Clear", variant="secondary")
    #                             load_file_btn = gr.Button(
    #                                 "Load", variant="primary")
    #                     with gr.Column(scale=3):
    #                         dataframe = gr.DataFrame(
    #                             headers=["prompt", "completion"],
    #                             wrap=True,
    #                             interactive=False
    #                         )

    #         gr.HTML(value="<hr>")
    #         gr.Markdown(value="## 2. Generate your dataset")
    #         with gr.Row(equal_height=False):
    #             with gr.Column(scale=2):
    #                 # org_name = get_org_dropdown()
    #                 repo_name = gr.Textbox(
    #                     label="Repo name",
    #                     placeholder="dataset_name",
    #                     value=f"my-distiset-{str(uuid.uuid4())[:8]}",
    #                     interactive=True,
    #                 )
    #                 num_rows = gr.Number(
    #                     label="Number of rows",
    #                     value=10,
    #                     interactive=True,
    #                     scale=1,
    #                 )
    #                 temperature = gr.Slider(
    #                     label="Temperature",
    #                     minimum=0.1,
    #                     maximum=1.5,
    #                     value=0.9,
    #                     step=0.1,
    #                     interactive=True,
    #                 )
    #                 temperature_completion = gr.Slider(
    #                     label="Temperature for completion",
    #                     minimum=0.1,
    #                     maximum=1.5,
    #                     value=None,
    #                     step=0.1,
    #                     interactive=True,
    #                     visible=False,
    #                 )
    #                 private = gr.Checkbox(
    #                     label="Private dataset",
    #                     value=False,
    #                     interactive=True,
    #                     scale=1,
    #                 )
    #                 # btn_push_to_hub = gr.Button(
    #                 #     "Push to Hub", variant="primary", scale=2
    #                 # )
    #                 # btn_save_local = gr.Button(
    #                 #     # "Save locally", variant="primary", scale=2, visible=False
    #                 #     "Save locally", variant="primary", scale=2
    #                 # )
    #                 with gr.Row():
    #                     btn_push_to_hub = gr.Button(
    #                         "Push to Hub", variant="secondary", scale=2
    #                     )
    #                     btn_save_local = gr.Button(
    #                         # "Save locally", variant="primary", scale=2, visible=False
    #                         "Save locally", variant="primary", scale=2
    #                     )
    #             with gr.Column(scale=3):
    #                 csv_file = gr.File(
    #                     label="CSV",
    #                     elem_classes="datasets",
    #                     visible=False,
    #                 )
    #                 json_file = gr.File(
    #                     label="JSON",
    #                     elem_classes="datasets",
    #                     visible=False,
    #                 )
    #                 success_message = gr.Markdown(
    #                     visible=False,
    #                     min_height=0  # don't remove this otherwise progress is not visible
    #                 )
    #             #     with gr.Accordion(
    #             #         "Customize your pipeline with distilabel",
    #             #         open=False,
    #             #         visible=False,
    #             #     ) as pipeline_code_ui:
    #             #         code = generate_pipeline_code(
    #             #             repo_id=search_in.value,
    #             #             input_type=input_type.value,
    #             #             system_prompt=system_prompt.value,
    #             #             document_column=document_column.value,
    #             #             num_turns=num_turns.value,
    #             #             num_rows=num_rows.value,
    #             #         )
    #             #         pipeline_code = gr.Code(
    #             #             value=code,
    #             #             language="python",
    #             #             label="Distilabel Pipeline Code",
    #             #         )

    #         load_file_btn.click(
    #             fn=load_dataset_file,
    #             inputs=[file_in],
    #             outputs=[dataframe],
    #         )

    #         clear_file_btn.click(
    #             fn=lambda: (None, None),  # Reset cả input và output
    #             inputs=[],
    #             outputs=[file_in, dataframe]
    #         )

    #         # btn_push_to_hub
            
    #         btn_save_local.click(
    #             fn=hide_success_message,
    #             outputs=[success_message],
    #         ).success(
    #             fn=show_save_local,
    #             inputs=[],
    #             outputs=[csv_file, json_file, success_message],
    #         )
    #         # .success(
    #         #     fn=save_local,
    #         #     inputs=[
    #         #         file_in,
    #         #         input_type,
    #         #         system_prompt,
    #         #         document_column,
    #         #         num_turns,
    #         #         num_rows,
    #         #         temperature,
    #         #         repo_name,
    #         #         temperature_completion,
    #         #     ],
    #         #     outputs=[csv_file, json_file],
    #         # )

    # Tab 4: Nhập câu hỏi (LLM)
    with gr.Tab("Nhập câu hỏi"):
        gr.Markdown("### Nhập câu hỏi (LLM)")
        question_input = gr.Textbox(placeholder="Nhập câu hỏi...")
        llm_output = gr.Textbox(interactive=False)
        gr.Button("Gửi").click(
            lambda x: f"Bạn đã hỏi: {x}", inputs=question_input, outputs=llm_output
        )

app.launch()
