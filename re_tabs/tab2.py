import gradio as gr
from re_func_tab.func_tab2 import *
import cv2, os
import pandas as pd

def Tab2():
    with gr.Tab("Remove") as remove_tab:
        with gr.Tab("Point"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("Choose way to segment")
                with gr.Column():
                    # method = gr.Dropdown(value="add_point",choices=["add_point", "remove_point"])
                    method = gr.Dropdown(value="add_point",choices=["add_point"])

            with gr.Row():
                with gr.Column():
                    try:
                        image = cv2.imread(os.path.join("outputs", "output_tab1", "input_WB_new.png"))
                        cv2.imwrite(os.path.join("outputs", "input_tab2", "image_0.png"), image) 
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)              
                        input_image = gr.Image(value=image)
                    except:
                        input_image = gr.Image()
                with gr.Column():
                    predict_image = gr.Image()
                    
            with gr.Row():
                with gr.Column():
                    reset_button = gr.Button("Reset")
                with gr.Column():
                    undo_button = gr.Button("Undo")
                with gr.Column():
                    apply_button = gr.Button("Apply")
            with gr.Row():
                remove_image = gr.Image()
            input_image.upload(upload_file, input_image)
            input_image.select(segment_func, inputs=[input_image, method], outputs=[predict_image])
            apply_button.click(apply_func, inputs=[input_image], outputs=[remove_image])
            undo_button.click(undo_func, inputs=[input_image], outputs=[predict_image])
            reset_button.click(reset_func, outputs=[predict_image])
        
        with gr.Tab("Box"):
            pass
        remove_tab.select(init_tab2, outputs=[input_image])
    return remove_tab