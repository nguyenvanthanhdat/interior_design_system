import gradio as gr
from pathlib import Path
import os
import rawpy
import torch
from src import wb_net
import numpy as np
import cv2
import imageio
from PIL import Image
from matplotlib import pyplot as plt
import sys
from typing import Any, Dict, List
# from re_tabs.tab1 import Tab1
from re_tabs.tab2 import Tab2
# from re_tabs.tab3 import Tab3


with gr.Blocks() as mvp:
    # Tab1()
    Tab2()
    # Tab3()

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    # for i in range(1,4):
    for i in range(2,3):
        os.makedirs(f"outputs/input_tab{i}", exist_ok=True)
        os.makedirs(f"outputs/output_tab{i}", exist_ok=True)
    os.makedirs(f"outputs/temps", exist_ok=True)
    mvp.launch(share=True)
    # main()