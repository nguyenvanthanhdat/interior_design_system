import numpy as np
import onnxruntime as ort

from src import image_utils

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL



class AWBNet:
    def __init__(self, model_path) -> None:
        self.load_model(model_path)
        
    def load_model(self, model_path):
        self.model = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        self.inp_name = self.model.get_inputs()[0].name
        self.opt_name = self.model.get_outputs()[0].name
        _, h, w, _ = self.model.get_inputs()[0].shape
        self.model_inpsize = (w, h)
    
    def postprocess(self, images):
        if isinstance(images, list):
            return [x.squeeze() for x in images]
        
        return images.squeeze()
    
    def preprocess(self, images, img_size:int=256, aspect_ratio:bool=False):
        s_img, t_img, d_img = images
        if aspect_ratio:
            s_img = image_utils.aspect_ratio_imresize(s_img, max_output=img_size)
            t_img = image_utils.aspect_ratio_imresize(t_img, max_output=img_size)
            d_img = image_utils.aspect_ratio_imresize(d_img, max_output=img_size)
        else:
            s_img = image_utils.imresize(s_img, output_shape=(img_size, img_size))
            t_img = image_utils.imresize(t_img, output_shape=(img_size, img_size))
            d_img = image_utils.imresize(d_img, output_shape=(img_size, img_size))
        
        imgs = np.stack([s_img, t_img, d_img], axis=0)
        num_im, h, w, c = imgs.shape
        imgs = np.reshape(imgs, (num_im*c, w, h))
        imgs = np.expand_dims(imgs, axis=0)
        
        return imgs
    
    def predict(self, images:list[np.array]):
        images = self.preprocess(images)
        
        result = self.model.run(
            [self.opt_name], {self.inp_name: images.astype("float32")}
        )[0]
        
        result = self.postprocess(result)
        
        return result
    