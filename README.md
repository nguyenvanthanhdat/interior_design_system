# mvp_wb
This is a mvp for cwcc project

## Install package
```
pip install -r requirements.txt
```

## Inferece
```python
import cv2
import os
from src.model import AWBNet


if __name__ == '__main__':
    model = AWBNet(os.path.join('weights', 'wb.onnx'))
    image = cv2.imread('samples/test.jpg')
    
    img = model.predict([image, image, image])
    
    print(img.shape)
```