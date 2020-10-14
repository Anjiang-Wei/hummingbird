
# coding: utf-8

# In[ ]:


get_ipython().system('pip install hummingbird_ml[extra,onnx]')


# In[1]:


import torch
import numpy as np
import lightgbm as lgb

import onnxruntime as ort
from onnxmltools.convert import convert_lightgbm
from onnxconverter_common.data_types import FloatTensorType

from hummingbird.ml import convert
from hummingbird.ml import constants

# Create some random data for binary classification.
num_classes = 2
X = np.array(np.random.rand(10000, 28), dtype=np.float32)
y = np.random.randint(num_classes, size=10000)


# In[2]:


# Create and train a model (LightGBM in this case).
model = lgb.LGBMClassifier()
model.fit(X, y)


# In[3]:


# Use ONNXMLTOOLS to convert the model to ONNXML.
initial_types = [("input", FloatTensorType([X.shape[0], X.shape[1]]))] # Define the inputs for the ONNX
onnx_ml_model = convert_lightgbm(
    model, initial_types=initial_types, target_opset=9
)


# In[4]:


# Use Hummingbird to convert the ONNXML model to ONNX.
onnx_model = convert(onnx_ml_model, "onnx", X)


# In[5]:


# Alternatively we can set the inital types using the extra_config parameters as in the ONNXMLTOOL converter.
extra_config = {}
extra_config[constants.ONNX_INITIAL_TYPES] = initial_types
onnx_model = convert(onnx_ml_model, "onnx", extra_config=extra_config)


# In[6]:


get_ipython().run_cell_magic('timeit', '-r 3', '\n# Run the ONNX model on CPU \nonnx_model.predict(X)')

