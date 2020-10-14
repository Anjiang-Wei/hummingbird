
# coding: utf-8

# In[ ]:


get_ipython().system('pip install hummingbird_ml[extra]')


# In[1]:


import numpy as np
import xgboost as xgb
from hummingbird.ml import convert

# Create some random data for binary classification.
num_classes = 2
X = np.random.rand(100000, 28)
y = np.random.randint(num_classes, size=100000)


# In[2]:


# Create and train a model (XGBoost in this case).
model = xgb.XGBRegressor()
model.fit(X, y)


# In[3]:


# Use Hummingbird to convert the model to PyTorch
# Note that XGBRegressor requires us to pass it some sample data.
hb_model = convert(model, 'torch', X[0:1])


# In[4]:


get_ipython().run_cell_magic('timeit', '-r 3', '\n# Run Hummingbird on CPU - By default CPU execution is used in Hummingbird.\nhb_model.predict(X)')


# In[5]:


get_ipython().run_cell_magic('timeit', '-r 3', "\n# Run Hummingbird on GPU (Note that you must have a GPU-enabled machine).\nhb_model.to('cuda')\nhb_model.predict(X)")

