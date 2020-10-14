# coding: utf-8
import time
import numpy as np
# In[ ]:


#get_ipython().system('pip install hummingbird_ml')


# In[1]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from hummingbird.ml import convert


# #### Create and fit the model

# In[2]:


# Create and train a RandomForestClassifier model
X, y = load_breast_cancer(return_X_y=True)
skl_model = RandomForestClassifier(n_estimators=500, max_depth=7)
skl_model.fit(X, y)


# #### Time scikit-learn

# In[3]:


# get_ipython().run_cell_magic('timeit', '', 'pred = skl_model.predict(X)')
rec = []
pred = skl_model.predict(X)
for i in range(0, 10):
    st = time.time()
    pred = skl_model.predict(X)
    ed = time.time()
    rec.append(ed-st)
print("scikit-learn:", np.mean(rec))

# #### Convert SKL model to PyTorch

# In[4]:

model = convert(skl_model, 'torch')


# #### Time PyTorch - CPU

# In[5]:


#get_ipython().run_cell_magic('timeit', '', 'pred_cpu_hb = model.predict(X)')
rec = []
pred_cpu_hb = model.predict(X)
for i in range(0, 10):
    st = time.time()
    pred = model.predict(X)
    ed = time.time()
    rec.append(ed-st)
print("pytorch_cpu:", np.mean(rec))
# #### Switch PyTorch from CPU to GPU

# In[6]:


#get_ipython().run_cell_magic('capture', '', "model.to('cuda')")
model.to('cuda')


# #### Time PyTorch - GPU

# In[7]:
rec = []
# get_ipython().run_cell_magic('timeit', '', 'pred_gpu_hb = model.predict(X)')
pred_gpu_hb = model.predict(X)
for i in range(0, 10):
    st = time.time()
    pred = model.predict(X)
    ed = time.time()
    rec.append(ed-st)
print("pytorch_gpu:", np.mean(rec))

# scikit-learn: 0.06132786273956299
# pytorch_cpu: 0.39469361305236816
# pytorch_gpu: 0.002163410186767578
