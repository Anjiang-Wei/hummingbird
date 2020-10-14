
# coding: utf-8

# In[ ]:


get_ipython().system('pip install hummingbird_ml')


# In[1]:


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from hummingbird.ml import convert


# In[2]:


# We are going to use the breast cancer dataset from scikit-learn for this example.
X, y = load_breast_cancer(return_X_y=True)
nrows=15000
X = X[0:nrows]
y = y[0:nrows]


# In[3]:


# Create and train a random forest model.
model = RandomForestClassifier(n_estimators=10, max_depth=10)
model.fit(X, y)


# In[4]:


get_ipython().run_cell_magic('timeit', '-r 3', '\n# Time for scikit-learn.\nmodel.predict(X)')


# In[5]:


model = convert(model, 'torch', extra_config={"tree_implementation":"gemm"})


# In[6]:


get_ipython().run_cell_magic('timeit', '-r 3', '\n# Time for HB.\nmodel.predict(X)')


# In[7]:


model.to('cuda')


# In[8]:


get_ipython().run_cell_magic('timeit', '-r 3', '\n# Time for HB GPU.\nmodel.predict(X)')

