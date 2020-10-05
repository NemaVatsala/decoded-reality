#!/usr/bin/env python
# coding: utf-8

# # Model Inputs

# In[1]:


import pandas as pd
from resistance_ai import decode_reality


# In[2]:


model_inputs = pd.read_csv("Data.csv")
decode_reality.visualize(model_inputs)


# ![An arcade mechanical picker game is depicted, where the toys are replaced by people’s faces. There’s a stark disparity between the abundance of ‘western’ faces and the dearth of those with darker skin tones. There is a sign on top of the game saying “diverse, ethical, fair” with the faces of prominent scholars Deborah Raji,Timnit Gebru and Joy Boulamwini. A person in a jacket with the logos of big tech companies is seen operating the joystick. His face is superimposed with the words ‘A Silicon Valley type’.The mechanical hand is moving downwards towards the toy faces and has the words Efficiency, Advancement, Progress and Innovation written on it](Data.gif)

# In[3]:


decode_reality.explain(model_inputs)


# Data is for the taking. No matter who, no matter how private or personal the data in question.
# 
# Even despite there being sufficient scholarship proving the harm of certain applications (such as Facial Recognition), the same old adage of Progress and Innovation is touted and used to justify the obsession with creating technological solutions to societal problems.
# 
# Self-affirming ‘Fairness’ and ‘Ethics’ certificates are stamped onto data collection strategies, while the ground reality remains discriminatory and predatory. 
# 
# A data point might just look like a bunch of numbers loaded in a Data frame, but the irrevocable truth is that it wasn’t created out of thin air – it was collected by an power-wielding entity, about a living-breathing human being, in a socio-techno-political context.
