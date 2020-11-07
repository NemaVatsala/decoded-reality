#!/usr/bin/env python
# coding: utf-8

# # Model Inputs

# In[1]:


import pandas as pd
from resistance_ai import decode_reality


# In[2]:


model_inputs = pd.read_csv("Data.csv")
decode_reality.visualize(model_inputs)


# <img src="Data.gif" width="800" class="responsive" alt="An arcade mechanical picker game is depicted, where the toys are replaced by people’s faces. There’s a stark disparity between the abundance of ‘western’ faces and the dearth of those with darker skin tones. There is a sign on top of the game saying “diverse, ethical, fair” with the faces of prominent scholars Deborah Raji,Timnit Gebru and Joy Boulamwini. A person in a jacket with the logos of big tech companies is seen operating the joystick. His face is superimposed with the words ‘A Silicon Valley type’.The mechanical hand is moving downwards towards the toy faces and has the words Efficiency, Advancement, Progress and Innovation written on it">
# <p> </p>

# In[3]:


decode_reality.explain(model_inputs)


# Data is for the taking. No matter who, no matter how private or personal the data in question.
# 
# Despite there being sufficient scholarship documenting the harm of existing approaches (such as Facial Recognition<sup>5</sup>), the same old adage of ‘Progress and Innovation’ is touted and used to justify the obsession with creating technological solutions to societal problems. Self-affirming ‘Fairness’ and ‘Ethics’ certificates are stamped onto data collection strategies, while the ground reality remains discriminatory and predatory. 
# 
# A data point might just look like a bunch of numbers loaded in a Data frame, but the irrevocable truth is that it was collected by a power-wielding entity, about a living-breathing human being, in a socio-techno-political context<sup>6</sup>. 
# 
# Datasets aren't created out of thin air. Digital artifacts live well beyond their intended shelf life. We need to stop abstracting away the Ethics of Data and start taking responsibility for validating the efficacy of the data we are using—whether we collected it ourselves or took it off someone else's shelf. 
