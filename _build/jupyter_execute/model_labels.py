#!/usr/bin/env python
# coding: utf-8

# # Model Labels

# In[1]:


import pandas as pd
from resistance_ai import decode_reality


# In[2]:


model_outputs = pd.read_csv("Labels.csv")
decode_reality.visualize(model_outputs)


# <img src="Labels.gif" width="800" class="responsive" alt="A hooded white male shoots an arrow into a large terrain. The male’s body is superimposed with the word “Engineering”. The arrow reads “Objectivity”. The arrow is shown to pierce into the ground and reads the word ‘Classification’ The ground starts splitting into 4-5 different land masses and people at the boundary start to fall into the abyss below. The people that remain on land are colored uniformly, while the people falling into the abyss are multicolored. There are a few people on the terrains trying to help the people falling below.">
# <p> </p>

#     

# In[3]:


decode_reality.explain(model_outputs)


# All models are wrong, some models are useful<sup>3</sup>. The problem at hand informs our choice of model targets (what to consider as 'true' predictions). In most settings, we use the predictions made by human predecessors as ground truth.
# 
# But, human decisions are inherently subjective! In most tasks, model targets are just proxies for underlying social phenomena. And so, we go about trying to quantify the qualitative<sup>7</sup>, enforcing ‘objective’ mathematical formalizations of social constructs that are inherently fluid. 
# 
# Take 'gender prediction' for example<sup>5</sup>. Gender is a social construct. Historically, gender has been reduced to sex and models trained to predict gender try to learn what a 'woman' looks like, based off of pictures of people labelled as 'female'. 
# 
# What happens to people who do not conform with stereotypical gender binaries? To be represented in the data, those at the fringes of society are forced to give up their identity and be pushed into one of the ‘acceptable’ predetermined classes<sup>8</sup>. On the other hand, those in power get to dictate the dominant social narrative and, by construction, get a rich representation in data. In a society that identifies people with short hair and masculine features as 'men', algorithms will also determine people's genders using the same reductive notions. 
# 
# In our hasty selection of model labels, we shoot the arrow of ‘objectivity’ on to an infinitely complex social landscape. As a result, those who do not neatly align with the prevalent stereotypes fall through the cracks. 
