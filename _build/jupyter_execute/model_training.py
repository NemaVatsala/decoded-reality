#!/usr/bin/env python
# coding: utf-8

# # Model Training

# In[1]:


from model_training import model
from resistance_ai import decode_reality


# In[2]:


model_inputs, model_outputs = model.see_previous_panels()


# In[3]:


optimization = model.fit(model_inputs, model_outputs)
decode_reality.visualize(optimization)


# ![Two ethereal forms are in the center of the screen. The one at the back is composed of colorful pixels and has the word ‘Data’’ written in hollow characters behind it. The words ‘Racism’ and ‘Oppression’ are written within the word ‘Data’. The figure in the front is composed of a network of graph nodes/vertices and edges and has the word ‘Model’ written behind it in hollow characters. The word ‘Discriminatory’ is written within the world ‘Model’. Two humans are looking at the ethereal beings. To the right stands a Caucasian male, who is emphatically blowing a kiss to the Model. To the left stands an African-American female, who is looking at the model in disapproval while flames rise around her. ](Optimization.gif)

# In[4]:


decode_reality.explain(optimization)


# When training a Learning algorithm, our objective is to learn as many associations as possible from the training data, so that the algorithm can use these to faithfully reproduce the same kind of predictions that were made by it’s human predecessors.
# 
# The algorithm has no inherent way of determining which learned associations are ‘good’ and which are ‘harmful’. That is for the human to decide.
# 
# Any association that the human decides is ‘bad’ or incorrectly learned for the task at hand, is deemed to be a ‘Bias’. What happens when the model is trained on data generated from an oppressive and racist society? An ‘accurate’ model will faithfully learn the pre-existing biases<sup>7</sup> in society as discriminatory associations in its training data. 
# 
# If the human in charge is in a position of privilege and does not deem this model behavior as ‘harmful’ or ‘biased’,  we end up deploying predatory models back into the broken ecosystem from which they were created, to in turn give rise to new emergent biases<sup>7</sup>. 
