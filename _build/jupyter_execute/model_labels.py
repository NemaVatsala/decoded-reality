#!/usr/bin/env python
# coding: utf-8

# # Model Labels

# In[1]:


import pandas as pd
from resistance_ai import decode_reality


# In[2]:


model_outputs = pd.read_csv("Labels.csv")
decode_reality.visualize(model_outputs)


# ![A hooded white male shoots an arrow into a large terrain. The male’s body is superimposed with the word “Engineering”. The arrow reads “Objectivity”. The arrow is shown to pierce into the ground and reads the word ‘Classification’ The ground starts splitting into 4-5 different land masses and people at the boundary start to fall into the abyss below. The people that remain on land are colored uniformly, while the people falling into the abyss are multicolored. There are a few people on the terrains trying to help the people falling below.](Labels.gif)

# In[3]:


decode_reality.explain(model_outputs)


# To teach an algorithm how to make a prediction, we must feed it with the ‘true’ predictions – usually, the predictions made by it’s human predecessors –  as class labels for the given task.
# 
# But, human decisions are inherently subjective! In most tasks, there is no ground truth and class labels are just proxies for underlying social phenomena. And so, we go about trying to quantify the qualitative<sup>7</sup>, enforcing ‘objective’ mathematical formalizations of social constructs that are inherently complex and fluid.
# 
# Those who are already well represented in society get a corresponding rich representation in data, whereas those who do not fit into any one social construct, fall through the cracks. To be represented in the data, they are forced to give up their identity and be pushed into one of the ‘acceptable’ predetermined classes<sup>8</sup>.
# 
# Moreover, since they do not actually conform with their allocated class label, the algorithm sees them as appearing ‘on the margin’ and the effects of misclassifications from misbehaving algorithms is later felt most profoundly by these demographics. 
