#!/usr/bin/env python
# coding: utf-8

# # Model Deployment

# In[1]:


from model_training import model
from resistance_ai import decode_reality


# In[2]:


real_world = model.extract_data()


# In[3]:


predictions = model.fit(real_world)
decode_reality.visualize(predictions)


# ![Three groups of people stand on a terrain. The group to the right shows a group of 2 male and 2 female caucasians, all dressed in different shades of red. The groups in the center and to the left show people of different ethnicities and genders. They are all dressed in greens and blues. There is a helicopter flying above them, dropping ‘color bombs’ onto the people in blue and green. The bombs have exploded and are coloring the people red. The helicopter blades read the words ‘Erasure’, ‘Oppression’ and ‘Colonization’.](Predictions.gif)

# In[4]:


decode_reality.explain(predictions)


# Remember those spurious class labels? The ones that did not represent the ground truth and were merely proxies for infinitely more complicated social phenomena. What do you think happens when data-driven algorithms get deployed back into the real world? The groups whose identities were erased while determining class labels are further oppressed by algorithmic decision making systems that enforce reductive categorizations in the real world.
# 
# Predictions from algorithms are used to make decisions about everything under the sun, and entities, cultures and values that do not conform with the system in which these models were created are forcibly erased<sup>8</sup>.
# 
# The ones in power control the algorithms and thereby quickly start dictating culture, values, judgements, popular opinion and a whole plethora of constructs, far beyond the scope of the prediction task itself<sup>9</sup>. 
# 
# If you think building AI is a purely engineering/math problem, that has nothing to do with Social hierarchies and Power dynamics, THINK AGAIN.  
