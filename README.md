# radar-bot
Repo for the Radar Discord Bot I created for Technica 2021. 

Welcome to Radar! Radar is a Discord bot trained to identify microaggressions in messages. Based on the category of microaggression, Radar then provides a specific explanation of why a user's comments were harmful. The bot also comes with an interactive learning module that educates users about different types of microaggressions.

Radar was built entirely in Python. I used NLTK to parse my training dataset, then I trained my logistic regression model using scikit-learn. My logistic regression classifies a message by determining if it is clean or if it falls into one of the identified microaggression categories. If so, the bot responds to the microaggression with a warning and explanation. I crafted the user-bot interactions with discord.py. In addition to the bot's automatic response system, users can manually interact with the bot to take a learning module and increase their own awareness of microaggressions. The learning module advances as the user clicks emoji depending on the path they want to take. 
