"""Signal construction: turning stored events and returns into measured effects.

``dataset`` builds the (event, target, horizon) rows every statistic is computed
from. Later modules consume those rows -- never the raw tables -- so the impact
table and any future model are measured on identical inputs.
"""
