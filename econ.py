# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:43:09 2024

@author: fsmai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pip install fredapi
import fredapi as fd
import plotly.express as px

fred = fd.Fred(api_key = 'dfa506a19c1b236758e828c26afdad25')
data = fred.search('Consumer Price Index for All Urban Consumers: All Items in U.S. City Average')
data.head(10)

data['title'][0]
cpi = fred.get_series('CPIAUCSL')
cpi.name-'values'
cpi
     

















