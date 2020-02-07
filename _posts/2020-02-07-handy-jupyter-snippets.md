---
title:  "Handy Jupyter Notebook Snippets for Data Scientists"
date:   2020-02-07 12:00:00 -0700
categories: python, data science workflows
---
There are a few snippets of code that I put at the top of literally every single Jupyter notebook I create. I hope some of them are useful to you, too.

<!--more-->

## For more real estate in your notebook
By default, Jupyter notebooks have wide margins and push all the code cells into a fairly narrow rectangle in the middle of the screen. I like to let my code "expand" a bit so that I can see error messages and results without as many line breaks. These two lines go at the top of every notebook I create and allow the cells to use the full width of the window.

```
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
```

## To auto-reload external packages
One of the biggest frustrations I have with notebooks is when I am using it to debug code that exists outside the notebook, like a relatively mature repo from which I am importing functions or classes. Unfortunately, by default the notebook only loads that module once and, if you change it, you have to restart the kernal in order to pick up those changes.

Except if you use the following lines of reload magic, it will automatically reload every package in the notebook every time you run a new cell. It does slow the notebook down a tiny bit, but I find that trade-off to be well worth it if I'm actively developing or debugging code that doesn't exist inside the notebook.


```
%load_ext autoreload
%autoreload 2
```

## The "Aways Gonna Need Them" Packages
These are the modules I import into every notebook because I'm almost guaranteed to need them. They are part of my notebook boilerplate.

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=250)
%matplotlib inline
```