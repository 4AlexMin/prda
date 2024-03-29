# prda

## Prda contains packages for data processing, analysis and visualization.

Prda ultimate goal is to **fill the “last mile” between analysts and packages**. During my research practice, I have felt how “learning a package before utilizing” can be time-consuming and exhausting. The resulted inefficiency leads to the creation of *prda*.

# Usage
```
pip install prda
```

See details in: https://pypi.org/project/prda/

You are welcome to clone *prda* for personal use and **pull request** of your modification is super!! encouraged.

----


~~To utilize prda, you only need to be familiar with `pandas` as most inputs is `pd.DataFrame`.~~

Currently with the help of **ChatGPT**, you can just tailor the input of demonstration code below to your data. And you don't need to be familiar with pandas or even python.



## Examples of Useage

1. ### For Visulization
```python
import prda
import pandas as pd
import numpy as np
df = pd.DataFrame(data=np.array([np.arange(100) for i in range(5)]).T,columns=['a', 'b', 'c', 'd', 'e'])
prda.graphic.scatter_3d_html(df, x='a', y='b', z='c', color_hue='d', size_hue='e', title='demo_3d_scatter', filepath='demo_3d_scatter.html')
```

the above code will provide an interactive html figure that look like this:



<!-- <iframe src="demo/demo_lineplot.html" width="500" height="300"></iframe> -->

![Image.png](/demo/demo_3d_scatter_screenshot.png)

[demo_3d_scatter.html](/demo/demo_3d_scatter.html)

----

```python
import prda
import pandas as pd
import numpy as np
datalen = 500
indices = np.arange(datalen)
col_a = np.arange(0, 10, 10/datalen)
col_b = np.random.randint(3, 8, datalen)
data = np.array([indices, col_a, col_b]).T
df = pd.DataFrame(data=data, columns=['idx', 'a', 'b'])

# draw
import random
point_markers = {
    'a': [(indices[i], col_a[i]) for i in random.sample(list(indices), 20)]
}
prda.graphic.lineplot_html(df, x='idx', y=['a', 'b'], markpoints=point_markers, filepath='demo_lineplot.html')
```
|     |  idx  |   a  |   b  |
|:---:|:-----:|:----:|:----:|
|  0  |  0.0  | 0.00 |  6.0 |
|  1  |  1.0  | 0.02 | 3.0 |
|  2  |  2.0  | 0.04 | 4.0 |
| ... |  ...  |  ... |  ... |
| 498 | 498.0 | 9.96 | 6.0 |
| 499 | 499.0 | 9.98 | 5.0 |

And code with the above DataFrame will draw anther plot look like this:

<!-- <iframe src="demo/demo_lineplot.html" width="500" height="300"></iframe> -->

![lineplot_screenshot.png](demo/demo_lineplot_screenshot.png)

[demo_lineplot.html](/demo/demo_lineplot.html)

---
2. ### For Data Preparation
Code for filtering continuous variables in data with unique-value threshold of 5:
```python
from prda import prep
prep.select_continuous_variables(data, unique_threshold=5)
```

3. ### For Machine Learning
Code for evaluating hyperparameters combinations for a given algorithm using user-specified cross-validation method:
```python
from prda.ml import evaluations
param_grid = {'k': [4,5,6,7]}
evaluations.evaluate_param_combinations(X, y, knn_algorithm, param_grid=param_grid, cv=10, visualize_results=True)
```

4. ### For IO
A common usage during my research practice is to make well structured folders to save experimential results. With the following function, you only need to think about how you want your files to be structured. All `related folders` will be created automatically:
```python
from prda import iostream
iostream.create_dirs([
    'results/experiment1/f1_score.csv',
    'results/experiment1/accuracy.csv',
    'results/experiment2/',
    'results/experiment10/accuracy/',
    'results/experiment10/f1_score/r1.txt',
    ])
```
The above one-line code will create all the folders for you which will have the corresponding structure below, after which you can then store your results without worrying about file structures whatsoever.

```
results/
├── experiment1/
│   ├── f1_score.csv
│   └── accuracy.csv
├── experiment2/
└── experiment10/
    ├── accuracy/
    └── f1_score/
        └── r1.txt
```


The `prda`'s methods are quite self-explanatory, as a result, we think providing the above demonstration is suffice at the moment. Although the current *prda* is far from completion, let along perfection. It is under improvement regularly.

----
# Updates
## 2023.5.3 Major Updates
Add several easy-to-use functions, including `prep::`pca, select_continuous_variables, handle_missing_data, apply_linear_func(row-wisely), and `ml::`match_clusters, evaluate_param_combinations(optimal parameters searching, with base class::sklearn.base.BaseEstimator), etc.

## 2023.11.10 Major Updates
1. Including a variant of kNN which allows you to **allocate customized k** (K sequence) for each sample in `ml::neighbors`::VariableKNN. The algorithm behaves as a `sklearn.classifier` which means you can employ it directly via `fit(·) and predict(·)`. (Originated from my work: https://arxiv.org/abs/2308.02442)
2. Add functions, e.g. `iostream::`create_dirs.
