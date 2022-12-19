# prda

## Prda contains packages for data processing, analysis and visualization.

Prda ultimate goal is to **fill the “last mile” between analysts and packages**. During my research practice, I have felt how “learning a package before utilizing” can be time-consuming and exhausting. The resulted inefficiency leads to the creation of *prda*.

To utilize prda, you only need to be familiar with `pandas` as most inputs is `pd.DataFrame`.

## For example：

```python
import prda
import pandas as pd
import numpy as np
df = pd.DataFrame(data=np.array([np.arange(100) for i in range(5)]).T,columns=['a', 'b', 'c', 'd', 'e'])
prda.graphic.scatter_3d_html(df, x='a', y='b', z='c', color_hue='d', size_hue='e', title='demo_3d_scatter', filepath='demo_3d_scatter.html')
```

the above code will provide an interactive html figure that look like this:

![Image.png](/demo/3d_scatter_screenshot.png)

[demo_3d_scatter.html](/demo/demo_3d_scatter.html)

----

```python
import prda
import pandas as pd
import numpy as np
datalen = 500
indices = np.arange(datalen)
col_a = np.arange(0, 10, 10/datalen)
col_b = np.random.randint(0, 15, datalen)
data = np.array([indices, col_a, col_b]).T
df = pd.DataFrame(data=data, columns=['idx', 'a', 'b'])

# draw
import random
point_markers = {
    'a': [(indices[i], col_a[i]) for i in random.sample(list(indices), 20)]
}
prda.graphic.lineplot_html(df, x='idx', y=['a', 'b'], markpoints=point_markers, filepath='demo_lineplot.html')
```

And code with the below DataFrame will draw anther plot look like this:

![Image.png](/demo/demo_lineplot_df.png)

![lineplot_screenshot.png](demo/lineplot_screenshot.png)

[demo_lineplot.html](/demo/demo_lineplot.html)

----

Although the current *prda* is far from completion, let along perfection. It is under improvement regularly.

You are welcome to clone *prda* for personal use (to use, simply add the folder to your system path) and **pull request** of your modification is super!! encouraged.

