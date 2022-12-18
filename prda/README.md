# prda

## Prda contains packages for data processing, analysis and visualization.

Prda ultimate goal is to **fill the “last mile” between analysts and packages**. During my research practice, I have felt how “learning a package before utilizing” can be time-consuming and exhausting. The resulted inefficiency leads to the creation of *prda*.

To utilize prda, you only need to be familiar with `pandas` as most inputs is `pd.DataFrame`.

For example, the following codes will provide an interactive html figure.

```other
import prda
import pandas as pd
import numpy as np
df = pd.DataFrame(data=np.array([np.arange(100) for i in range(5)]).T,columns=['a', 'b', 'c', 'd', 'e'])
prda.graphic.scatter_3d_html(df, x='a', y='b', z='c', color_hue='d', size_hue='e', title='demo_3d_scatter')
```

[demo_3d_scatter.html](https://res.craft.do/user/full/5c4db410-b4e7-c66b-4069-b35740ab0329/doc/E1EE1C20-1BDD-47E0-9B2D-3AA3212F2C3D/F39EBB0D-6597-4D3E-9667-A8514131B51B_2/LK7QIqFxDhOeZDvRbI3ZbS38hwIsk8tx1bN920mxLx0z/demo_3d_scatter.html)

Although the current *prda* is far from completion, let along perfection. It is under improvement regularly.

You are welcome to clone *prda* for personal use (to use, simply add the folder to your system path) and **pull request** of your modification is super!! encouraged.

