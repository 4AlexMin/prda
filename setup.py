from setuptools import setup, find_packages

with open('LICENSE', 'r') as f:
    license = f.read()


setup(
    name='prda',
    version='1.0.2',
    author='Shaojie Min',
    author_email='alexmin@cqu.edu.cn',
    description='Prda contains packages for data processing, analysis and visualization. The ultimate goal is to fill the “last mile” between analysts and packages.',
    url='https://github.com/4AlexMin/prda',
    license=license,
    packages=find_packages(),
    install_requires=[
        'scipy',
        'scikit-learn',
        'pyspark',
        'lifelines',
        'seaborn',
        'pyecharts',
        'numpy',
        'pandas',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
