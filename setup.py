from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('LICENSE', 'r') as f:
    license = f.read()

setup(
    name='prda',
    version='1.2.0',
    author='Shaojie Min',
    author_email='alexmin@cqu.edu.cn',
    description='Prda contains packages for data processing, analysis and visualization. The ultimate goal is to fill the “last mile” between analysts and packages.',
    long_description=long_description,  # Use the contents of README.md as the project description
    long_description_content_type='text/markdown',  # Specify the type of long description
    readme='README.md',
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
