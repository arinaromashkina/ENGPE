from setuptools import setup, find_packages

setup(
    name='engpe',
    version='1.0.0',
    description='Empirical Null Generation-based Performance Evaluation',
    author='Arina Romashkina, Viktoria Fokina, Attila Kertesz-Farkas',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scipy>=1.10.0',
        'tqdm>=4.65.0',
    ],
)
