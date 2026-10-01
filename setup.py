from setuptools import setup, find_packages

setup(
    name='quant-validator',
    version='0.3.0',
    description='Rigorous validation toolkit for quantitative trading',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=['numpy>=1.20', 'pandas>=1.3', 'scipy>=1.7'],
)
