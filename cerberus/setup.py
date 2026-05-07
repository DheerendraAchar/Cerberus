from setuptools import setup, find_packages

setup(
    name='cerberus',
    version='1.0.0',
    description='Adversarial robustness framework',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'matplotlib',
        'scikit-learn',
    ],
    python_requires='>=3.8',
)
