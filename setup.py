from setuptools import setup, find_packages

setup(
    name="rice-production-analyzer",
    version="1.0.0",
    description="Análise preditiva de produção de arroz usando ML",
    author="Seu Nome",
    author_email="seu.email@domain.com",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "plotly>=5.15.0",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
)