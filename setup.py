from setuptools import setup, find_packages

setup(
    name="unlearn_ai",
    version="0.1.0",
    description="A machine unlearning toolkit and benchmark for language models",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.6.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "evaluate>=0.4.0",
        "rouge-score>=0.1.2",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
    ],
)
