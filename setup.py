from setuptools import setup, find_packages

setup(
    name="speechscore",
    version="2.0.0-alpha",
    description=(
        "SpeechScore 2.0: Temporal-Adaptive Multi-Modal "
        "Communication Assessment Framework"
    ),
    author="SpeechScore Research",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "openai-whisper>=20231117",
        "torch>=2.1.0",
        "librosa>=0.10.1",
        "soundfile>=0.12.1",
        "praat-parselmouth>=0.4.3",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "spacy>=3.7.0",
        "language-tool-python>=2.7.1",
        "pydantic>=2.5.0",
    ],
)
