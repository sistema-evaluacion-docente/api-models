"""
NLP module for sentiment and hate speech analysis using pysentimiento.
"""

from pysentimiento import create_analyzer
from transformers import pipeline

sentiment_analyzer = create_analyzer(task="sentiment", lang="es")
hate_analyzer = create_analyzer(task="hate_speech", lang="es")

danger_analyzer = pipeline(
    "text-classification", model="DevOB/modelo-distilbeto-nivelDeRiesgo5", tokenizer="DevOB/modelo-distilbeto-nivelDeRiesgo5"
)

categories_analyzer = pipeline(
    "text-classification", model="DevOB/modelo-distilbeto-categorias-3", tokenizer="DevOB/modelo-distilbeto-categorias-3"
)
