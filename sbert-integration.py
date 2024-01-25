from typing import Type, List

from cat.mad_hatter.decorators import tool, hook, plugin
from cat.factory.embedder import EmbedderSettings
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import ConfigDict


class SBERTEmbedderConfig(EmbedderSettings):
    model_name: str='paraphrase-multilingual-MiniLM-L12-v2'
    _pyclass: Type = HuggingFaceEmbeddings

    model_config = ConfigDict(
        json_schema_extra = {
            "humanReadableName": "SBERT embedder",
            "description": "Sentence Transformers Embedder",
            "link": "https://www.sbert.net/index.html",
        }
    )


@hook
def factory_allowed_embedders(allowed, cat) -> List:
    allowed.append(SBERTEmbedderConfig)
    return allowed