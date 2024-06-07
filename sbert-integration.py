from typing import Type, List, Dict, Any

from cat.mad_hatter.decorators import tool, hook, plugin
from cat.factory.embedder import EmbedderSettings
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import ConfigDict

import os
import json
def get_settings():
    if os.path.isfile("cat/plugins/sbert-integration/settings.json"):
        with open("cat/plugins/sbert-integration/settings.json", "r") as json_file:
            settings = json.load(json_file)
    return settings

class SBERTEmbedderConfig(EmbedderSettings):
    model_name: str='sentence-transformers/all-MiniLM-L6-v2'
    cache_folder: str = "cat/data/models/sbert"
    _pyclass: Type = HuggingFaceEmbeddings

    model_config = ConfigDict(
        json_schema_extra = {
            "humanReadableName": "SBERT embedder",
            "description": "Sentence Transformers Embedder",
            "link": "https://www.sbert.net/index.html",
        }
    )
settings = get_settings()
print(settings)
global settings
class MatryoshkaSBERTEmbedderConfig(EmbedderSettings):

    model_name: str='mixedbread-ai/mxbai-embed-large-v1'
    cache_folder: str = "cat/data/models/sbert"
    model_kwargs: Dict[str, Any] = {"truncate_dim":settings['truncate_dim']}
    _pyclass: Type = HuggingFaceEmbeddings

    model_config = ConfigDict(
        json_schema_extra = {
            "humanReadableName": "Matryoshka SBERT embedder",
            "description": "Matryoshka Sentence Transformers Embedder",
            "link": "https://www.sbert.net/index.html",
        }
    )


@hook
def factory_allowed_embedders(allowed, cat) -> List:
    allowed.append(SBERTEmbedderConfig)
    allowed.append(MatryoshkaSBERTEmbedderConfig)
    return allowed