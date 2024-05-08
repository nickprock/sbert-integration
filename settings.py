from cat.mad_hatter.decorators import plugin
from pydantic import BaseModel, Field
from typing import Dict, Literal


class MySettings(BaseModel):
    truncate_dim: int = 512


@plugin
def settings_model():
    return MySettings