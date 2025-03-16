"""
Content Generator Module

This module is responsible for generating Q&A content based on input texts using
LangChain and various LLM backends. It handles the interaction with the AI model and
provides methods to generate and save the generated content.
"""

import os
from typing import Optional, Dict, Any, List
import re


from langchain_community.chat_models import ChatLiteLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms.llamafile import Llamafile
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from podcastfy.utils.config_conversation import load_conversation_config
from podcastfy.utils.config import load_config
import logging
from langchain.prompts import HumanMessagePromptTemplate
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMBackend:
    def __init__(
        self,
        is_local: bool,
        temperature: float,
        max_output_tokens: int,
        model_name: str,
        api_key_label: str = "GEMINI_API_KEY",
    ):
        """
        Initialize the LLMBackend.

        Args:
                is_local (bool): Whether to use a local LLM or not.
                temperature (float): The temperature for text generation.
                max_output_tokens (int): The maximum number of output tokens.
                model_name (str): The name of the model to use.
        """
        self.is_local = is_local
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.model_name = model_name
        self.is_multimodal = not is_local  # Does not assume local LLM is multimodal

        common_params = {
            "temperature": temperature,
            "presence_penalty": 0.75,  # Encourage diverse content
            "frequency_penalty": 0.75,  # Avoid repetition
        }

        if is_local:
            self.llm = Llamafile() # replace with ollama
        elif (
            "gemini" in self.model_name.lower()
        ):  # keeping original gemini as a special case while we build confidence on LiteLLM

            self.llm = ChatGoogleGenerativeAI(
                api_key=os.environ["GEMINI_API_KEY"],
                model=model_name,
                max_output_tokens=max_output_tokens,
                **common_params,
            )
        else:  # user should set api_key_label from input
            self.llm = ChatLiteLLM(
                model=self.model_name,
                temperature=temperature,
                api_key=os.environ[api_key_label],
            )