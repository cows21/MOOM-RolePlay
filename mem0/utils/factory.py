import importlib

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.embeddings.base import BaseEmbedderConfig


def load_class(class_type):
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class LlmFactory:
    provider_to_class = {
        "ollama": "mem0.llms.ollama.OllamaLLM",
        "openai": "mem0.llms.openai.OpenAILLM",
        "groq": "mem0.llms.groq.GroqLLM",
        "together": "mem0.llms.together.TogetherLLM",
        "aws_bedrock": "mem0.llms.aws_bedrock.AWSBedrockLLM",
        "litellm": "mem0.llms.litellm.LiteLLM",
        "azure_openai": "mem0.llms.azure_openai.AzureOpenAILLM",
        "openai_structured": "mem0.llms.openai_structured.OpenAIStructuredLLM",
        "interface_openai": "mem0.llms.interface_openai.Interface_OpenAI",
        "qwen2": "mem0.llms.qwen2.Qwen2",
        "qwen72b_api": "mem0.llms.qwen72b_api.Qwen72B_api"
    }

    local_model_name = ["qwen2"]

    @classmethod
    def create(cls, provider_name, config, local_model=None):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            if provider_name not in cls.local_model_name:
                llm_instance = load_class(class_type)
                base_config = BaseLlmConfig(**config)
                return llm_instance(base_config)
            else:
                llm_instance = load_class(class_type)
                base_config = BaseLlmConfig(**config)
                return llm_instance(base_config, local_model)

        else:
            raise ValueError(f"Unsupported Llm provider: {provider_name}")


class EmbedderFactory:
    provider_to_class = {
        "openai": "mem0.embeddings.openai.OpenAIEmbedding",
        "ollama": "mem0.embeddings.ollama.OllamaEmbedding",
        "huggingface": "mem0.embeddings.huggingface.HuggingFaceEmbedding",
        "azure_openai": "mem0.embeddings.azure_openai.AzureOpenAIEmbedding",
        "interface_openai": "mem0.embeddings.interface_openai.Interface_OpenAIEmbedding",
        "text2vec_large_chinese": "mem0.embeddings.text2vec_large_chinese.Text2vec_Large_ChineseEmbedding"
    }

    local_model_name = ["text2vec_large_chinese"]

    @classmethod
    def create(cls, provider_name, config, local_model=None):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            if provider_name not in cls.local_model_name:
                embedder_instance = load_class(class_type)
                base_config = BaseEmbedderConfig(**config)
                return embedder_instance(base_config)
            else:
                embedder_instance = load_class(class_type)
                base_config = BaseEmbedderConfig(**config)
                return embedder_instance(base_config, local_model)

        else:
            raise ValueError(f"Unsupported Embedder provider: {provider_name}")


class VectorStoreFactory:
    provider_to_class = {
        "qdrant": "mem0.vector_stores.qdrant.Qdrant",
        "chroma": "mem0.vector_stores.chroma.ChromaDB",
        "pgvector": "mem0.vector_stores.pgvector.PGVector",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            if not isinstance(config, dict):
                config = config.model_dump()
            vector_store_instance = load_class(class_type)
            return vector_store_instance(**config)
        else:
            raise ValueError(f"Unsupported VectorStore provider: {provider_name}")
