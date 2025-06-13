"""
Conversation management for Notebook-RAG application.
"""

import os
from typing import Optional
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import requests

from .vector_store_manager import VectorStoreManager
from .prompt_builder import PromptBuilder
from .config_manager import ConfigManager

class ConversationManager:
    """Class for managing conversations with documents."""
    
    @staticmethod
    def get_llm(model_name: Optional[str] = None):
        """
        Get a language model instance based on the configuration.
        
        Args:
            model_name: Optional model name to use. If not provided, will use the one from config.
            
        Returns:
            Language model instance.
        """
        # Get config
        app_config = ConfigManager.get_app_config()
        llm_config = app_config.get("llm", {})
        
        # Get provider and model
        provider = llm_config.get("provider", "groq")
        if not model_name:
            model_name = llm_config.get("model", "meta-llama/llama-4-scout-17b-16e-instruct")
        
        # Get provider-specific configurations
        providers_config = app_config.get("providers", {})
        
        if provider == "groq":
            return ChatGroq(model=model_name)
        
        elif provider == "gemini":
            return ChatGoogleGenerativeAI(model=model_name)
        
        elif provider == "ollama":
            host = providers_config.get("ollama", {}).get("host", "http://localhost:11434")
            
            # Custom implementation for Ollama
            class OllamaChat:
                def __init__(self, model, host):
                    self.model = model
                    self.host = host
                
                def invoke(self, prompt):
                    # Extract prompt content
                    if isinstance(prompt, list):
                        # Handle list of messages
                        messages = []
                        for msg in prompt:
                            if isinstance(msg, SystemMessage):
                                messages.append({"role": "system", "content": msg.content})
                            elif isinstance(msg, HumanMessage):
                                messages.append({"role": "user", "content": msg.content})
                            else:
                                messages.append({"role": "assistant", "content": msg.content})
                        
                        payload = {
                            "model": self.model,
                            "messages": messages
                        }
                    else:
                        # Handle string prompt
                        payload = {
                            "model": self.model,
                            "prompt": prompt
                        }
                    
                    # Make API call
                    response = requests.post(f"{self.host}/api/chat", json=payload)
                    response.raise_for_status()
                    result = response.json()
                    
                    # Create a response object with content attribute
                    class Response:
                        def __init__(self, content):
                            self.content = content
                    
                    return Response(result.get("message", {}).get("content", ""))
            
            return OllamaChat(model=model_name, host=host)
        
        elif provider == "lmstudio":
            host = providers_config.get("lmstudio", {}).get("host", "http://localhost:1234/v1")
            
            # Use OpenAI client with custom base URL for LM Studio
            return ChatOpenAI(model=model_name, base_url=host)
        
        else:
            raise Exception("Invalid LLM provider")
    
    @staticmethod
    def respond_to_query(
        notebook_name: str,
        query: str,
        n_results: int = 5,
        threshold: float = 0.3,
        model_name: Optional[str] = None,
    ) -> str:
        """
        Respond to a query using RAG.
        
        Args:
            notebook_name: Name of the notebook to query.
            query: Query text.
            n_results: Number of results to retrieve.
            threshold: Similarity threshold.
            model_name: Optional model name to use.
            
        Returns:
            Response text.
            
        Raises:
            FileNotFoundError: If the notebook does not exist.
        """
        # Retrieve relevant documents
        relevant_documents = VectorStoreManager.retrieve_relevant_documents(
            notebook_name=notebook_name,
            query=query,
            n_results=n_results,
            threshold=threshold,
        )
        
        # If no relevant documents were found
        if not relevant_documents:
            return "I couldn't find any relevant information in this notebook to answer your question."
        
        # Get prompt config
        prompt_config = ConfigManager.get_prompt_config()
        rag_assistant_prompt = prompt_config.get("rag_assistant_prompt", {})
        
        # Prepare input data
        input_data = (
            f"Relevant documents:\n\n{relevant_documents}\n\nUser's question:\n\n{query}"
        )
        
        # Build prompt
        prompt = PromptBuilder.build_prompt_from_config(
            config=rag_assistant_prompt,
            input_data=input_data,
        )
        
        # Get LLM
        llm = ConversationManager.get_llm(model_name)
        
        # Generate response
        response = llm.invoke(prompt)
        
        return response.content
    
    @staticmethod
    def create_system_prompt(notebook_name: str) -> str:
        """
        Create a system prompt for a notebook.
        
        Args:
            notebook_name: Name of the notebook.
            
        Returns:
            System prompt text.
        """
        prompt_config = ConfigManager.get_prompt_config()
        system_prompt_config = prompt_config.get("ai_assistant_system_prompt_advanced", {})
        
        return PromptBuilder.build_system_prompt_from_config(
            config=system_prompt_config,
            document_content=f"You are assisting with the notebook '{notebook_name}'."
        )
