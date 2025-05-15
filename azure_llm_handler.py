import os
import logging
import traceback
from typing import Dict, List, Tuple, Any, Optional, Callable, Union

# Import Azure SDK packages
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ServiceRequestError, HttpResponseError

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AzureFoundryHandler:
    """
    Class for handling interactions with Azure AI models using the Azure AI Inference SDK.
    Provides methods for test case generation with both streaming and non-streaming options.
    """
    
    def __init__(
        self, 
        api_key: str = None,
        endpoint: str = None,
        deployment_name: str = "Phi-3.5-vision-instruct-deploy",
        temperature: float = 0.1, 
        max_tokens: int = 4096,
        top_p: float = 0.95,
        **kwargs
    ):
        """
        Initialize the Azure AI handler.
        
        Args:
            api_key: Azure API key
            endpoint: Azure endpoint
            deployment_name: Name of the deployment
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            **kwargs: Additional parameters for the model
        """
        self.api_key = api_key or os.environ.get("AZURE_AI_FOUNDRY_API_KEY")
        self.endpoint = endpoint or os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT")
        self.deployment_name = deployment_name
        
        if not self.api_key or not self.endpoint:
            logger.warning("Azure API key or endpoint not provided. Please set AZURE_AI_FOUNDRY_API_KEY and AZURE_AI_FOUNDRY_ENDPOINT environment variables or pass them directly.")
            
        self.model_params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            **kwargs
        }
        
        # Initialize the client to None, will create when needed
        self.client = None
        
    def _get_client(self):
        """
        Create or return the ChatCompletionsClient
        
        Returns:
            ChatCompletionsClient instance
        """
        if self.client is None:
            try:
                self.client = ChatCompletionsClient(
                    endpoint=self.endpoint,
                    credential=AzureKeyCredential(self.api_key)
                )
                logger.info(f"Created ChatCompletionsClient with endpoint: {self.endpoint}")
            except Exception as e:
                logger.error(f"Failed to create ChatCompletionsClient: {str(e)}")
                raise
                
        return self.client
    
    def check_connection(self) -> bool:
        """Check if the Azure AI service is accessible."""
        if not self.api_key or not self.endpoint:
            logger.error("Azure client not initialized - missing API key or endpoint")
            return False
            
        try:
            # Create a client
            client = self._get_client()
            
            # Try a simple completion to check connection
            logger.info(f"Testing connection with model: {self.deployment_name}")
            response = client.complete(
                messages=[
                    SystemMessage(content="You are a helpful assistant."),
                    UserMessage(content="Hello, are you working? Just reply with yes or no.")
                ],
                max_tokens=10,
                temperature=0,
                model=self.deployment_name
            )
            
            # Extract response content
            content = response.choices[0].message.content
            logger.info(f"Connection successful. Response: {content[:50]}...")
            return True
            
        except HttpResponseError as http_err:
            logger.error(f"HTTP error during connection check: {str(http_err)}")
            if hasattr(http_err, 'status_code'):
                logger.error(f"Status code: {http_err.status_code}")
            if hasattr(http_err, 'response') and hasattr(http_err.response, 'text'):
                logger.error(f"Response text: {http_err.response.text[:500]}")
            return False
            
        except ServiceRequestError as req_err:
            logger.error(f"Service request error during connection check: {str(req_err)}")
            return False
            
        except Exception as e:
            logger.error(f"Connection check failed with unexpected error: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def generate_test_cases(
        self, 
        prompt: str, 
        callback: Optional[Callable[[str], None]] = None, 
        timeout: int = 300,
        max_retries: int = 2
    ) -> str:
        """
        Generate test cases using Azure AI with streaming capability.
        
        Args:
            prompt: The input prompt for test case generation
            callback: Optional function to call with each response chunk
            timeout: Request timeout in seconds (default: 300)
            max_retries: Maximum number of retries on failure (default: 2)
            
        Returns:
            str: The complete generated text
        """
        if not self.api_key or not self.endpoint:
            error_msg = "Azure client not initialized. Check API key and endpoint."
            logger.error(error_msg)
            return error_msg
        
        # Get client
        client = self._get_client()
        
        # Start with retry count at 0    
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                # Prepare messages
                messages = [
                    SystemMessage(content="You are a specialized test case generator that creates high-quality, comprehensive test cases from user stories."),
                    UserMessage(content=prompt)
                ]
                
                # Get parameters from model_params
                params = {
                    "messages": messages,
                    "temperature": self.model_params.get("temperature", 0.1),
                    "max_tokens": self.model_params.get("max_tokens", 4096),
                    "top_p": self.model_params.get("top_p", 0.95),
                    "model": self.deployment_name
                }
                
                # Add streaming if callback provided
                if callback:
                    params["stream"] = True
                
                logger.info(f"Generating test cases with model: {self.deployment_name}")
                logger.info(f"Parameters: temperature={params['temperature']}, max_tokens={params['max_tokens']}")
                
                # Initialize accumulated response
                full_response = ""
                
                # Use streaming for real-time updates if callback is provided
                if callback:
                    # Handle streaming response
                    response = client.complete(**params)
                    
                    for update in response:
                        if update.choices:
                            content = update.choices[0].delta.content or ""
                            full_response += content
                            callback(content)
                else:
                    # Handle non-streaming response
                    response = client.complete(**params)
                    full_response = response.choices[0].message.content
                
                # Log success metrics
                logger.info(f"Successfully generated response with {len(full_response)} characters")
                return full_response
                
            except HttpResponseError as http_err:
                logger.error(f"HTTP error during generation: {str(http_err)}")
                if hasattr(http_err, 'status_code'):
                    logger.error(f"Status code: {http_err.status_code}")
                if hasattr(http_err, 'response') and hasattr(http_err.response, 'text'):
                    logger.error(f"Response text: {http_err.response.text[:500]}")
                
                # Handle retry for certain errors
                if (hasattr(http_err, 'status_code') and 
                    (http_err.status_code == 429 or http_err.status_code >= 500)):
                    retry_count += 1
                    if retry_count <= max_retries:
                        logger.warning(f"Retrying... (Attempt {retry_count}/{max_retries})")
                        continue
                    
                return f"Error: HTTP error {http_err.status_code if hasattr(http_err, 'status_code') else 'unknown'}: {str(http_err)}"
                
            except Exception as e:
                error_msg = f"Error generating test cases: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                
                retry_count += 1
                if retry_count <= max_retries:
                    logger.warning(f"Retrying... (Attempt {retry_count}/{max_retries})")
                    continue
                
                return error_msg
        
        return "Failed to generate response after all retry attempts"
    
    def update_params(self, **kwargs) -> None:
        """
        Update model parameters.
        
        Args:
            **kwargs: The parameters to update (temperature, max_tokens, etc.)
        """
        for key, value in kwargs.items():
            self.model_params[key] = value
                
        logger.info(f"Updated model parameters: {self.model_params}")