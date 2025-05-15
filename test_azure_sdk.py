import os
import toml
import logging
from azure_llm_handler import AzureFoundryHandler

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Load credentials from .streamlit/secrets.toml
    try:
        secrets = toml.load('.streamlit/secrets.toml')
        api_key = secrets.get('AZURE_AI_FOUNDRY_API_KEY')
        endpoint = secrets.get('AZURE_AI_FOUNDRY_ENDPOINT')
        deployment_name = secrets.get('AZURE_AI_FOUNDRY_DEPLOYMENT_NAME', 'Phi-3.5-vision-instruct-deploy')
    except Exception as e:
        logger.error(f"Error loading secrets: {str(e)}")
        return

    # Create handler
    handler = AzureFoundryHandler(
        api_key=api_key,
        endpoint=endpoint,
        deployment_name=deployment_name
    )
    
    # Test connection
    logger.info("Testing connection to Azure AI service...")
    if handler.check_connection():
        logger.info("✅ Connection successful!")
        
        # Test generation with a simple prompt
        logger.info("Testing generation with a simple prompt...")
        prompt = "Generate a simple test case for a login feature."
        
        def callback(chunk):
            print(chunk, end='', flush=True)
        
        print("\nResponse: ", end='')
        response = handler.generate_test_cases(prompt, callback=callback)
        print("\n")
        
        logger.info("✅ Test completed!")
    else:
        logger.error("❌ Connection failed!")

if __name__ == "__main__":
    main()