from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Get your Azure credentials
api_key = os.getenv('AZURE_OPENAI_API_KEY')
endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', 'https://nidhi-mhaa1721-eastus2.cognitiveservices.azure.com/')
deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o-mini')
api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')

print(f"🔵 Testing Azure OpenAI Connection...")
print(f"   Endpoint: {endpoint}")
print(f"   Deployment: {deployment}")
print(f"   API Version: {api_version}")
print(f"   API Key: {'SET' if api_key else 'NOT SET'}")

try:
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint
    )
    
    response = client.chat.completions.create(
        model=deployment,  # Use deployment name!
        messages=[{"role": "user", "content": "Say 'Azure works!'"}],
        max_tokens=10
    )
    
    print(f"\n✅ CONNECTION SUCCESSFUL!")
    print(f"   Response: {response.choices[0].message.content}")
    print(f"   Tokens used: {response.usage.total_tokens}")
    print(f"\n🎉 Your Azure OpenAI is ready to use!")
    
except Exception as e:
    print(f"\n❌ CONNECTION FAILED!")
    print(f"   Error: {e}")
    print(f"\n💡 Troubleshooting:")
    print(f"   1. Check your .env file has all 4 variables set")
    print(f"   2. Verify API key from Azure Portal → Keys and Endpoint")
    print(f"   3. Ensure deployment name matches what you created in Step 3")