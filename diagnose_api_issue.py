#!/usr/bin/env python3
"""
Diagnostic script to check OpenAI API key status and account information
"""

import os
import sys
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_api_key_status():
    """Check the current API key status and account information."""
    try:
        import openai
        
        # Use the hardcoded key from your files
        api_key = ""
        
        # Set the API key
        openai.api_key = api_key
        
        logger.info("🔍 Checking API key status...")
        
        # Try a simple API call to check status
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello, this is a test."}],
                max_tokens=10
            )
            
            logger.info("✅ API key is working!")
            logger.info(f"Response: {response.choices[0].message.content}")
            
            # Check usage information if available
            if hasattr(response, 'usage') and response.usage:
                logger.info(f"Token usage: {response.usage.total_tokens}")
            
            return True
            
        except Exception as api_error:
            error_str = str(api_error)
            logger.error(f"❌ API call failed: {api_error}")
            
            if "rate_limit_exceeded" in error_str or "429" in error_str:
                logger.error("🚨 RATE LIMIT EXCEEDED - This is the issue!")
                logger.error("Possible causes:")
                logger.error("1. Organization rate limit exceeded")
                logger.error("2. Account credits depleted")
                logger.error("3. API key has been rate limited")
                
                # Extract retry time if available
                if "try again in" in error_str:
                    import re
                    retry_match = re.search(r'try again in ([^.]*)', error_str)
                    if retry_match:
                        logger.error(f"Retry time: {retry_match.group(1)}")
                
            elif "insufficient_quota" in error_str or "quota" in error_str:
                logger.error("💰 INSUFFICIENT QUOTA - Add credits to your account")
            elif "invalid_api_key" in error_str:
                logger.error("🔑 INVALID API KEY - Check your API key")
            else:
                logger.error(f"Other error: {error_str}")
            
            return False
            
    except ImportError:
        logger.error("❌ OpenAI package not installed. Run: pip install openai")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

def suggest_solutions():
    """Suggest solutions for the API issues."""
    logger.info("\n" + "="*60)
    logger.info("🔧 SUGGESTED SOLUTIONS:")
    logger.info("="*60)
    
    logger.info("\n1. 💰 CHECK YOUR ACCOUNT CREDITS:")
    logger.info("   - Go to: https://platform.openai.com/account/billing")
    logger.info("   - Add payment method if not already added")
    logger.info("   - Add credits to your account")
    
    logger.info("\n2. 🔄 WAIT FOR RATE LIMIT RESET:")
    logger.info("   - Organization rate limits can take 24+ hours to reset")
    logger.info("   - Check: https://platform.openai.com/account/rate-limits")
    
    logger.info("\n3. 🔑 GET A NEW API KEY:")
    logger.info("   - Go to: https://platform.openai.com/api-keys")
    logger.info("   - Create a new API key")
    logger.info("   - Update your code to use the new key")
    
    logger.info("\n4. 🏢 CHECK ORGANIZATION LIMITS:")
    logger.info("   - If using an organization, check org-level limits")
    logger.info("   - Consider switching to personal account")
    
    logger.info("\n5. 📊 MONITOR USAGE:")
    logger.info("   - Check: https://platform.openai.com/account/usage")
    logger.info("   - Monitor your token consumption")

if __name__ == "__main__":
    logger.info("🚀 Starting OpenAI API diagnostic...")
    logger.info("="*60)
    
    success = check_api_key_status()
    
    if not success:
        suggest_solutions()
    
    logger.info("\n" + "="*60)
    if success:
        logger.info("✅ API is working! The rate limiting code should help prevent future issues.")
    else:
        logger.info("❌ API has issues. Please follow the suggested solutions above.")
