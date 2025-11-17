#!/usr/bin/env python3
"""
Test script to verify rate limiting functionality
"""

import sys
import os
import time
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from improved_automated_test_generator import ImprovedAutomatedTestGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_rate_limiting():
    """Test the rate limiting functionality."""
    try:
        # Initialize the generator
        generator = ImprovedAutomatedTestGenerator()
        
        # Test with a simple prompt
        test_prompt = "Generate a simple test method for a Unity MonoBehaviour class."
        
        logger.info("Testing rate limiting with a simple API call...")
        
        # Make a test API call
        start_time = time.time()
        result = generator._make_api_call_with_retry(test_prompt, max_tokens=500)
        end_time = time.time()
        
        logger.info(f"API call completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Result length: {len(result)} characters")
        logger.info(f"Token usage this minute: {generator.token_usage_per_minute}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting rate limiting test...")
    success = test_rate_limiting()
    
    if success:
        logger.info("✅ Rate limiting test passed!")
    else:
        logger.error("❌ Rate limiting test failed!")
        sys.exit(1)
