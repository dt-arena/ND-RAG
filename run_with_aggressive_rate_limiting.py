#!/usr/bin/env python3
"""
Run the improved test generator with aggressive rate limiting
"""

import os
import sys
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

def run_with_aggressive_limits():
    """Run the test generator with very conservative rate limits."""
    try:
        # Initialize with very conservative settings
        generator = ImprovedAutomatedTestGenerator()
        
        # Override with very aggressive rate limiting
        generator.max_requests_per_minute = 10  # Very conservative
        generator.max_tokens_per_minute = 20000  # Very conservative
        generator.base_delay = 5.0  # Longer delays
        
        logger.info("🚀 Starting with AGGRESSIVE rate limiting...")
        logger.info(f"Max requests per minute: {generator.max_requests_per_minute}")
        logger.info(f"Max tokens per minute: {generator.max_tokens_per_minute}")
        logger.info(f"Base delay: {generator.base_delay} seconds")
        
        # Test with a simple function
        test_function = """
        public void TestMethod()
        {
            Debug.Log("Hello World");
        }
        """
        
        logger.info("🧪 Testing with a simple function...")
        
        # This will use the rate-limited API call
        result = generator._make_api_call_with_retry(
            f"Generate a test for this Unity method: {test_function}",
            max_tokens=1000
        )
        
        logger.info("✅ Test completed successfully!")
        logger.info(f"Generated result length: {len(result)} characters")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🔧 Running with AGGRESSIVE rate limiting to avoid 429 errors...")
    success = run_with_aggressive_limits()
    
    if success:
        logger.info("✅ Success! The aggressive rate limiting worked.")
        logger.info("You can now run your main test generation with these settings.")
    else:
        logger.error("❌ Still getting errors. You may need to:")
        logger.error("1. Wait longer for rate limits to reset")
        logger.error("2. Get a new API key")
        logger.error("3. Add credits to your account")
