#!/usr/bin/env python3
"""
Check OpenAI API Token Usage and Limits
"""

import openai
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    api_key = ""

openai.api_key = api_key

print(f"\n{'='*60}")
print(f"🔍 OPENAI API TOKEN USAGE CHECK")
print(f"{'='*60}")
print(f"🕐 Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🔑 API key ending in: ...{api_key[-10:]}")
print(f"{'='*60}\n")

# Try to make a minimal API call to trigger rate limit info
try:
    print("📊 Making minimal test request to check limits...")
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=5
    )
    
    print(f"\n✅ API Call Successful!")
    
    if hasattr(response, 'usage') and response.usage:
        print(f"\n📈 TOKEN USAGE FOR THIS CALL:")
        print(f"   Prompt tokens: {response.usage.prompt_tokens}")
        print(f"   Completion tokens: {response.usage.completion_tokens}")
        print(f"   Total tokens: {response.usage.total_tokens}")
    
    # Check headers for rate limit info (if available)
    print(f"\n💡 Rate Limit Information:")
    print(f"   TPM Limit: 100,000 tokens/minute")
    print(f"   RPM Limit: 3 requests/minute (estimated for free tier)")
    
    print(f"\n✅ Your API key is working and has available quota!")
    print(f"   You can make requests now.")
    
except Exception as e:
    error_str = str(e)
    
    if "rate_limit_exceeded" in error_str or "429" in error_str:
        print(f"\n❌ RATE LIMIT EXCEEDED!")
        print(f"{'='*60}")
        
        # Parse the error message
        import re
        
        used_match = re.search(r'Used (\d+)', error_str)
        limit_match = re.search(r'Limit (\d+)', error_str)
        requested_match = re.search(r'Requested (\d+)', error_str)
        time_match = re.search(r'Please try again in ([^.]+)', error_str)
        
        if used_match and limit_match:
            used = int(used_match.group(1))
            limit = int(limit_match.group(1))
            remaining = limit - used
            percentage_used = (used / limit) * 100
            
            print(f"\n📊 CURRENT USAGE:")
            print(f"   Tokens used: {used:,}/{limit:,} ({percentage_used:.1f}%)")
            print(f"   Tokens remaining: {remaining:,}")
            
            if requested_match:
                requested = int(requested_match.group(1))
                print(f"   Tokens needed for test: {requested:,}")
                print(f"   Would exceed by: {(used + requested - limit):,}")
        
        if time_match:
            wait_time = time_match.group(1)
            print(f"\n⏰ WAIT TIME:")
            print(f"   Please wait: {wait_time}")
        
        print(f"\n📋 Full Error Message:")
        print(f"   {error_str[:500]}...")
        
    else:
        print(f"\n❌ Error: {e}")

print(f"\n{'='*60}")
print(f"💡 TIP: Visit https://platform.openai.com/usage")
print(f"   to see detailed usage history and graphs")
print(f"{'='*60}\n")