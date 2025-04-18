import os

# Load API key from environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# To use this:
# 1. Copy this file to api_key.py
# 2. Set your OPENAI_API_KEY environment variable:
#    - For Unix/Linux/Mac: export OPENAI_API_KEY='your-api-key'
#    - For Windows: set OPENAI_API_KEY=your-api-key 