"""
Diagnostic Script: Test LLM Connection and API Keys
This script helps diagnose why the workflow is returning 0 results
"""

import os
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv()

print("=" * 70)
print("🔍 AIPR Backend - LLM Connection Diagnostics")
print("=" * 70)
print()

# Check API keys
print("📋 Step 1: Checking API Keys...")
print("-" * 70)

google_key = os.getenv("GOOGLE_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

if google_key:
    masked_google = google_key[:20] + "..." + google_key[-4:] if len(google_key) > 24 else "***"
    print(f"✅ GOOGLE_API_KEY found: {masked_google}")
else:
    print("❌ GOOGLE_API_KEY not found in .env file")

if openai_key:
    masked_openai = openai_key[:10] + "..." + openai_key[-4:] if len(openai_key) > 14 else "***"
    print(f"✅ OPENAI_API_KEY found: {masked_openai}")
else:
    print("⚠️  OPENAI_API_KEY not found (fallback not available)")

print()

# Test Gemini Connection
print("📋 Step 2: Testing Google Gemini Connection...")
print("-" * 70)

if google_key:
    try:
        print("Attempting to initialize Gemini with google.generativeai...")
        import google.generativeai as genai
        
        genai.configure(api_key=google_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        print("Sending test request...")
        response = model.generate_content("Say 'Hello' in one word only")
        
        print(f"✅ Gemini is WORKING!")
        print(f"   Test response: {response.text[:50]}")
        
    except Exception as e:
        error_str = str(e)
        print(f"❌ Gemini FAILED: {error_str[:200]}")
        
        if "PERMISSION_DENIED" in error_str or "suspended" in error_str.lower():
            print("\n🚨 ERROR DIAGNOSIS:")
            print("   Your Gemini API key has been SUSPENDED by Google.")
            print("   Possible reasons:")
            print("   - Key exposed publicly (e.g., committed to GitHub)")
            print("   - Billing issues")
            print("   - Terms of service violation")
            print()
            print("   SOLUTION:")
            print("   1. Go to: https://aistudio.google.com/app/apikey")
            print("   2. Create a NEW API key")
            print("   3. Update your .env file with the new key")
            print("   4. Restart the backend")
else:
    print("⏭️  Skipping (no API key found)")

print()

# Test LangChain Integration
print("📋 Step 3: Testing LangChain Integration...")
print("-" * 70)

if google_key:
    try:
        print("Attempting to create LLM with fallback...")
        from llm_fallback import create_llm_with_fallback
        
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=google_key,
            openai_api_key=openai_key
        )
        
        print("Sending test invoke...")
        result = llm.invoke("Say 'Hello' in one word")
        
        print(f"✅ LangChain integration is WORKING!")
        print(f"   Test response: {result.content if hasattr(result, 'content') else str(result)[:50]}")
        
    except Exception as e:
        error_str = str(e)
        print(f"❌ LangChain integration FAILED: {error_str[:200]}")
        
        if openai_key:
            print("\n💡 Attempting OpenAI fallback...")
            try:
                from langchain_openai import ChatOpenAI
                
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.1,
                    openai_api_key=openai_key
                )
                
                result = llm.invoke("Say 'Hello' in one word")
                print(f"✅ OpenAI fallback is WORKING!")
                print(f"   Test response: {result.content[:50]}")
                print()
                print("   RECOMMENDATION:")
                print("   Enable OpenAI fallback in llm_fallback.py (see FIX_GUIDE.md)")
                
            except Exception as openai_error:
                print(f"❌ OpenAI also failed: {str(openai_error)[:100]}")
        else:
            print("\n   No OpenAI key available for fallback testing")
else:
    print("⏭️  Skipping (no API key found)")

print()

# Test Instrument Tools
print("📋 Step 4: Testing Instrument Identification Tools...")
print("-" * 70)

if google_key or openai_key:
    try:
        print("Importing identify_instruments_tool...")
        from tools.instrument_tools import identify_instruments_tool
        
        print("Testing with sample requirement...")
        result = identify_instruments_tool.invoke({
            "requirements": "I need a temperature sensor for monitoring reactor core"
        })
        
        if result.get("success"):
            print(f"✅ Instrument identification is WORKING!")
            print(f"   Found {len(result.get('instruments', []))} instruments")
        else:
            print(f"⚠️  Tool returned success=False")
            print(f"   Error: {result.get('error', 'Unknown')}")
            
    except Exception as e:
        print(f"❌ Instrument tool FAILED: {str(e)[:200]}")
else:
    print("⏭️  Skipping (no API keys available)")

print()
print("=" * 70)
print("🎯 DIAGNOSTIC SUMMARY")
print("=" * 70)
print()

# Summary
has_working_llm = False
needs_new_key = False

if google_key:
    try:
        import google.generativeai as genai
        genai.configure(api_key=google_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content("test")
        has_working_llm = True
    except Exception as e:
        if "PERMISSION_DENIED" in str(e) or "suspended" in str(e).lower():
            needs_new_key = True

if needs_new_key:
    print("❌ CRITICAL ISSUE: Your Gemini API key is suspended")
    print()
    print("🔧 IMMEDIATE ACTION REQUIRED:")
    print("   1. Get a new Gemini API key from: https://aistudio.google.com/app/apikey")
    print("   2. Update .env file: GOOGLE_API_KEY=your_new_key")
    print("   3. Restart backend: python main.py")
    print()
    print("   Note: The workflow will return 0 items until this is fixed.")
    
elif has_working_llm:
    print("✅ All systems operational!")
    print()
    print("   Your LLM connection is working correctly.")
    print("   If you're still seeing 0 results, check:")
    print("   - Backend logs for other errors")
    print("   - User input format")
    print("   - Workflow state in database")
    
else:
    print("⚠️  Unable to establish LLM connection")
    print()
    print("🔧 POSSIBLE SOLUTIONS:")
    
    if not google_key and not openai_key:
        print("   1. Add GOOGLE_API_KEY to .env file")
        print("      Get key from: https://aistudio.google.com/app/apikey")
    
    if openai_key and not has_working_llm:
        print("   2. Enable OpenAI fallback in llm_fallback.py")
        print("      See: FIX_GUIDE.md for instructions")
    
    if not openai_key:
        print("   3. (Optional) Add OPENAI_API_KEY for fallback support")

print()
print("=" * 70)
print("For detailed fix instructions, see: FIX_GUIDE.md")
print("=" * 70)
