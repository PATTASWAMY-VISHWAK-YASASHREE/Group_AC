from fastapi import FastAPI, Form, Request
from fastapi.responses import Response
import uvicorn
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

app = FastAPI()

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Initialize Gemini model - using fast, lightweight model for voice
model = genai.GenerativeModel('models/gemini-2.5-flash-lite')

# Telecom knowledge base for context
TELECOM_CONTEXT = """
You are a helpful customer service AI for a telecom company called TelecomCare.

Common Issues and Solutions:

WiFi Problems:
- Restart router by unplugging for 30 seconds
- Check if other devices can connect
- Reset router by holding reset button for 10 seconds
- Check if WiFi is enabled on device
- Move closer to router

Mobile Data Issues:
- Turn airplane mode on and off
- Check if mobile data is enabled in settings
- Restart phone
- Remove and reinsert SIM card
- Check data balance

Billing Questions:
- Log into account on website or mobile app
- Bills generated on same date each month
- Download PDF from account portal
- Check payment history in app

Payment Methods:
- Website payment portal
- Mobile app
- Auto-pay setup available
- Credit card, debit card, net banking accepted
- Payment at authorized retail stores

Router Setup:
- Connect power cable
- Wait for lights to stabilize (2-3 minutes)
- Connect via WiFi or ethernet cable
- Default WiFi name and password on router sticker
- Access settings at 192.168.1.1

SIM Card Issues:
- Check if SIM is properly inserted
- Clean SIM card gently with soft cloth
- Try SIM in another phone to test
- Visit service center if damaged

Network Coverage:
- Check coverage map on website
- Report network issues via app
- Temporary outages possible during maintenance
- Try manual network selection in settings

Instructions:
- Be helpful and friendly
- Give clear, step-by-step solutions
- Keep responses concise (2-3 sentences max for voice calls)
- If issue is complex or requires account access, offer to escalate
- Use simple language, avoid technical jargon
- Speak naturally as if talking on phone
"""

def get_ai_answer(question):
    """Use Gemini to generate intelligent responses"""
    try:
        # Create prompt with context
        prompt = f"{TELECOM_CONTEXT}\n\nCustomer Question: {question}\n\nYour Response (keep it brief for phone call, 2-3 sentences):"
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=150,
                temperature=0.7,
            )
        )
        
        answer = response.text.strip()
        return answer
        
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        return "I'm having trouble connecting to my knowledge base right now. Let me transfer you to a human agent who can help."


@app.get("/")
def home():
    return {
        "message": "TelecomCare AI Voice Bot Running!",
        "status": "active",
        "ai_model": "Google Gemini 2.5 Flash Lite",
        "cost": "FREE",
        "endpoints": {
            "incoming_call": "/voice/incoming",
            "process_speech": "/voice/process",
            "followup": "/voice/followup"
        }
    }


@app.post("/voice/incoming")
async def incoming_call(request: Request):
    """Handle incoming calls - greet and ask for issue"""
    
    form_data = await request.form()
    caller = form_data.get('From', 'Unknown')
    
    print(f"\n📞 Incoming call from: {caller}")
    
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">
        Hello! Welcome to TelecomCare AI Assistant.
        I can help you with any telecom issue.
        Please tell me your problem or question.
    </Say>
    <Gather 
        input="speech" 
        action="/voice/process" 
        language="en-IN" 
        speechTimeout="auto"
        timeout="5"
        hints="wifi, mobile data, internet, bill, payment, router, network, not working, slow, problem, recharge, plan, sim card">
        <Say voice="Polly.Aditi" language="en-IN">
            I'm listening. Please speak now.
        </Say>
    </Gather>
    <Say voice="Polly.Aditi" language="en-IN">
        I didn't hear anything. Please call back when you're ready. Goodbye!
    </Say>
</Response>"""
    
    return Response(content=twiml, media_type="application/xml")


@app.post("/voice/process")
async def process_speech(
    SpeechResult: str = Form(None),
    Confidence: float = Form(0.0),
    From: str = Form(None)
):
    """Process speech using AI and respond"""
    
    print(f"\n🎤 Speech received from {From}")
    print(f"   Transcription: {SpeechResult}")
    print(f"   Confidence: {Confidence * 100:.1f}%")
    
    # Check if speech was understood
    if not SpeechResult or Confidence < 0.4:
        print("   ⚠️ Low confidence, asking user to repeat")
        
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">
        Sorry, I couldn't understand that clearly. Could you please repeat your question more slowly?
    </Say>
    <Gather 
        input="speech" 
        action="/voice/process" 
        language="en-IN" 
        speechTimeout="auto"
        timeout="5">
        <Say voice="Polly.Aditi" language="en-IN">
            Please speak your question again.
        </Say>
    </Gather>
</Response>"""
        
        return Response(content=twiml, media_type="application/xml")
    
    # Get AI-powered answer from Gemini
    print(f"   🌟 Getting Gemini AI response...")
    answer = get_ai_answer(SpeechResult)
    print(f"   💬 AI Answer: {answer[:100]}...")
    
    # Escape special characters for XML
    answer = answer.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')
    
    # Speak the answer back
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">
        {answer}
    </Say>
    <Pause length="1"/>
    <Say voice="Polly.Aditi" language="en-IN">
        Do you have another question? Say yes or no.
    </Say>
    <Gather 
        input="speech" 
        action="/voice/followup" 
        language="en-IN" 
        speechTimeout="auto"
        timeout="3"
        hints="yes, no, yeah, nope, sure, okay">
        <Say voice="Polly.Aditi" language="en-IN">
            Say yes for another question, or no to end the call.
        </Say>
    </Gather>
    <Say voice="Polly.Aditi" language="en-IN">
        Thank you for calling TelecomCare. Have a great day!
    </Say>
</Response>"""
    
    return Response(content=twiml, media_type="application/xml")


@app.post("/voice/followup")
async def followup(
    SpeechResult: str = Form(None),
    From: str = Form(None)
):
    """Handle follow-up questions"""
    
    user_input = (SpeechResult or "").lower()
    print(f"\n🔄 Follow-up from {From}: {user_input}")
    
    # Check if user wants to continue
    if any(word in user_input for word in ['yes', 'yeah', 'yep', 'sure', 'okay', 'another']):
        print("   ↻ User wants to ask another question")
        
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">
        Sure! What's your next question?
    </Say>
    <Gather 
        input="speech" 
        action="/voice/process" 
        language="en-IN" 
        speechTimeout="auto"
        timeout="5">
        <Say voice="Polly.Aditi" language="en-IN">
            I'm listening.
        </Say>
    </Gather>
</Response>"""
    else:
        print("   ✓ Ending call")
        
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">
        Thank you for calling TelecomCare. Have a wonderful day!
    </Say>
    <Hangup/>
</Response>"""
    
    return Response(content=twiml, media_type="application/xml")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 TelecomCare AI Voice Bot (Google Gemini)")
    print("="*60)
    print("\n📍 Server starting at: http://localhost:8000")
    print("🌟 AI Model: Gemini 2.5 Flash Lite (FREE!)")
    print("💰 Cost: $0.00 per call")
    print("\n🌐 Endpoints:")
    print("   GET  /              - Health check")
    print("   POST /voice/incoming - Handle incoming calls")
    print("   POST /voice/process  - Process speech with Gemini")
    print("   POST /voice/followup - Handle follow-ups")
    print("\n⚠️  Make sure:")
    print("   1. GEMINI_API_KEY is set in .env")
    print("   2. ngrok is running")
    print("   3. Twilio webhook is updated")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
