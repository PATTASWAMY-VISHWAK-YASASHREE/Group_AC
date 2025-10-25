from twilio.rest import Client
from dotenv import load_dotenv
import os

# Load credentials
load_dotenv()

print("\n" + "="*50)
print("📞 TWILIO CALL TEST")
print("="*50)

# Get credentials from .env
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
from_number = os.getenv('TWILIO_PHONE_NUMBER')
to_number = os.getenv('YOUR_PHONE_NUMBER')

print(f"\n📋 Configuration:")
print(f"   From: {from_number}")
print(f"   To: {to_number}")
print(f"   Account: {account_sid[:10]}...")

# Create Twilio client
client = Client(account_sid, auth_token)

print(f"\n🚀 Initiating call...")
print(f"   Your phone will ring in 5-10 seconds!")

try:
    # Make the call
    call = client.calls.create(
        # What the AI will say when you pick up
        twiml='''
        <Response>
            <Say voice="Polly.Aditi" language="en-IN">
                Hello! This is your Twilio test call. 
                If you can hear this message, your Twilio API is working perfectly. 
                Congratulations! You are now ready to build your AI voice bot. 
                Goodbye!
            </Say>
        </Response>
        ''',
        from_=from_number,
        to=to_number
    )
    
    print(f"\n✅ Call initiated successfully!")
    print(f"   Call SID: {call.sid}")
    print(f"   Status: {call.status}")
    print(f"\n📱 Pick up your phone and listen to the message!")
    print("="*50 + "\n")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print(f"\n🔍 Troubleshooting:")
    print(f"   1. Check if credentials in .env are correct")
    print(f"   2. Ensure YOUR_PHONE_NUMBER is verified in Twilio")
    print(f"   3. Include country code (+91 for India, +1 for US)")
    print("="*50 + "\n")
