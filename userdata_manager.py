import json
import re
from typing import Optional, Dict, Any
from pathlib import Path


class UserDataManager:
    """
    Manages user data from userdata.json.
    Handles loading, querying, and formatting user information.
    """

    def __init__(self, filepath: str = "userdata.json"):
        """
        Initialize UserDataManager and load user data.
        
        Args:
            filepath: Path to userdata.json file
        """
        self.filepath = filepath
        self.users_data = {}
        self.load_data()

    def load_data(self):
        """Load user data from JSON file."""
        try:
            if not Path(self.filepath).exists():
                print(f"⚠️  UserData file not found at {self.filepath}")
                self.users_data = {}
                return

            with open(self.filepath, 'r') as f:
                data = json.load(f)
                
            # Create a phone number index for fast lookup
            self.users_data = {}
            if "users" in data and isinstance(data["users"], list):
                for user in data["users"]:
                    phone = user.get("phone_number", "").strip()
                    if phone:
                        self.users_data[phone] = user
                print(f"✓ Loaded {len(self.users_data)} users from {self.filepath}")
            else:
                print(f"⚠️  Invalid userdata.json format")
                
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing {self.filepath}: {e}")
            self.users_data = {}
        except Exception as e:
            print(f"❌ Error loading user data: {e}")
            self.users_data = {}

    def extract_phone_number(self, text: str) -> Optional[str]:
        """
        Extract phone number from various formats.
        Handles formats like +919876543210, 9876543210, +91-98765-43210, etc.
        
        Args:
            text: Text that may contain a phone number
            
        Returns:
            Normalized phone number (+91...) or None
        """
        if not text:
            return None

        # Remove common separators and spaces
        cleaned = re.sub(r'[\s\-\(\)\.]+', '', text)
        
        # Try to find 10-digit number (Indian format)
        match = re.search(r'(\d{10})$', cleaned)
        if match:
            return "+91" + match.group(1)
        
        # Try to find 12-digit number starting with 91
        match = re.search(r'91(\d{10})', cleaned)
        if match:
            return "+91" + match.group(1)
        
        # Try to find number starting with +91
        match = re.search(r'(\+91\d{10})', cleaned)
        if match:
            return match.group(1)
        
        return None

    def get_user_by_phone(self, phone_number: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Get user data by phone number.
        
        Args:
            phone_number: Phone number in any format
            
        Returns:
            User data dict or None if not found
        """
        if not phone_number:
            return None
            
        # Normalize the phone number
        normalized = self.extract_phone_number(phone_number)
        
        if not normalized:
            return None
        
        user = self.users_data.get(normalized)
        if user:
            print(f"✓ Found user: {user.get('name')} ({normalized})")
        else:
            print(f"⚠️  User not found for: {normalized}")
            
        return user

    def format_user_context(self, user_data: Optional[Dict[str, Any]]) -> str:
        """
        Format user data into readable context for the LLM.
        
        Args:
            user_data: User data dict from userdata.json
            
        Returns:
            Formatted string with user information
        """
        if not user_data:
            return ""
        
        context = "\n=== CUSTOMER ACCOUNT INFO ===\n"
        context += f"Name: {user_data.get('name', 'N/A')}\n"
        context += f"Phone: {user_data.get('phone_number', 'N/A')}\n"
        context += f"Current Balance: ₹{user_data.get('current_balance', 0)}\n"
        context += f"Recharge Plan: {user_data.get('recharge_plan', 'N/A')}\n"
        context += f"Data Left: {user_data.get('data_left', 'N/A')}\n"
        context += f"Plan Validity: {user_data.get('validity', 'N/A')}\n"
        context += f"Last Recharge: {user_data.get('last_recharge_date', 'N/A')} (₹{user_data.get('last_recharge_amount', 0)})\n"
        context += f"Talktime Balance: ₹{user_data.get('talktime_balance', 0)}\n"
        context += f"SMS Balance: {user_data.get('sms_balance', 0)} SMS\n"
        
        active_services = user_data.get('active_services', [])
        if active_services:
            context += f"Active Services: {', '.join(active_services)}\n"
        
        context += f"Upcoming Bill Date: {user_data.get('upcoming_bill_date', 'N/A')}\n"
        context += "=== END ACCOUNT INFO ===\n"
        
        return context


# Global instance for easy access
_user_manager = None


def get_user_manager() -> UserDataManager:
    """Get or create the global UserDataManager instance."""
    global _user_manager
    if _user_manager is None:
        _user_manager = UserDataManager()
    return _user_manager


def get_user_by_phone(phone_number: Optional[str]) -> Optional[Dict[str, Any]]:
    """Convenience function to get user by phone number."""
    return get_user_manager().get_user_by_phone(phone_number)


def format_user_context(user_data: Optional[Dict[str, Any]]) -> str:
    """Convenience function to format user context."""
    return get_user_manager().format_user_context(user_data)


def extract_phone_number(text: str) -> Optional[str]:
    """Convenience function to extract phone number."""
    return get_user_manager().extract_phone_number(text)


if __name__ == "__main__":
    # Test the manager
    manager = UserDataManager()
    
    # Test phone number extraction
    test_phones = [
        "9876543210",
        "+919876543210",
        "+91-98765-43210",
        "919876543210",
        "From +91 9876543210"
    ]
    
    print("\n📱 Testing phone number extraction:")
    for test in test_phones:
        extracted = manager.extract_phone_number(test)
        print(f"  '{test}' → {extracted}")
    
    # Test user lookup
    print("\n🔍 Testing user lookup:")
    user = manager.get_user_by_phone("+919876543210")
    if user:
        print(manager.format_user_context(user))
