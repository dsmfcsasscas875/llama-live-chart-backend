import resend
from app.core.config import settings
from typing import List, Dict, Any

class EmailService:
    def __init__(self):
        resend.api_key = settings.RESEND_API_KEY
        
    def send_watchlist_alert(self, user_email: str, watchlist_data: List[Dict[str, Any]]):
        if not settings.RESEND_API_KEY:
            print("Email skipped: RESEND_API_KEY not configured.")
            return

        # Build elegant HTML content
        html_content = """
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; color: #333;">
            <h2 style="color: #0ab5d8; border-bottom: 2px solid #eee; padding-bottom: 10px;">
                📈 Your Daily Watchlist Summary
            </h2>
            <p>Hello! Here is the current status of the assets you are tracking:</p>
            <table style="width: 100%; border-collapse: collapse; margin-top: 20px;">
                <thead>
                    <tr style="background-color: #f8f9fa; text-align: left;">
                        <th style="padding: 12px; border-bottom: 1px solid #ddd;">Asset</th>
                        <th style="padding: 12px; border-bottom: 1px solid #ddd;">Current Price</th>
                        <th style="padding: 12px; border-bottom: 1px solid #ddd;">Daily Change</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for item in watchlist_data:
            symbol = item.get('symbol', 'N/A')
            price = round(item.get('price', 0), 2)
            change = item.get('change_percent', 0)
            
            color = "#22c55e" if change >= 0 else "#ef4444"
            sign = "+" if change >= 0 else ""
            
            html_content += f"""
                <tr>
                    <td style="padding: 12px; border-bottom: 1px solid #eee;"><strong>{symbol}</strong></td>
                    <td style="padding: 12px; border-bottom: 1px solid #eee;">${price}</td>
                    <td style="padding: 12px; border-bottom: 1px solid #eee; color: {color}; font-weight: bold;">
                        {sign}{round(change, 2)}%
                    </td>
                </tr>
            """
            
        html_content += """
                </tbody>
            </table>
            <p style="margin-top: 30px; font-size: 12px; color: #888;">
                Stay ahead of the market with Live-Chart Portfolio Tracker.
            </p>
        </div>
        """

        try:
            params = {
                "from": "Acme <onboarding@resend.dev>",
                "to": [user_email],
                "subject": "📈 Watchlist Alert: Market Update",
                "html": html_content
            }
            
            resend.Emails.send(params)
            print(f"Watchlist alert sent to {user_email}")
        except Exception as e:
            print(f"Error sending email to {user_email}: {e}")

email_service = EmailService()
