# slack_notify.py
import requests

webhook_url = "REMOVED"
message = {
    "text": "🎉 *Your PBS job 'Simple_test' has completed successfully!*",
}
requests.post(webhook_url, json=message)

