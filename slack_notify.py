# slack_notify.py
import requests

webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
message = {
    "text": "🎉 *Your PBS job 'Simple_test' has completed successfully!*",
}
requests.post(webhook_url, json=message)

