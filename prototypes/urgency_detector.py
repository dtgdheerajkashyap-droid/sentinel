def analyze_urgency(text):

    text = text.lower()

    risk_score = 0
    reasons = []

    keywords = {
        "urgent": 10,
        "immediately": 10,
        "verify": 10,
        "account suspended": 15,
        "suspended": 10,
        "click now": 10,
        "act now": 10,
        "limited time": 10,
        "password": 5,
        "otp": 5
    }

    for word, score in keywords.items():
        if word in text:
            risk_score += score
            reasons.append(f"Contains '{word}'")

    risk_score = min(risk_score, 15)

    return risk_score, reasons


email = """
Dear Customer,

Your SBI account will be suspended.

Verify immediately:
https://amaz0n-security.xyz/login
"""

score, reasons = analyze_urgency(email)

print("Urgency Score:", score)

for reason in reasons:
    print("-", reason)