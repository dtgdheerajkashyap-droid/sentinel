import re

def analyze_sender(sender):

    risk_score = 0
    reasons = []

    sender = sender.lower()

    suspicious_tlds = [
        ".xyz",
        ".top",
        ".click",
        ".site",
        ".online"
    ]

    for tld in suspicious_tlds:
        if tld in sender:
            risk_score += 40
            reasons.append(f"Suspicious TLD: {tld}")

    if re.search(r'\d', sender):
        risk_score += 30
        reasons.append("Sender contains numbers")

    if len(sender) > 30:
        risk_score += 10
        reasons.append("Unusually long sender address")

    return risk_score, reasons


sender = "support@amaz0n-security.xyz"

score, reasons = analyze_sender(sender)

print("Sender:", sender)
print("Risk Score:", score)

for reason in reasons:
    print("-", reason)