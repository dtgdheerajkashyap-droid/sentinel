import re
import tldextract

def analyze_url(url):

    ext = tldextract.extract(url)

    domain = ext.domain
    suffix = ext.suffix

    risk_score = 0
    reasons = []

    suspicious_tlds = [
        "xyz",
        "top",
        "click",
        "site",
        "online",
        "live"
    ]

    if suffix in suspicious_tlds:
        risk_score += 40
        reasons.append(f"Suspicious TLD: .{suffix}")

    if any(char.isdigit() for char in domain):
        risk_score += 30
        reasons.append("Domain contains numbers")

    if len(domain) > 20:
        risk_score += 20
        reasons.append("Unusually long domain")

    return risk_score, reasons


text = """
Verify your account now:
https://amaz0n-security.xyz/login
"""

urls = re.findall(r'https?://\S+', text)

for url in urls:

    score, reasons = analyze_url(url)

    print("\nURL:", url)
    print("Risk Score:", score)

    for reason in reasons:
        print("-", reason)