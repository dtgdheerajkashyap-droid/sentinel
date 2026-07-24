import os
import sys
import re
import html
import base64
import pickle
import sqlite3
from email.utils import parseaddr
from urllib.parse import urlparse, parse_qs, unquote

import tldextract
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build


# --------------------------------------------------
# UTF-8 FIX FOR WINDOWS CONSOLE
# --------------------------------------------------

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT_DIR, "sentinelai.db")
CREDENTIALS_PATH = os.path.join(ROOT_DIR, "config", "credentials.json")
TOKEN_PATH = os.path.join(ROOT_DIR, "config", "token.pickle")


TRUSTED_DOMAINS = [
    "google.com",
    "gmail.com",
    "naukri.com",
    "canarabank.com",
    "reddit.com",
    "redditmail.com",
    "snapchat.com",
    "playo.co",
    "googleapis.com",
    "gstatic.com",
    "w3.org",
]


SUSPICIOUS_TLDS = [
    "xyz",
    "top",
    "click",
    "site",
    "online",
    "live",
]


# --------------------------------------------------
# DATABASE SETUP
# --------------------------------------------------

def ensure_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS email_scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT UNIQUE NOT NULL,
            sender TEXT,
            subject TEXT,
            risk_score INTEGER,
            risk_level TEXT,
            urls_found INTEGER,
            urgency_risk INTEGER,
            scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    extra_columns = {
        "reasons": "TEXT",
        "detected_urls": "TEXT",
        "scan_summary": "TEXT",
    }

    cursor.execute("PRAGMA table_info(email_scans)")
    existing_columns = [column[1] for column in cursor.fetchall()]

    for column_name, column_type in extra_columns.items():
        if column_name not in existing_columns:
            cursor.execute(
                f"ALTER TABLE email_scans ADD COLUMN {column_name} {column_type}"
            )

    conn.commit()
    conn.close()


# --------------------------------------------------
# GMAIL AUTH
# --------------------------------------------------

def get_gmail_service():
    creds = None

    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "rb") as token_file:
            creds = pickle.load(token_file)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            CREDENTIALS_PATH,
            SCOPES
        )
        creds = flow.run_local_server(port=0)

        os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)

        with open(TOKEN_PATH, "wb") as token_file:
            pickle.dump(creds, token_file)

    return build("gmail", "v1", credentials=creds)


# --------------------------------------------------
# EMAIL BODY EXTRACTION
# --------------------------------------------------

def extract_body(payload):
    try:
        if "data" in payload.get("body", {}):
            data = payload["body"]["data"]
            decoded = base64.urlsafe_b64decode(data).decode(
                "utf-8",
                errors="ignore"
            )
            return decoded

        for part in payload.get("parts", []):
            body = extract_body(part)
            if body:
                return body

    except Exception:
        return ""

    return ""


# --------------------------------------------------
# URL CLEANING AND EXTRACTION
# --------------------------------------------------

def clean_url(url):
    url = html.unescape(url)
    url = url.strip()
    url = url.rstrip(".,);]'\"")

    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)

    redirect_keys = [
        "redirect",
        "redirect_url",
        "url",
        "target",
        "u",
    ]

    for key in redirect_keys:
        if key in query_params:
            redirected_url = unquote(query_params[key][0])

            if redirected_url.startswith("http"):
                return clean_url(redirected_url)

    return url


def should_ignore_url(url):
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    full_url = url.lower()

    ignored_extensions = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".webp",
        ".css",
        ".woff",
        ".woff2",
        ".ttf",
        ".ico",
    )

    ignored_domains = [
        "fonts.googleapis.com",
        "fonts.gstatic.com",
        "w3.org",
    ]

    ignored_keywords = [
        "collectorapi",
        "communicationopen",
        "unsubscribe",
        "tracking",
        "pixel",
        "open.gif",
        "utm_",
    ]

    if not domain:
        return True

    if any(domain.endswith(d) for d in ignored_domains):
        return True

    if path.endswith(ignored_extensions):
        return True

    if any(keyword in full_url for keyword in ignored_keywords):
        return True

    return False


def extract_urls(text):
    text = html.unescape(text)

    raw_urls = re.findall(r"https?://[^\s<>'\"]+", text)

    cleaned_urls = []

    for raw_url in raw_urls:
        url = clean_url(raw_url)

        if should_ignore_url(url):
            continue

        if url not in cleaned_urls:
            cleaned_urls.append(url)

    return cleaned_urls


# --------------------------------------------------
# DOMAIN HELPERS
# --------------------------------------------------

def is_trusted_domain(domain):
    domain = domain.lower()

    for trusted in TRUSTED_DOMAINS:
        if domain == trusted or domain.endswith("." + trusted):
            return True

    return False


def get_domain_from_email(sender):
    _, email_address = parseaddr(sender)

    if "@" not in email_address:
        return ""

    return email_address.split("@")[-1].lower()


# --------------------------------------------------
# URL RISK ANALYSIS
# --------------------------------------------------

def analyze_url(url):
    url = clean_url(url)

    if should_ignore_url(url):
        return 0, []

    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    hostname = hostname.lower()

    if hostname.startswith("www."):
        hostname = hostname[4:]

    ext = tldextract.extract(hostname)

    domain = ext.domain.lower()
    suffix = ext.suffix.lower()
    full_domain = f"{domain}.{suffix}" if suffix else domain

    raw_score = 0
    reasons = []

    if is_trusted_domain(full_domain):
        return 0, []

    if suffix in SUSPICIOUS_TLDS:
        raw_score += 40
        reasons.append(f"Suspicious URL TLD: .{suffix}")

    if any(char.isdigit() for char in domain):
        raw_score += 30
        reasons.append("URL domain contains numbers")

    if len(domain) > 20:
        raw_score += 20
        reasons.append("Unusually long URL domain")

    query_params = parse_qs(parsed.query)

    redirect_keys = [
        "redirect",
        "redirect_url",
        "url",
        "target",
        "u",
    ]

    for key in redirect_keys:
        if key in query_params:
            redirected_url = unquote(query_params[key][0])

            if redirected_url.startswith("http"):
                redirect_score, redirect_reasons = analyze_url(redirected_url)
                raw_score += redirect_score
                reasons.extend(redirect_reasons)

    return raw_score, reasons


# --------------------------------------------------
# SENDER RISK ANALYSIS
# --------------------------------------------------

def analyze_sender(sender):
    sender = sender.lower()
    sender_domain = get_domain_from_email(sender)

    raw_score = 0
    reasons = []

    if sender_domain and is_trusted_domain(sender_domain):
        return 0, []

    for tld in SUSPICIOUS_TLDS:
        if sender_domain.endswith("." + tld):
            raw_score += 40
            reasons.append(f"Suspicious sender TLD: .{tld}")

    if re.search(r"\d", sender_domain):
        raw_score += 30
        reasons.append("Sender domain contains numbers")

    if len(sender_domain) > 35:
        raw_score += 10
        reasons.append("Unusually long sender domain")

    return raw_score, reasons


# --------------------------------------------------
# URGENCY LANGUAGE ANALYSIS
# --------------------------------------------------

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
        "otp": 5,
    }

    for word, score in keywords.items():
        if word in text:
            risk_score += score
            reasons.append(f"Contains '{word}'")

    risk_score = min(risk_score, 15)

    return risk_score, reasons


# --------------------------------------------------
# FINAL RISK ENGINE
# --------------------------------------------------

def calculate_risk(url_score_raw, sender_score_raw, urgency_risk):
    url_risk = min(url_score_raw / 2, 35)
    sender_risk = min(sender_score_raw / 3, 25)

    final_score = round(url_risk + sender_risk + urgency_risk)

    if final_score >= 70:
        level = "HIGH RISK"
    elif final_score >= 30:
        level = "MEDIUM RISK"
    else:
        level = "LOW RISK"

    return final_score, level, round(url_risk), round(sender_risk)


# --------------------------------------------------
# AI-STYLE SUMMARY GENERATOR
# --------------------------------------------------

def generate_scan_summary(
    sender,
    subject,
    risk_score,
    risk_level,
    reasons,
    urls_found,
    urgency_risk
):
    sender_text = sender if sender else "Unknown sender"
    subject_text = subject if subject else "No subject"

    if not reasons:
        reasons = ["No major risk detected"]

    reason_text = ", ".join(reasons[:3])

    if risk_level == "HIGH RISK":
        return (
            f"This email is marked HIGH RISK. The message from {sender_text} "
            f"with subject '{subject_text}' shows strong warning signs such as "
            f"{reason_text}. It contains {urls_found} detected URL(s) and has "
            f"a risk score of {risk_score}/100. The user should not click any "
            f"links and should verify the sender through an official source."
        )

    if risk_level == "MEDIUM RISK":
        return (
            f"This email is marked MEDIUM RISK. The message from {sender_text} "
            f"with subject '{subject_text}' contains suspicious indicators such "
            f"as {reason_text}. It contains {urls_found} detected URL(s) and has "
            f"a risk score of {risk_score}/100. The user should review the email "
            f"carefully before clicking any links."
        )

    if urgency_risk > 0 or urls_found > 0:
        return (
            f"This email is marked LOW RISK. The message from {sender_text} "
            f"with subject '{subject_text}' does not show strong phishing "
            f"indicators, but it contains {urls_found} URL(s) and minor signals "
            f"such as {reason_text}. The user can treat it as mostly safe, but "
            f"should still avoid clicking unknown links blindly."
        )

    return (
        f"This email is marked LOW RISK. The message from {sender_text} "
        f"with subject '{subject_text}' does not show major suspicious "
        f"indicators. No immediate action is required."
    )


# --------------------------------------------------
# SAVE SCAN TO DATABASE
# --------------------------------------------------

def save_scan(
    message_id,
    sender,
    subject,
    risk_score,
    risk_level,
    urls_found,
    urgency_risk,
    reasons,
    detected_urls,
    scan_summary
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    reasons_text = "\n".join(reasons) if reasons else "No major risk detected"
    urls_text = "\n".join(detected_urls) if detected_urls else "No URLs found"

    cursor.execute("""
        INSERT INTO email_scans (
            message_id,
            sender,
            subject,
            risk_score,
            risk_level,
            urls_found,
            urgency_risk,
            reasons,
            detected_urls,
            scan_summary
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(message_id) DO UPDATE SET
            sender = excluded.sender,
            subject = excluded.subject,
            risk_score = excluded.risk_score,
            risk_level = excluded.risk_level,
            urls_found = excluded.urls_found,
            urgency_risk = excluded.urgency_risk,
            reasons = excluded.reasons,
            detected_urls = excluded.detected_urls,
            scan_summary = excluded.scan_summary
    """, (
        message_id,
        sender,
        subject,
        risk_score,
        risk_level,
        urls_found,
        urgency_risk,
        reasons_text,
        urls_text,
        scan_summary,
    ))

    conn.commit()
    conn.close()

    return True


# --------------------------------------------------
# MAIN SCANNER
# --------------------------------------------------

def run_scanner():
    ensure_database()

    service = get_gmail_service()

    results = service.users().messages().list(
        userId="me",
        q="is:unread",
        maxResults=10
    ).execute()

    messages = results.get("messages", [])

    print("\nUnread Emails Found:", len(messages))

    if not messages:
        print("No unread emails found.")
        return

    for message in messages:
        msg = service.users().messages().get(
            userId="me",
            id=message["id"],
            format="full"
        ).execute()

        headers = msg["payload"].get("headers", [])

        sender = ""
        subject = ""

        for header in headers:
            header_name = header["name"].lower()

            if header_name == "from":
                sender = header["value"]

            elif header_name == "subject":
                subject = header["value"]

        body = extract_body(msg["payload"])
        urls = extract_urls(body)

        total_url_raw_score = 0
        all_reasons = []

        for url in urls:
            score, reasons = analyze_url(url)
            total_url_raw_score += score
            all_reasons.extend(reasons)

        sender_raw_score, sender_reasons = analyze_sender(sender)
        all_reasons.extend(sender_reasons)

        urgency_risk, urgency_reasons = analyze_urgency(subject + " " + body)
        all_reasons.extend(urgency_reasons)

        final_score, risk_level, url_risk, sender_risk = calculate_risk(
            total_url_raw_score,
            sender_raw_score,
            urgency_risk
        )

        clean_reasons = list(dict.fromkeys(all_reasons))

        scan_summary = generate_scan_summary(
            sender,
            subject,
            final_score,
            risk_level,
            clean_reasons,
            len(urls),
            urgency_risk
        )

        save_scan(
            message["id"],
            sender,
            subject,
            final_score,
            risk_level,
            len(urls),
            urgency_risk,
            clean_reasons,
            urls,
            scan_summary
        )

        print("\n" + "=" * 80)
        print("Sender      :", sender)
        print("Subject     :", subject)
        print("URLs Found  :", len(urls))
        print("URL Risk    :", url_risk, "/35")
        print("Sender Risk :", sender_risk, "/25")
        print("Urgency Risk:", urgency_risk, "/15")
        print("Final Score :", final_score, "/100")
        print("Risk Level  :", risk_level)

        print("\nReasons:")
        if clean_reasons:
            for reason in clean_reasons:
                print("-", reason)
        else:
            print("- No major risk detected")

        print("\nSummary:")
        print(scan_summary)

        print("\nDatabase    : Scan saved/updated")
        print("=" * 80)


if __name__ == "__main__":
    run_scanner()