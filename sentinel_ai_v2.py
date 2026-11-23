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
MODEL_PATH = os.path.join(ROOT_DIR, "model.pkl")

# ML probability bands: predict_proba >= HIGH → HIGH RISK,
# >= MEDIUM → MEDIUM RISK, otherwise LOW RISK.
RISK_HIGH_THRESHOLD = 0.70
RISK_MEDIUM_THRESHOLD = 0.30


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


# Bare brand names (no TLD) used for typosquat detection via edit distance.
# Covers major global brands plus banks present in TRUSTED_DOMAINS.
BRAND_NAMES = [
    "google",
    "gmail",
    "paypal",
    "amazon",
    "microsoft",
    "canarabank",
]

# Edit distance 1..TYPOSQUAT_MAX_DISTANCE from a brand name counts as a
# possible typosquat (distance 0 means the brand itself, handled by the
# trusted-domain check instead).
TYPOSQUAT_MAX_DISTANCE = 2

# Sentinel value for "no meaningful brand distance" (empty/very short domain).
BRAND_DISTANCE_NONE = 99

URGENCY_KEYWORDS = {
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


# --------------------------------------------------
# EDIT DISTANCE (python-Levenshtein with pure fallback)
# --------------------------------------------------

try:
    import Levenshtein

    def edit_distance(a, b):
        return Levenshtein.distance(a, b)

except ImportError:

    def edit_distance(a, b):
        if len(a) < len(b):
            a, b = b, a

        previous = list(range(len(b) + 1))

        for i, char_a in enumerate(a):
            current = [i + 1]
            for j, char_b in enumerate(b):
                current.append(min(
                    previous[j + 1] + 1,
                    current[j] + 1,
                    previous[j] + (char_a != char_b),
                ))
            previous = current

        return previous[-1]


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
        try:
            creds.refresh(Request())
        except Exception:
            # Token expired or revoked - force full re-authentication
            creds = None

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

    try:
        parsed = urlparse(url)
    except ValueError:
        # Malformed URL (e.g. stray brackets parsed as IPv6) — return as-is.
        return url

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


def num_urls_in_body(body):
    """Count of distinct non-ignored URLs found in the email body."""
    return len(extract_urls(body))


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
# TYPOSQUAT DETECTION
# --------------------------------------------------

def brand_edit_distance(domain_name):
    """Minimum edit distance from a bare domain name to any known brand.

    Returns BRAND_DISTANCE_NONE when the domain is empty or too short to
    compare meaningfully.
    """
    domain_name = (domain_name or "").lower()

    if len(domain_name) < 4:
        return BRAND_DISTANCE_NONE

    return min(edit_distance(domain_name, brand) for brand in BRAND_NAMES)


def is_typosquat(domain_name):
    """True when the domain is close to a brand name but not exactly it."""
    distance = brand_edit_distance(domain_name)
    return 1 <= distance <= TYPOSQUAT_MAX_DISTANCE


# --------------------------------------------------
# URL FEATURE EXTRACTION
# --------------------------------------------------

def analyze_url(url):
    """Extract raw features from a URL.

    Returns (features, reasons). Features are typed raw values
    (booleans/counts), not point scores; scoring happens in the risk
    engine (ML model or heuristic fallback).
    """
    url = clean_url(url)

    features = {
        "ignored": False,
        "is_trusted": False,
        "suspicious_tld": False,
        "has_digit_in_domain": False,
        "domain_length": 0,
        "brand_distance": BRAND_DISTANCE_NONE,
        "is_typosquat": False,
        "num_redirects": 0,
    }

    if should_ignore_url(url):
        features["ignored"] = True
        return features, []

    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()

    if hostname.startswith("www."):
        hostname = hostname[4:]

    ext = tldextract.extract(hostname)

    domain = ext.domain.lower()
    suffix = ext.suffix.lower()
    full_domain = f"{domain}.{suffix}" if suffix else domain

    features["domain_length"] = len(domain)
    features["brand_distance"] = brand_edit_distance(domain)

    if is_trusted_domain(full_domain):
        features["is_trusted"] = True
        return features, []

    reasons = []

    if suffix in SUSPICIOUS_TLDS:
        features["suspicious_tld"] = True
        reasons.append(f"Suspicious URL TLD: .{suffix}")

    if any(char.isdigit() for char in domain):
        features["has_digit_in_domain"] = True
        reasons.append("URL domain contains numbers")

    if len(domain) > 20:
        reasons.append("Unusually long URL domain")

    if is_typosquat(domain):
        features["is_typosquat"] = True
        reasons.append(
            f"URL domain '{domain}' looks like a typosquat of a known brand"
        )

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
                redirect_features, redirect_reasons = analyze_url(
                    redirected_url
                )
                features["num_redirects"] += 1 + redirect_features[
                    "num_redirects"
                ]
                features["suspicious_tld"] |= redirect_features[
                    "suspicious_tld"
                ]
                features["has_digit_in_domain"] |= redirect_features[
                    "has_digit_in_domain"
                ]
                features["is_typosquat"] |= redirect_features["is_typosquat"]
                features["brand_distance"] = min(
                    features["brand_distance"],
                    redirect_features["brand_distance"],
                )
                reasons.extend(redirect_reasons)

    return features, reasons


def url_heuristic_score(features):
    """Legacy additive score for a single URL's features (fallback path)."""
    if features["ignored"] or features["is_trusted"]:
        return 0

    score = 0

    if features["suspicious_tld"]:
        score += 40

    if features["has_digit_in_domain"]:
        score += 30

    if features["domain_length"] > 20:
        score += 20

    if features["is_typosquat"]:
        score += 40

    return score


# --------------------------------------------------
# SENDER FEATURE EXTRACTION
# --------------------------------------------------

def analyze_sender(sender):
    """Extract raw features from the From address.

    Returns (features, reasons) — raw typed values, no point scores.
    """
    sender = sender.lower()
    sender_domain = get_domain_from_email(sender)

    ext = tldextract.extract(sender_domain)
    domain_name = ext.domain.lower()

    features = {
        "is_trusted": False,
        "suspicious_tld": False,
        "has_digit_in_domain": False,
        "domain_length": len(sender_domain),
        "brand_distance": brand_edit_distance(domain_name),
        "is_typosquat": False,
    }

    if sender_domain and is_trusted_domain(sender_domain):
        features["is_trusted"] = True
        return features, []

    reasons = []

    for tld in SUSPICIOUS_TLDS:
        if sender_domain.endswith("." + tld):
            features["suspicious_tld"] = True
            reasons.append(f"Suspicious sender TLD: .{tld}")
            break

    if re.search(r"\d", sender_domain):
        features["has_digit_in_domain"] = True
        reasons.append("Sender domain contains numbers")

    if len(sender_domain) > 35:
        reasons.append("Unusually long sender domain")

    if is_typosquat(domain_name):
        features["is_typosquat"] = True
        reasons.append(
            f"Sender domain '{sender_domain}' looks like a typosquat "
            f"of a known brand"
        )

    return features, reasons


def sender_heuristic_score(features):
    """Legacy additive score for sender features (fallback path)."""
    if features["is_trusted"]:
        return 0

    score = 0

    if features["suspicious_tld"]:
        score += 40

    if features["has_digit_in_domain"]:
        score += 30

    if features["domain_length"] > 35:
        score += 10

    if features["is_typosquat"]:
        score += 40

    return score


# --------------------------------------------------
# EMAIL AUTHENTICATION (SPF / DKIM / DMARC)
# --------------------------------------------------

def analyze_authentication(headers):
    """Parse Gmail's Authentication-Results header into pass/fail features.

    `headers` is the list of {name, value} dicts from the Gmail API payload.
    Each feature is True (pass), False (explicit non-pass), or None when the
    mechanism is absent from the header.
    """
    features = {
        "spf_pass": None,
        "dkim_pass": None,
        "dmarc_pass": None,
    }

    auth_results = ""

    for header in headers or []:
        if header.get("name", "").lower() == "authentication-results":
            auth_results = header.get("value", "").lower()
            break

    if not auth_results:
        return features, []

    reasons = []

    for mechanism in ("spf", "dkim", "dmarc"):
        match = re.search(rf"\b{mechanism}=(\w+)", auth_results)

        if match:
            passed = match.group(1) == "pass"
            features[f"{mechanism}_pass"] = passed

            if not passed:
                reasons.append(
                    f"{mechanism.upper()} check failed ({match.group(1)})"
                )

    return features, reasons


# --------------------------------------------------
# URGENCY LANGUAGE FEATURE EXTRACTION
# --------------------------------------------------

def analyze_urgency(text):
    """Extract urgency-language features from subject + body text.

    Returns (features, reasons) — keyword hit counts, no point scores.
    """
    text = text.lower()

    matched_keywords = [word for word in URGENCY_KEYWORDS if word in text]

    features = {
        "num_urgency_keywords": len(matched_keywords),
        "matched_keywords": matched_keywords,
    }

    reasons = [f"Contains '{word}'" for word in matched_keywords]

    return features, reasons


def urgency_heuristic_score(features):
    """Legacy capped additive score for urgency features (fallback path)."""
    score = sum(
        URGENCY_KEYWORDS[word] for word in features["matched_keywords"]
    )
    return min(score, 15)


# --------------------------------------------------
# FINAL RISK ENGINE
# --------------------------------------------------

def _load_risk_model():
    """Load the trained model saved by train_model.py, or None if absent."""
    if not os.path.exists(MODEL_PATH):
        return None

    try:
        with open(MODEL_PATH, "rb") as model_file:
            payload = pickle.load(model_file)

        return payload

    except Exception as error:
        print(f"Warning: could not load {MODEL_PATH} ({error}); "
              f"falling back to heuristic scoring.")
        return None


RISK_MODEL = _load_risk_model()


def _heuristic_risk(url_score_raw, sender_score_raw, urgency_risk):
    """Original additive-weight scoring, kept as the no-model fallback."""
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


def calculate_risk(url_score_raw, sender_score_raw, urgency_risk,
                   features=None):
    """Compute (final_score, level, url_risk, sender_risk) for an email.

    When a trained model (model.pkl) is available and a feature vector is
    provided, risk comes from model.predict_proba() mapped to bands via
    RISK_HIGH_THRESHOLD / RISK_MEDIUM_THRESHOLD. Otherwise the original
    heuristic scoring is used, so the scanner runs without a trained model.

    The first three arguments and the return shape are unchanged for
    backward compatibility.
    """
    heuristic = _heuristic_risk(url_score_raw, sender_score_raw, urgency_risk)

    if RISK_MODEL is None or features is None:
        return heuristic

    try:
        feature_row = [
            [float(features[name]) for name in RISK_MODEL["feature_names"]]
        ]
        probability = RISK_MODEL["model"].predict_proba(feature_row)[0][1]

    except Exception as error:
        print(f"Warning: model prediction failed ({error}); "
              f"using heuristic scoring.")
        return heuristic

    final_score = round(probability * 100)

    if probability >= RISK_HIGH_THRESHOLD:
        level = "HIGH RISK"
    elif probability >= RISK_MEDIUM_THRESHOLD:
        level = "MEDIUM RISK"
    else:
        level = "LOW RISK"

    # url_risk / sender_risk sub-scores remain heuristic-derived for
    # display and DB compatibility.
    return final_score, level, heuristic[2], heuristic[3]


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

        all_reasons = []
        url_features_list = []

        for url in urls:
            url_features, reasons = analyze_url(url)
            url_features_list.append(url_features)
            all_reasons.extend(reasons)

        sender_features, sender_reasons = analyze_sender(sender)
        all_reasons.extend(sender_reasons)

        auth_features, auth_reasons = analyze_authentication(headers)
        all_reasons.extend(auth_reasons)

        urgency_features, urgency_reasons = analyze_urgency(
            subject + " " + body
        )
        all_reasons.extend(urgency_reasons)

        total_url_raw_score = sum(
            url_heuristic_score(f) for f in url_features_list
        )
        sender_raw_score = sender_heuristic_score(sender_features)
        urgency_risk = urgency_heuristic_score(urgency_features)

        features = None

        if RISK_MODEL is not None:
            try:
                # Imported lazily: feature_vector depends on this module.
                from feature_vector import build_feature_vector

                features = build_feature_vector(
                    sender, subject, body, headers=headers
                )

            except Exception as error:
                print(f"Warning: feature extraction failed ({error}); "
                      f"using heuristic scoring.")

        final_score, risk_level, url_risk, sender_risk = calculate_risk(
            total_url_raw_score,
            sender_raw_score,
            urgency_risk,
            features=features,
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