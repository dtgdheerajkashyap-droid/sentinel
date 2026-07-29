"""Unit tests for the SentinelAI feature extractors (not the ML model)."""

from sentinel_ai_v2 import (
    analyze_authentication,
    analyze_sender,
    analyze_url,
    analyze_urgency,
    num_urls_in_body,
)


# --------------------------------------------------
# TRUSTED DOMAIN SHORT-CIRCUIT
# --------------------------------------------------

def test_trusted_sender_short_circuits():
    features, reasons = analyze_sender("Google <no-reply@google.com>")

    assert features["is_trusted"] is True
    assert reasons == []
    assert features["suspicious_tld"] is False
    assert features["has_digit_in_domain"] is False


def test_trusted_url_short_circuits():
    features, reasons = analyze_url("https://accounts.google.com/signin")

    assert features["is_trusted"] is True
    assert reasons == []
    assert features["suspicious_tld"] is False


def test_trusted_subdomain_is_trusted():
    features, _ = analyze_sender("Reddit <noreply@mail.redditmail.com>")

    assert features["is_trusted"] is True


# --------------------------------------------------
# TYPOSQUAT DETECTION
# --------------------------------------------------

def test_sender_typosquat_detected():
    features, reasons = analyze_sender("PayPal <security@paypa1.com>")

    assert features["is_typosquat"] is True
    assert features["brand_distance"] == 1
    assert any("typosquat" in reason for reason in reasons)


def test_url_typosquat_detected():
    features, _ = analyze_url("http://micros0ft.com/login")

    assert features["is_typosquat"] is True
    assert features["brand_distance"] <= 2


def test_exact_brand_is_not_typosquat():
    # amazon.com is distance 0 from the brand — not a typosquat.
    features, _ = analyze_url("https://amazon.com/orders")

    assert features["is_typosquat"] is False
    assert features["brand_distance"] == 0


def test_unrelated_domain_is_not_typosquat():
    features, _ = analyze_sender("Newsletter <hello@substack.com>")

    assert features["is_typosquat"] is False


# --------------------------------------------------
# DIGIT IN DOMAIN
# --------------------------------------------------

def test_sender_digit_in_domain():
    features, reasons = analyze_sender("Alerts <admin@secure123bank.com>")

    assert features["has_digit_in_domain"] is True
    assert any("numbers" in reason for reason in reasons)


def test_url_digit_in_domain():
    features, _ = analyze_url("http://login44.net/verify")

    assert features["has_digit_in_domain"] is True


# --------------------------------------------------
# SPF / DKIM / DMARC
# --------------------------------------------------

def test_spf_fail_detected():
    headers = [{
        "name": "Authentication-Results",
        "value": (
            "mx.google.com; spf=fail (google.com: domain of x does not "
            "designate sender) smtp.mailfrom=x@evil.com; dkim=pass; "
            "dmarc=fail (p=REJECT)"
        ),
    }]

    features, reasons = analyze_authentication(headers)

    assert features["spf_pass"] is False
    assert features["dkim_pass"] is True
    assert features["dmarc_pass"] is False
    assert any("SPF" in reason for reason in reasons)
    assert any("DMARC" in reason for reason in reasons)


def test_missing_auth_header_returns_none_values():
    features, reasons = analyze_authentication([])

    assert features == {
        "spf_pass": None,
        "dkim_pass": None,
        "dmarc_pass": None,
    }
    assert reasons == []


def test_all_auth_pass_produces_no_reasons():
    headers = [{
        "name": "Authentication-Results",
        "value": "mx.google.com; spf=pass; dkim=pass; dmarc=pass",
    }]

    features, reasons = analyze_authentication(headers)

    assert features["spf_pass"] is True
    assert features["dkim_pass"] is True
    assert features["dmarc_pass"] is True
    assert reasons == []


# --------------------------------------------------
# CLEAN EMAIL — FALSE-POSITIVE CHECK
# --------------------------------------------------

def test_clean_email_has_no_risk_signals():
    sender_features, sender_reasons = analyze_sender(
        "Alice <alice@example.com>"
    )
    urgency_features, urgency_reasons = analyze_urgency(
        "Lunch tomorrow? Let me know if noon works for you."
    )

    assert sender_features["suspicious_tld"] is False
    assert sender_features["has_digit_in_domain"] is False
    assert sender_features["is_typosquat"] is False
    assert sender_reasons == []

    assert urgency_features["num_urgency_keywords"] == 0
    assert urgency_reasons == []

    assert num_urls_in_body("Lunch tomorrow? Let me know.") == 0


def test_urgency_keywords_counted():
    features, reasons = analyze_urgency(
        "URGENT: verify your account immediately"
    )

    assert features["num_urgency_keywords"] == 3
    assert set(features["matched_keywords"]) == {
        "urgent", "verify", "immediately"
    }
    assert len(reasons) == 3
