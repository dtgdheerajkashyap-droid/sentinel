"""Feature vector assembly for SentinelAI.

Combines the raw feature extractors in sentinel_ai_v2.py with the BERT
spam classifier's phishing probability into a single ordered feature
vector per email, consumed by train_model.py and the ML risk engine.

FEATURE SCHEMA
--------------
Order is significant: the trained model expects features in exactly the
order defined by FEATURE_SCHEMA below.

| name                    | type  | description                                        |
|-------------------------|-------|----------------------------------------------------|
| sender_is_trusted       | bool  | Sender domain is in TRUSTED_DOMAINS                |
| sender_suspicious_tld   | bool  | Sender domain uses a suspicious TLD                |
| sender_has_digit        | bool  | Sender domain contains a digit                     |
| sender_domain_length    | int   | Length of the sender domain string                 |
| sender_brand_distance   | int   | Min edit distance from sender domain to a brand    |
| sender_is_typosquat     | bool  | Sender domain within typosquat range of a brand    |
| sender_spf_pass         | float | SPF result: 1 pass, 0 fail, 0.5 absent             |
| sender_dkim_pass        | float | DKIM result: 1 pass, 0 fail, 0.5 absent            |
| sender_dmarc_pass       | float | DMARC result: 1 pass, 0 fail, 0.5 absent           |
| num_urls_in_body        | int   | Distinct non-ignored URLs in the body              |
| num_trusted_urls        | int   | URLs whose domain is trusted                       |
| num_suspicious_tld_urls | int   | URLs with a suspicious TLD                         |
| num_digit_domain_urls   | int   | URLs whose domain contains a digit                 |
| num_typosquat_urls      | int   | URLs within typosquat range of a brand             |
| num_redirect_urls       | int   | Redirect hops found across all URLs                |
| max_url_domain_length   | int   | Longest URL domain seen                            |
| min_url_brand_distance  | int   | Min brand edit distance across URL domains         |
| num_urgency_keywords    | int   | Urgency keyword hits in subject + body             |
| body_length             | int   | Character length of the extracted body             |
| bert_phishing_prob      | float | BERT classifier spam/phishing probability [0, 1]   |
"""

from sentinel_ai_v2 import (
    analyze_authentication,
    analyze_sender,
    analyze_url,
    analyze_urgency,
    extract_urls,
)


# Ordered (name, type, description) tuples — the canonical schema.
FEATURE_SCHEMA = [
    ("sender_is_trusted", "bool", "Sender domain is in TRUSTED_DOMAINS"),
    ("sender_suspicious_tld", "bool", "Sender domain uses a suspicious TLD"),
    ("sender_has_digit", "bool", "Sender domain contains a digit"),
    ("sender_domain_length", "int", "Length of the sender domain string"),
    ("sender_brand_distance", "int",
     "Min edit distance from sender domain to a known brand"),
    ("sender_is_typosquat", "bool",
     "Sender domain within typosquat range of a brand"),
    ("sender_spf_pass", "float", "SPF result: 1 pass, 0 fail, 0.5 absent"),
    ("sender_dkim_pass", "float", "DKIM result: 1 pass, 0 fail, 0.5 absent"),
    ("sender_dmarc_pass", "float", "DMARC result: 1 pass, 0 fail, 0.5 absent"),
    ("num_urls_in_body", "int", "Distinct non-ignored URLs in the body"),
    ("num_trusted_urls", "int", "URLs whose domain is trusted"),
    ("num_suspicious_tld_urls", "int", "URLs with a suspicious TLD"),
    ("num_digit_domain_urls", "int", "URLs whose domain contains a digit"),
    ("num_typosquat_urls", "int",
     "URLs within typosquat range of a brand"),
    ("num_redirect_urls", "int", "Redirect hops found across all URLs"),
    ("max_url_domain_length", "int", "Longest URL domain seen"),
    ("min_url_brand_distance", "int",
     "Min brand edit distance across URL domains"),
    ("num_urgency_keywords", "int", "Urgency keyword hits in subject + body"),
    ("body_length", "int", "Character length of the extracted body"),
    ("bert_phishing_prob", "float",
     "BERT classifier spam/phishing probability [0, 1]"),
]

FEATURE_NAMES = [name for name, _, _ in FEATURE_SCHEMA]

# Sentinel used when no URLs are present (matches BRAND_DISTANCE_NONE).
NO_URL_BRAND_DISTANCE = 99

# Lazily loaded BERT pipeline (transformers import is expensive).
_bert_classifier = None


def get_bert_classifier():
    global _bert_classifier

    if _bert_classifier is None:
        from transformers import pipeline

        _bert_classifier = pipeline(
            "text-classification",
            model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
        )

    return _bert_classifier


def bert_phishing_probability(text, classifier=None):
    """Probability that `text` is spam/phishing according to the BERT model.

    The SMS spam model labels: LABEL_0 = ham, LABEL_1 = spam. Returns a
    float in [0, 1] regardless of which label the pipeline reports.
    """
    if not text.strip():
        return 0.0

    if classifier is None:
        classifier = get_bert_classifier()

    prediction = classifier(text[:512])[0]

    label = prediction["label"].upper()
    score = prediction["score"]

    if label in ("LABEL_1", "SPAM"):
        return score

    return 1.0 - score


def _auth_to_float(value):
    """Map True/False/None auth results to 1.0/0.0/0.5."""
    if value is None:
        return 0.5

    return 1.0 if value else 0.0


def build_feature_vector(sender, subject, body, headers=None,
                         bert_prob=None, classifier=None):
    """Assemble the ordered feature dict for one email.

    Args:
        sender: raw From header value.
        subject: subject line.
        body: extracted plain-text body.
        headers: Gmail API header dicts (for Authentication-Results);
            may be None when unavailable (e.g. offline training data).
        bert_prob: precomputed BERT probability. When None, the BERT
            pipeline is invoked on subject + body.
        classifier: optional BERT pipeline override (used by tests).

    Returns:
        dict keyed and ordered by FEATURE_NAMES.
    """
    sender_features, _ = analyze_sender(sender)
    auth_features, _ = analyze_authentication(headers)
    urgency_features, _ = analyze_urgency(subject + " " + body)

    urls = extract_urls(body)
    url_features_list = [analyze_url(url)[0] for url in urls]
    url_features_list = [f for f in url_features_list if not f["ignored"]]

    if bert_prob is None:
        bert_prob = bert_phishing_probability(
            subject + " " + body, classifier=classifier
        )

    features = {
        "sender_is_trusted": sender_features["is_trusted"],
        "sender_suspicious_tld": sender_features["suspicious_tld"],
        "sender_has_digit": sender_features["has_digit_in_domain"],
        "sender_domain_length": sender_features["domain_length"],
        "sender_brand_distance": sender_features["brand_distance"],
        "sender_is_typosquat": sender_features["is_typosquat"],
        "sender_spf_pass": _auth_to_float(auth_features["spf_pass"]),
        "sender_dkim_pass": _auth_to_float(auth_features["dkim_pass"]),
        "sender_dmarc_pass": _auth_to_float(auth_features["dmarc_pass"]),
        "num_urls_in_body": len(url_features_list),
        "num_trusted_urls": sum(
            f["is_trusted"] for f in url_features_list
        ),
        "num_suspicious_tld_urls": sum(
            f["suspicious_tld"] for f in url_features_list
        ),
        "num_digit_domain_urls": sum(
            f["has_digit_in_domain"] for f in url_features_list
        ),
        "num_typosquat_urls": sum(
            f["is_typosquat"] for f in url_features_list
        ),
        "num_redirect_urls": sum(
            f["num_redirects"] for f in url_features_list
        ),
        "max_url_domain_length": max(
            (f["domain_length"] for f in url_features_list),
            default=0,
        ),
        "min_url_brand_distance": min(
            (f["brand_distance"] for f in url_features_list),
            default=NO_URL_BRAND_DISTANCE,
        ),
        "num_urgency_keywords": urgency_features["num_urgency_keywords"],
        "body_length": len(body),
        "bert_phishing_prob": float(bert_prob),
    }

    return {name: features[name] for name in FEATURE_NAMES}


def feature_vector_to_array(features):
    """Convert a feature dict to a flat numeric list in schema order."""
    return [float(features[name]) for name in FEATURE_NAMES]
