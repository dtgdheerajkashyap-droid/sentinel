# 🛡️ SentinelAI

### Real-Time Email Threat Intelligence Platform

SentinelAI is a full-stack email threat intelligence platform that connects to Gmail, scans emails for suspicious indicators, calculates an explainable risk score, and presents the results through an interactive security dashboard.

The system combines a Python-based email scanning engine, SQLite database, FastAPI backend, and Next.js frontend to provide a complete email security monitoring workflow.

---

## 🚀 Key Features

* 📧 **Gmail Inbox Scanning** — Scans emails from a connected Gmail account using the Gmail API and OAuth 2.0.
* 🎯 **Explainable Risk Scoring** — Generates a risk score from 0–100 based on indicators such as suspicious URLs, sender characteristics, urgency patterns, and other threat signals.
* 🚦 **Risk Classification** — Categorizes emails into:

  * 🟢 Safe
  * 🟡 Medium Risk
  * 🔴 High Risk
* 🔍 **Threat Analysis** — Stores detected URLs, risk reasons, scan summaries, and threat metadata.
* 📊 **Security Dashboard** — Provides risk distribution, scan trends, detection signals, threat timelines, and top risky sender analytics.
* 🛡️ **Threat Response Actions** — Allows users to mark scans as safe or quarantine suspicious emails.
* 🗄️ **Persistent Threat Storage** — Stores scan results and threat intelligence in SQLite.
* 🔌 **REST API Backend** — FastAPI endpoints connect the scanning engine, database, and frontend.
* 🧩 **Modular Architecture** — Designed for future integration of Machine Learning, NLP, URL intelligence, domain reputation, and brand impersonation detection.

---

## 📊 Evaluation

A reproducible benchmark of the scanner and three machine-learning alternatives on **77,252 public
e-mails** (the corpora behind the Kaggle *Phishing Email Dataset*), all scored on the same held-out
test split of 11,656 e-mails. The split is stratified and grouped by e-mail template, so no spam
campaign appears in both training and test data. Thresholds are chosen on the validation split only.

| Model (test split) | Precision | Recall | F1 | ROC-AUC | Latency (median, CPU) |
|---|---|---|---|---|---|
| Production rules (`sentinel_ai_v2.py`, score ≥ 30) | 0.394 | 0.004 | 0.009 | 0.540 | 0.36 ms |
| BERT-tiny SMS spam model, zero-shot | 0.504 | 0.561 | 0.531 | 0.508 | 9.3 ms |
| **BERT-tiny fine-tuned on e-mail** | **0.984** | **0.990** | **0.987** | **0.999** | 9.6 ms |
| Hybrid: rule URL/sender scores + 40 × BERT probability | 0.987 | 0.985 | 0.986 | 0.995 | 10.3 ms |
| Reference: TF-IDF + logistic regression | 0.987 | 0.990 | 0.988 | 0.999 | – |

What the results show:

- **The fine-tuned model catches 99% of phishing and spam** while flagging 1.6% of legitimate mail. The hybrid
  lowers false positives by 17% (92 → 76) at the cost of 27 more misses.
- **The benchmark is easier than it looks.** A bag-of-words baseline matches BERT, and the source corpus of an
  e-mail can be predicted with 95% accuracy. These scores describe 1999–2008 corpora, not a modern inbox.
- **The rule engine almost never fires on this data.** Most URLs in the release are stripped or space-split, the
  2008-era spam domains don't use the `.xyz`/`.top` TLDs the rules look for, and the trusted-domain list is
  tuned to one personal inbox.

Full report with dataset statistics, confusion matrices, PR curves, error analysis, leakage checks and
limitations: [`reports/results.md`](reports/results.md).

Reproduce (about 85 minutes on a laptop CPU; downloads the data and verifies checksums):

```
pip install -r requirements-eval.txt
python eval/run_all.py
```

---

## 🏗️ System Architecture

```text
┌──────────────────────┐
│      Gmail Inbox     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Gmail API + OAuth  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Python Email       │
│   Threat Scanner     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Risk Scoring Engine│
│                      │
│ • URL Analysis       │
│ • Sender Analysis    │
│ • Urgency Detection  │
│ • Threat Indicators  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    Risk Score 0–100  │
│  Safe / Medium / High│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    SQLite Database   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    FastAPI Backend   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    Next.js Dashboard │
└──────────────────────┘
```

---

## 🧠 Risk Analysis Pipeline

```text
Email
  ↓
Extract Sender, Subject, Body and URLs
  ↓
Analyze Threat Indicators
  ↓
Calculate Risk Components
  ↓
Generate 0–100 Risk Score
  ↓
Assign Risk Level
  ↓
Store Scan Results
  ↓
Display Threat Intelligence Dashboard
```

---

## 🛠️ Tech Stack

### Backend & Detection

* **Python**
* **FastAPI**
* **SQLite**
* **Gmail API**
* **OAuth 2.0**
* **REST APIs**

### Frontend

* **Next.js**
* **React**
* **TypeScript**
* **Tailwind CSS**
* **Recharts**

### Development Tools

* **Git**
* **GitHub**
* **Uvicorn**
* **Node.js / npm**

---

## 📁 Project Structure

```text
sentinelAI/
│
├── api.py
├── sentinel_ai_v2.py
├── sentinelai.db
│
├── config/
│   ├── credentials.json
│   └── token.pickle
│
├── sentinelai-frontend/
│   ├── app/
│   ├── components/
│   ├── lib/
│   ├── public/
│   └── package.json
│
└── README.md
```

> ⚠️ OAuth credentials, access tokens, databases, environment files, and other sensitive files should not be committed to a public repository.

---

## 🔄 Application Workflow

### 1. Connect Gmail

The user authorizes SentinelAI to access the Gmail inbox using OAuth 2.0.

### 2. Scan Inbox

The FastAPI backend triggers the Python email scanning engine.

### 3. Analyze Emails

The scanner extracts:

* Sender
* Subject
* Email body
* URLs
* Urgency indicators
* Other suspicious signals

### 4. Calculate Risk

The detected indicators contribute to the final risk score.

```text
Risk Score: 0–100
```

### 5. Store Results

The scan results are stored in SQLite, including:

* Sender
* Subject
* Risk score
* Risk level
* URLs found
* Risk reasons
* Scan summary
* Action status

### 6. Display Intelligence

The Next.js dashboard retrieves the data through FastAPI and displays security analytics.

---

## 🔌 Main API Endpoints

| Endpoint                     | Method | Description                      |
| ---------------------------- | ------ | -------------------------------- |
| `/`                          | GET    | API health check                 |
| `/api/stats`                 | GET    | Retrieve overall scan statistics |
| `/api/scans`                 | GET    | Retrieve email scan history      |
| `/api/scan-inbox`            | POST   | Scan the connected Gmail inbox   |
| `/api/top-risky-senders`     | GET    | Retrieve top risky senders       |
| `/api/scans/{id}/mark-safe`  | POST   | Mark a scan as safe              |
| `/api/scans/{id}/quarantine` | POST   | Quarantine a suspicious scan     |

---

## 🛡️ Security Considerations

SentinelAI uses OAuth 2.0 for Gmail authentication rather than storing Gmail passwords.

Sensitive files must be excluded from version control:

```text
credentials.json
token.pickle
.env
*.db
```

These files should be added to `.gitignore` before pushing the project to a public repository.

---

## 🔮 Future Enhancements

The current modular architecture can be extended with:

* 🤖 Machine Learning-based phishing classification
* 🧠 Natural Language Processing for threat and urgency analysis
* 🌐 Domain reputation analysis
* 🔗 Advanced URL intelligence
* 🎭 Brand impersonation detection
* 🔄 Automated model retraining and drift detection
* 📧 Multi-channel threat detection for SMS and messaging platforms
* 🧩 RAG-based security policy reasoning

---

## 🎯 Project Objective

SentinelAI aims to provide an explainable and extensible email security platform capable of identifying suspicious email threats, presenting understandable risk explanations, and providing actionable threat-response workflows through a centralized security dashboard.

---

## 👨‍💻 Built With

**Python • FastAPI • Gmail API • OAuth 2.0 • SQLite • Next.js • React • TypeScript • Tailwind CSS • Recharts**

---
