def calculate_risk(url_score, sender_score):

    total_score = url_score + sender_score

    if total_score >= 80:
        level = "HIGH RISK"

    elif total_score >= 40:
        level = "MEDIUM RISK"

    else:
        level = "LOW RISK"

    return total_score, level


url_score = 70
sender_score = 70

risk_score, risk_level = calculate_risk(
    url_score,
    sender_score
)

print("\nFINAL RISK SCORE:", risk_score)
print("RISK LEVEL:", risk_level)