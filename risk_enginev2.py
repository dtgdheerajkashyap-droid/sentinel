def calculate_risk(
    bert_risk,
    url_risk,
    sender_risk
):

    final_score = (
        bert_risk +
        url_risk +
        sender_risk
    )

    if final_score >= 80:
        level = "HIGH RISK"

    elif final_score >= 50:
        level = "MEDIUM RISK"

    else:
        level = "LOW RISK"

    return final_score, level


# Example values

bert_risk = 35      # max 40
url_risk = 30       # max 35
sender_risk = 20    # max 25

score, level = calculate_risk(
    bert_risk,
    url_risk,
    sender_risk
)

print("\nFINAL SCORE:", score, "/100")
print("RISK LEVEL:", level)