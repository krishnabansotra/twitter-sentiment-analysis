"""Draft replies. These are suggestions for a human to review, never sent automatically."""

TEMPLATES = {
    "Late Flight": "We're sorry about the delay. We know how disruptive it is. Please DM us your booking reference so we can check your options.",
    "Cancelled Flight": "We're very sorry your flight was cancelled. Please DM your booking reference and we'll help with rebooking or a refund right away.",
    "Lost Luggage": "We're sorry your bag hasn't arrived. Please DM your bag tag number and contact details so our baggage team can trace it urgently.",
    "Damaged Luggage": "We're sorry your luggage was damaged. Please DM photos and your booking reference so we can start a claim.",
    "Customer Service Issue": "We're sorry for the experience you had. Please DM us the details so we can look into it and make it right.",
    "Flight Attendant Complaints": "We're sorry about your experience with our crew. Please DM your flight number and date so we can review this.",
    "Flight Booking Problems": "Sorry for the trouble with your booking. Please DM your booking reference and we'll sort it out.",
    "Bad Flight": "We're sorry your flight wasn't up to standard. Please DM us your flight details so we can pass your feedback to the team.",
    "longlines": "We're sorry about the long wait. Thanks for letting us know; we're sharing this with our airport team.",
    "Can't Tell": "We're sorry you had a poor experience. Please DM us more details so we can help.",
    "Unclear": "We're sorry you had a poor experience. Please DM us more details so we can help.",
}


def suggest_reply(sentiment: str, issue: str | None = None) -> str:
    if sentiment == "positive":
        return "Thank you for the kind words! We're glad you had a good experience."
    if sentiment == "neutral":
        return "Thanks for reaching out. Please DM us if you need any help."
    return TEMPLATES.get(issue, TEMPLATES["Unclear"])