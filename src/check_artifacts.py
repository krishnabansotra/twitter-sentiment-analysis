"""Count how often complaint-reason labels appear inside tweet text (label leakage check)."""
import pandas as pd

d = pd.read_csv("data/processed/airline.csv")
for k in ["Cancelled Flight", "Late Flight", "Flight Booking Problems", "Customer Service Issue",
          "Lost Luggage", "Bad Flight", "Damaged Luggage", "Flight Attendant Complaints",
          "longlines", "Can't Tell"]:
    print(f"{k:30s} {d['text'].str.contains(k, case=False, regex=False).sum()}")