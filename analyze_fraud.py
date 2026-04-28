from __future__ import annotations

from pathlib import Path

import pandas as pd

from features import build_model_frame
from risk_rules import label_risk, score_transaction


DATA_DIR = Path(__file__).resolve().parent


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    accounts = pd.read_csv(DATA_DIR / "accounts.csv")
    transactions = pd.read_csv(DATA_DIR / "transactions.csv")
    chargebacks = pd.read_csv(DATA_DIR / "chargebacks.csv")
    return accounts, transactions, chargebacks


def score_transactions(transactions: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    model_frame = build_model_frame(transactions, accounts)
    model_frame["risk_score"] = model_frame.apply(
        lambda row: score_transaction(row.to_dict()), axis=1
    )
    model_frame["risk_label"] = model_frame["risk_score"].apply(label_risk)
    return model_frame


def summarize_results(scored: pd.DataFrame, chargebacks: pd.DataFrame) -> pd.DataFrame:
    summary = (
        scored.groupby("risk_label", as_index=False)
        .agg(
            transactions=("transaction_id", "count"),
            total_amount_usd=("amount_usd", "sum"),
            avg_amount_usd=("amount_usd", "mean"),
        )
        .sort_values("risk_label")
    )

    known_fraud = scored.merge(chargebacks[["transaction_id"]], on="transaction_id", how="left", indicator=True)
    known_fraud["is_chargeback"] = (known_fraud["_merge"] == "both").astype(int)

    fraud_by_label = (
        known_fraud.groupby("risk_label", as_index=False)
        .agg(
            chargebacks=("is_chargeback", "sum")
        )
    )

    summary = summary.merge(fraud_by_label, on="risk_label", how="left")
    summary["chargeback_rate"] = summary["chargebacks"] / summary["transactions"]
    return summary


def main() -> None:
    accounts, transactions, chargebacks = load_inputs()
    scored = score_transactions(transactions, accounts)

    print("\nTop 10 scored transactions\n")
    print(
        scored[
            [
                "transaction_id",
                "account_id",
                "amount_usd",
                "device_risk_score",
                "is_international",
                "velocity_24h",
                "prior_chargebacks",
                "risk_score",
                "risk_label",
            ]
        ]
        .sort_values(["risk_score", "amount_usd"], ascending=[False, False])
        .head(10)
        .to_string(index=False)
    )

    print("\nRisk summary\n")
    print(summarize_results(scored, chargebacks).to_string(index=False))


if __name__ == "__main__":
    main()
