from __future__ import annotations

import pandas as pd


def build_model_frame(transactions: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    df = transactions.merge(accounts, on="account_id", how="left")

    df["is_large_amount"] = (df["amount_usd"] >= 1000).astype(int)
    df["login_pressure"] = pd.cut(
        df["failed_logins_24h"],
        bins=[-1, 0, 2, 100],
        labels=["none", "low", "high"]
    )

    return df
