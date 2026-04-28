"""
Tests for fraud detection logic.

Covers:
- score_transaction(): exact point contributions, boundary values, clamping
- label_risk(): threshold boundaries
- build_model_frame(): merge and derived columns
- score_transactions() / summarize_results(): pipeline integration
"""
import pandas as pd
import pytest

from features import build_model_frame
from risk_rules import label_risk, score_transaction
from analyze_fraud import score_transactions, summarize_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TX = {
    "device_risk_score": 10,
    "is_international": 0,
    "amount_usd": 100,
    "velocity_24h": 1,
    "failed_logins_24h": 0,
    "prior_chargebacks": 0,
}


def _tx(**overrides):
    return {**BASE_TX, **overrides}


# ---------------------------------------------------------------------------
# label_risk — boundary values
# ---------------------------------------------------------------------------

class TestLabelRisk:
    def test_below_medium_threshold(self):
        assert label_risk(0) == "low"
        assert label_risk(29) == "low"

    def test_at_medium_threshold(self):
        assert label_risk(30) == "medium"
        assert label_risk(31) == "medium"

    def test_below_high_threshold(self):
        assert label_risk(59) == "medium"

    def test_at_high_threshold(self):
        assert label_risk(60) == "high"
        assert label_risk(100) == "high"


# ---------------------------------------------------------------------------
# score_transaction — each signal in isolation
# ---------------------------------------------------------------------------

class TestDeviceRiskScore:
    def test_low_device_risk_adds_nothing(self):
        assert score_transaction(_tx(device_risk_score=39)) == 0

    def test_medium_device_risk_adds_10(self):
        assert score_transaction(_tx(device_risk_score=40)) == 10
        assert score_transaction(_tx(device_risk_score=69)) == 10

    def test_high_device_risk_adds_25(self):
        assert score_transaction(_tx(device_risk_score=70)) == 25
        assert score_transaction(_tx(device_risk_score=100)) == 25

    def test_boundary_69_vs_70(self):
        assert score_transaction(_tx(device_risk_score=70)) > score_transaction(_tx(device_risk_score=69))


class TestInternational:
    def test_domestic_adds_nothing(self):
        assert score_transaction(_tx(is_international=0)) == 0

    def test_international_adds_15(self):
        assert score_transaction(_tx(is_international=1)) == 15


class TestAmount:
    def test_low_amount_adds_nothing(self):
        assert score_transaction(_tx(amount_usd=499)) == 0

    def test_medium_amount_adds_10(self):
        assert score_transaction(_tx(amount_usd=500)) == 10
        assert score_transaction(_tx(amount_usd=999)) == 10

    def test_large_amount_adds_25(self):
        assert score_transaction(_tx(amount_usd=1000)) == 25
        assert score_transaction(_tx(amount_usd=9999)) == 25

    def test_boundary_999_vs_1000(self):
        assert score_transaction(_tx(amount_usd=1000)) > score_transaction(_tx(amount_usd=999))


class TestVelocity:
    def test_low_velocity_adds_nothing(self):
        assert score_transaction(_tx(velocity_24h=2)) == 0

    def test_medium_velocity_adds_5(self):
        assert score_transaction(_tx(velocity_24h=3)) == 5
        assert score_transaction(_tx(velocity_24h=5)) == 5

    def test_high_velocity_adds_20(self):
        assert score_transaction(_tx(velocity_24h=6)) == 20
        assert score_transaction(_tx(velocity_24h=99)) == 20

    def test_boundary_5_vs_6(self):
        assert score_transaction(_tx(velocity_24h=6)) > score_transaction(_tx(velocity_24h=5))


class TestFailedLogins:
    def test_no_failures_adds_nothing(self):
        assert score_transaction(_tx(failed_logins_24h=1)) == 0

    def test_moderate_failures_adds_10(self):
        assert score_transaction(_tx(failed_logins_24h=2)) == 10
        assert score_transaction(_tx(failed_logins_24h=4)) == 10

    def test_high_failures_adds_20(self):
        assert score_transaction(_tx(failed_logins_24h=5)) == 20
        assert score_transaction(_tx(failed_logins_24h=20)) == 20

    def test_boundary_4_vs_5(self):
        assert score_transaction(_tx(failed_logins_24h=5)) > score_transaction(_tx(failed_logins_24h=4))


class TestPriorChargebacks:
    def test_no_chargebacks_adds_nothing(self):
        assert score_transaction(_tx(prior_chargebacks=0)) == 0

    def test_one_chargeback_adds_5(self):
        assert score_transaction(_tx(prior_chargebacks=1)) == 5

    def test_multiple_chargebacks_adds_20(self):
        assert score_transaction(_tx(prior_chargebacks=2)) == 20
        assert score_transaction(_tx(prior_chargebacks=10)) == 20

    def test_escalates_with_history(self):
        assert score_transaction(_tx(prior_chargebacks=2)) > score_transaction(_tx(prior_chargebacks=1))
        assert score_transaction(_tx(prior_chargebacks=1)) > score_transaction(_tx(prior_chargebacks=0))


# ---------------------------------------------------------------------------
# score_transaction — combined / clamping
# ---------------------------------------------------------------------------

class TestScoreCombined:
    def test_all_zero_signals_scores_zero(self):
        assert score_transaction(BASE_TX) == 0

    def test_score_clamped_at_100(self):
        tx = _tx(
            device_risk_score=80,
            is_international=1,
            amount_usd=2000,
            velocity_24h=10,
            failed_logins_24h=6,
            prior_chargebacks=3,
        )
        # Raw sum = 25+15+25+20+20+20 = 125; must clamp to 100
        assert score_transaction(tx) == 100

    def test_score_never_negative(self):
        assert score_transaction(BASE_TX) >= 0

    def test_known_combination(self):
        # device=75 (+25), international (+15), amount=600 (+10) = 50
        tx = _tx(device_risk_score=75, is_international=1, amount_usd=600)
        assert score_transaction(tx) == 50

    def test_worst_case_is_high_label(self):
        tx = _tx(
            device_risk_score=80,
            is_international=1,
            amount_usd=2000,
            velocity_24h=10,
            failed_logins_24h=6,
            prior_chargebacks=3,
        )
        assert label_risk(score_transaction(tx)) == "high"

    def test_clean_low_value_tx_is_low_label(self):
        assert label_risk(score_transaction(BASE_TX)) == "low"


# ---------------------------------------------------------------------------
# build_model_frame
# ---------------------------------------------------------------------------

class TestBuildModelFrame:
    def _make_frames(self):
        transactions = pd.DataFrame([
            {"transaction_id": 1, "account_id": 10, "amount_usd": 1500},
            {"transaction_id": 2, "account_id": 10, "amount_usd": 200},
            {"transaction_id": 3, "account_id": 99, "amount_usd": 800},  # no matching account
        ])
        accounts = pd.DataFrame([
            {"account_id": 10, "customer_name": "Alice"},
        ])
        return transactions, accounts

    def test_returns_all_transactions(self):
        txns, accts = self._make_frames()
        df = build_model_frame(txns, accts)
        assert len(df) == 3

    def test_is_large_amount_flag(self):
        txns, accts = self._make_frames()
        df = build_model_frame(txns, accts)
        row_large = df[df["transaction_id"] == 1].iloc[0]
        row_small = df[df["transaction_id"] == 2].iloc[0]
        assert row_large["is_large_amount"] == 1
        assert row_small["is_large_amount"] == 0

    def test_is_large_amount_boundary(self):
        txns = pd.DataFrame([
            {"transaction_id": 1, "account_id": 10, "amount_usd": 999},
            {"transaction_id": 2, "account_id": 10, "amount_usd": 1000},
        ])
        accts = pd.DataFrame([{"account_id": 10}])
        df = build_model_frame(txns, accts)
        assert df[df["transaction_id"] == 1].iloc[0]["is_large_amount"] == 0
        assert df[df["transaction_id"] == 2].iloc[0]["is_large_amount"] == 1

    def test_account_columns_merged(self):
        txns, accts = self._make_frames()
        df = build_model_frame(txns, accts)
        assert "customer_name" in df.columns

    def test_unmatched_account_produces_nan(self):
        txns, accts = self._make_frames()
        df = build_model_frame(txns, accts)
        unmatched = df[df["account_id"] == 99].iloc[0]
        assert pd.isna(unmatched["customer_name"])


# ---------------------------------------------------------------------------
# Pipeline: score_transactions and summarize_results
# ---------------------------------------------------------------------------

def _make_pipeline_data():
    transactions = pd.DataFrame([
        {
            "transaction_id": 1, "account_id": 10, "amount_usd": 50,
            "device_risk_score": 10, "is_international": 0,
            "velocity_24h": 1, "failed_logins_24h": 0, "prior_chargebacks": 0,
        },
        {
            "transaction_id": 2, "account_id": 10, "amount_usd": 2000,
            "device_risk_score": 80, "is_international": 1,
            "velocity_24h": 8, "failed_logins_24h": 6, "prior_chargebacks": 2,
        },
    ])
    accounts = pd.DataFrame([{"account_id": 10}])
    chargebacks = pd.DataFrame([{"transaction_id": 2}])
    return transactions, accounts, chargebacks


class TestScoreTransactions:
    def test_adds_risk_score_column(self):
        txns, accts, _ = _make_pipeline_data()
        scored = score_transactions(txns, accts)
        assert "risk_score" in scored.columns

    def test_adds_risk_label_column(self):
        txns, accts, _ = _make_pipeline_data()
        scored = score_transactions(txns, accts)
        assert "risk_label" in scored.columns

    def test_low_risk_tx_labelled_correctly(self):
        txns, accts, _ = _make_pipeline_data()
        scored = score_transactions(txns, accts)
        row = scored[scored["transaction_id"] == 1].iloc[0]
        assert row["risk_label"] == "low"

    def test_high_risk_tx_labelled_correctly(self):
        txns, accts, _ = _make_pipeline_data()
        scored = score_transactions(txns, accts)
        row = scored[scored["transaction_id"] == 2].iloc[0]
        assert row["risk_label"] == "high"

    def test_row_count_preserved(self):
        txns, accts, _ = _make_pipeline_data()
        scored = score_transactions(txns, accts)
        assert len(scored) == len(txns)


class TestSummarizeResults:
    def test_summary_has_expected_columns(self):
        txns, accts, cbs = _make_pipeline_data()
        scored = score_transactions(txns, accts)
        summary = summarize_results(scored, cbs)
        for col in ("risk_label", "transactions", "total_amount_usd", "chargeback_rate"):
            assert col in summary.columns

    def test_chargeback_rate_high_risk_is_one(self):
        txns, accts, cbs = _make_pipeline_data()
        scored = score_transactions(txns, accts)
        summary = summarize_results(scored, cbs)
        high = summary[summary["risk_label"] == "high"].iloc[0]
        assert high["chargeback_rate"] == 1.0

    def test_chargeback_rate_low_risk_is_zero(self):
        txns, accts, cbs = _make_pipeline_data()
        scored = score_transactions(txns, accts)
        summary = summarize_results(scored, cbs)
        low = summary[summary["risk_label"] == "low"].iloc[0]
        assert low["chargeback_rate"] == 0.0

    def test_transaction_counts_sum_to_total(self):
        txns, accts, cbs = _make_pipeline_data()
        scored = score_transactions(txns, accts)
        summary = summarize_results(scored, cbs)
        assert summary["transactions"].sum() == len(txns)

    def test_no_chargebacks_returns_zero_rate(self):
        txns, accts, _ = _make_pipeline_data()
        scored = score_transactions(txns, accts)
        empty_cbs = pd.DataFrame(columns=["transaction_id"])
        summary = summarize_results(scored, empty_cbs)
        assert (summary["chargeback_rate"] == 0).all()
