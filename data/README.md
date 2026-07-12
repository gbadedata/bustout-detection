# Data

There is no public labelled bust-out dataset; lenders do not release one. This project runs
on a schema-faithful account-month panel from `bustout.panel_data.mock_panel`, which
simulates four account archetypes (good, revolver, genuine distress, and bust-out) so the
model can be judged on the one thing that matters: telling a bust-out ramp apart from
genuine financial distress.

To run on your own book, drop a CSV at `data/panel.csv` with one row per account per
statement month and these columns:

    account_id, month_index, statement_date, credit_limit, purchases, cash_advance,
    payments, balance, min_payment_due, dpd, utilization

`month_index` counts statements from account open. `dpd` is days past due. The forward
label is derived by the pipeline, not required in the feed. Raw CSVs are git-ignored.
