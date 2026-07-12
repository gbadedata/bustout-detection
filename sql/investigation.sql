-- Bust-out investigation queries, run over the account-month panel with DuckDB.
-- Each is a question an analyst asks behind a flag. None uses the label or the account type;
-- they surface the trajectory and leave the judgement to a person. The view `t` is the panel
-- (see scripts/run_investigation.py), one row per account per statement month.

-- name: rising_utilisation
-- Accounts whose utilisation has climbed sharply over the last three months (a ramp).
WITH seq AS (
  SELECT account_id, month_index, statement_date, utilization, credit_limit, balance, dpd,
         lag(utilization, 3) OVER w AS util_3m_ago,
         row_number() OVER (PARTITION BY account_id ORDER BY month_index DESC) AS rn
  FROM t
  WINDOW w AS (PARTITION BY account_id ORDER BY month_index)
)
SELECT account_id, month_index, statement_date,
       round(util_3m_ago, 3) AS util_3m_ago,
       round(utilization, 3) AS utilization,
       round(utilization - util_3m_ago, 3) AS util_jump_3m,
       round(credit_limit - balance, 0) AS undrawn_line
FROM seq
WHERE rn = 1 AND util_3m_ago IS NOT NULL AND utilization - util_3m_ago > 0.30
ORDER BY util_jump_3m DESC
LIMIT 15;

-- name: maxed_after_limit_increase
-- Accounts that ran to a near-full line within a few months of a limit increase.
WITH seq AS (
  SELECT account_id, month_index, utilization, credit_limit,
         lag(credit_limit) OVER w AS prev_limit
  FROM t
  WINDOW w AS (PARTITION BY account_id ORDER BY month_index)
),
increases AS (
  SELECT account_id, month_index AS increase_month
  FROM seq WHERE prev_limit IS NOT NULL AND credit_limit > prev_limit * 1.001
),
maxed AS (
  SELECT i.account_id, i.increase_month, t.month_index AS maxed_month
  FROM increases i JOIN t ON t.account_id = i.account_id
  WHERE t.month_index > i.increase_month
    AND t.month_index <= i.increase_month + 4
    AND t.utilization >= 0.90
)
SELECT account_id, increase_month, min(maxed_month) AS maxed_month,
       min(maxed_month) - increase_month AS months_after_increase
FROM maxed
GROUP BY account_id, increase_month
ORDER BY months_after_increase ASC, account_id
LIMIT 15;

-- name: full_pay_then_stopped
-- Accounts that paid in full and then abruptly stopped while carrying a high balance.
WITH ratios AS (
  SELECT account_id, month_index, statement_date, balance, utilization,
         payments / (lag(balance) OVER w + 1) AS pay_ratio
  FROM t
  WINDOW w AS (PARTITION BY account_id ORDER BY month_index)
),
seq AS (
  SELECT account_id, month_index, statement_date, utilization, pay_ratio,
         lag(pay_ratio) OVER w AS prev_pay_ratio
  FROM ratios
  WINDOW w AS (PARTITION BY account_id ORDER BY month_index)
)
SELECT account_id, month_index, statement_date,
       round(prev_pay_ratio, 2) AS prev_pay_ratio,
       round(pay_ratio, 2) AS pay_ratio,
       round(utilization, 3) AS utilization
FROM seq
WHERE prev_pay_ratio >= 0.90 AND pay_ratio < 0.10 AND utilization > 0.70
ORDER BY utilization DESC
LIMIT 15;

-- name: cash_draw_spike
-- Statements with a large cash draw on the line, a common bust-out move.
SELECT account_id, month_index, statement_date,
       cash_advance, credit_limit,
       round(cash_advance / credit_limit, 3) AS cash_share,
       round(utilization, 3) AS utilization
FROM t
WHERE cash_advance > 0
ORDER BY cash_share DESC
LIMIT 15;

-- name: fast_limit_growth
-- Accounts whose credit line grew fastest, the cultivation that earns a bigger line to drain.
SELECT account_id,
       arg_min(credit_limit, month_index) AS first_limit,
       max(credit_limit) AS max_limit,
       round(max(credit_limit) / arg_min(credit_limit, month_index), 2) AS growth,
       count(*) AS months
FROM t
GROUP BY account_id
HAVING max(credit_limit) > arg_min(credit_limit, month_index) * 1.001
ORDER BY growth DESC
LIMIT 15;

-- name: undrawn_exposure_now
-- Where a freeze prevents the most loss: for each account still current but climbing fast,
-- the month with the most undrawn line a bust-out would take.
WITH seq AS (
  SELECT account_id, month_index, statement_date, credit_limit, balance, utilization, dpd,
         lag(utilization, 3) OVER w AS util_3m_ago
  FROM t
  WINDOW w AS (PARTITION BY account_id ORDER BY month_index)
)
SELECT account_id, month_index, statement_date,
       round(utilization, 3) AS utilization,
       round(utilization - util_3m_ago, 3) AS util_jump_3m,
       round(credit_limit - balance, 0) AS undrawn_line
FROM seq
WHERE dpd = 0 AND util_3m_ago IS NOT NULL AND (utilization - util_3m_ago) > 0.20
QUALIFY row_number() OVER (PARTITION BY account_id ORDER BY credit_limit - balance DESC) = 1
ORDER BY undrawn_line DESC
LIMIT 15;
