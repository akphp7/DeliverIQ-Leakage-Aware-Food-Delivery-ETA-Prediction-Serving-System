-- Leak-free rider history features, one row per order.
--
-- Rule: an order may only use deliveries from EARLIER DAYS. Same-day
-- deliveries may still be in progress when the order is placed, so they are
-- excluded (a simple, defensible cut-off).
--
-- SQL concepts: CTEs, window functions with a frame that ends at
-- "1 PRECEDING", LAG, running sums, LEFT JOIN, COALESCE.
WITH rider_day AS (                      -- 1 row per rider per day
    SELECT
        rider_id,
        CAST(order_date AS DATE)          AS day,
        COUNT(*)                          AS n_orders,
        SUM(time_taken_min)               AS sum_minutes,
        SUM(CASE WHEN time_taken_min > 40 THEN 1 ELSE 0 END) AS n_slow
    FROM orders
    GROUP BY rider_id, CAST(order_date AS DATE)
),
rider_cum AS (                           -- totals over all PREVIOUS days
    SELECT
        rider_id,
        day,
        SUM(n_orders)    OVER w AS prev_orders,
        SUM(sum_minutes) OVER w AS prev_sum_minutes,
        SUM(n_slow)      OVER w AS prev_slow,
        LAG(day)         OVER (PARTITION BY rider_id ORDER BY day) AS prev_active_day
    FROM rider_day
    WINDOW w AS (PARTITION BY rider_id ORDER BY day
                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
)
SELECT
    o.row_id,
    COALESCE(c.prev_orders, 0)                                   AS rider_prev_orders,
    c.prev_sum_minutes / NULLIF(c.prev_orders, 0)                AS rider_prev_avg_minutes,
    100.0 * c.prev_slow / NULLIF(c.prev_orders, 0)               AS rider_prev_slow_pct,
    DATE_DIFF('day', c.prev_active_day, CAST(o.order_date AS DATE)) AS rider_days_since_active
FROM orders o
LEFT JOIN rider_cum c
       ON c.rider_id = o.rider_id
      AND c.day = CAST(o.order_date AS DATE)
ORDER BY o.row_id;
