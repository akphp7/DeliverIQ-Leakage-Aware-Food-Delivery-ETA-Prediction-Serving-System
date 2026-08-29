-- Daily trend with day-over-day change and a 7-day moving average.
-- Topics: LAG, moving-average frame, percentage change, NULLIF.
WITH daily AS (
    SELECT CAST(order_date AS DATE) AS day,
           COUNT(*)                 AS orders,
           AVG(time_taken_min)      AS avg_minutes
    FROM orders
    GROUP BY 1
)
SELECT
    day,
    orders,
    ROUND(avg_minutes, 2)                                                        AS avg_minutes,
    ROUND(avg_minutes - LAG(avg_minutes) OVER (ORDER BY day), 2)                 AS change_vs_prev_day,
    ROUND(100.0 * (orders - LAG(orders) OVER (ORDER BY day))
          / NULLIF(LAG(orders) OVER (ORDER BY day), 0), 1)                       AS orders_change_pct,
    ROUND(AVG(avg_minutes) OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2)
                                                                                 AS avg_minutes_ma7
FROM daily
ORDER BY day;
