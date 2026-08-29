-- Leak-free restaurant features: rolling 7-day window of PREVIOUS days.
-- Uses a RANGE frame on dates, so gaps in activity are handled correctly.
WITH rest_day AS (
    SELECT
        restaurant_id,
        CAST(order_date AS DATE)  AS day,
        COUNT(*)                  AS n_orders,
        SUM(time_taken_min)       AS sum_minutes
    FROM orders
    GROUP BY 1, 2
),
rest_roll AS (
    SELECT
        restaurant_id,
        day,
        SUM(n_orders)    OVER w AS orders_7d,
        SUM(sum_minutes) OVER w AS minutes_7d
    FROM rest_day
    WINDOW w AS (PARTITION BY restaurant_id ORDER BY day
                 RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING)
)
SELECT
    o.row_id,
    COALESCE(r.orders_7d, 0)                         AS rest_orders_prev7d,
    r.minutes_7d / NULLIF(r.orders_7d, 0)            AS rest_avg_minutes_prev7d
FROM orders o
LEFT JOIN rest_roll r
       ON r.restaurant_id = o.restaurant_id
      AND r.day = CAST(o.order_date AS DATE)
ORDER BY o.row_id;
