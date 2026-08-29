-- Average delivery time by hour and traffic level (pivot with FILTER).
-- This query is what revealed that traffic is a function of the hour here.
SELECT
    CAST(FLOOR(order_minute_of_day / 60) AS INTEGER)                 AS order_hour,
    COUNT(*)                                                        AS orders,
    ROUND(AVG(time_taken_min), 2)                                   AS avg_minutes,
    COUNT(*) FILTER (WHERE traffic = 'Low')                         AS low_orders,
    COUNT(*) FILTER (WHERE traffic = 'Medium')                      AS medium_orders,
    COUNT(*) FILTER (WHERE traffic = 'High')                        AS high_orders,
    COUNT(*) FILTER (WHERE traffic = 'Jam')                         AS jam_orders
FROM orders
GROUP BY 1
ORDER BY 1;
