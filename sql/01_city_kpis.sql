-- City-level delivery KPIs (aggregation, conditional counts, percentiles).
-- SQL concepts: GROUP BY, CASE WHEN inside aggregates, QUANTILE, HAVING.
SELECT
    city_code,
    COUNT(*)                                                    AS orders,
    ROUND(AVG(time_taken_min), 2)                               AS avg_minutes,
    QUANTILE_CONT(time_taken_min, 0.5)                          AS p50_minutes,
    QUANTILE_CONT(time_taken_min, 0.9)                          AS p90_minutes,
    ROUND(AVG(distance_km), 2)                                  AS avg_distance_km,
    ROUND(100.0 * SUM(CASE WHEN traffic = 'Jam' THEN 1 ELSE 0 END) / COUNT(*), 1)
                                                                AS jam_share_pct,
    ROUND(100.0 * SUM(CASE WHEN time_taken_min > 40 THEN 1 ELSE 0 END) / COUNT(*), 1)
                                                                AS over_40min_pct,
    COUNT(DISTINCT rider_id)                                    AS riders,
    COUNT(DISTINCT restaurant_id)                               AS restaurants
FROM orders
GROUP BY city_code
HAVING COUNT(*) >= 500
ORDER BY avg_minutes DESC, city_code;
