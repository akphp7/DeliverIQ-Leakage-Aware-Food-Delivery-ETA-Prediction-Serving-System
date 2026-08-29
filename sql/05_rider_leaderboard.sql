-- Rank riders within each city.
-- Topics: ranking functions, QUALIFY, NTILE, ties.
WITH rider_stats AS (
    SELECT
        city_code,
        rider_id,
        COUNT(*)                        AS orders,
        AVG(time_taken_min)             AS avg_minutes,
        AVG(rider_rating)               AS avg_rating
    FROM orders
    GROUP BY city_code, rider_id
    HAVING COUNT(*) >= 20               -- ignore riders with too few orders
)
SELECT
    city_code,
    rider_id,
    orders,
    ROUND(avg_minutes, 2)                                               AS avg_minutes,
    ROUND(avg_rating, 2)                                                AS avg_rating,
    RANK()       OVER (PARTITION BY city_code ORDER BY avg_minutes)     AS speed_rank,
    DENSE_RANK() OVER (PARTITION BY city_code ORDER BY ROUND(avg_rating, 1) DESC) AS rating_rank,
    NTILE(4)     OVER (PARTITION BY city_code ORDER BY avg_minutes)     AS speed_quartile
FROM rider_stats
QUALIFY RANK() OVER (PARTITION BY city_code ORDER BY avg_minutes) <= 3
ORDER BY city_code, speed_rank;
