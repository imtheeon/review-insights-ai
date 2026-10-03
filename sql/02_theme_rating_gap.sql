-- Q: How much lower are star ratings for reviews with a theme vs without (raw gap, overlaps double-counted)?
WITH long AS (
    SELECT 'fit' AS theme, fit AS has, rating, recommended FROM read_parquet('data/clean/labels.parquet')
    UNION ALL SELECT 'fabric', fabric, rating, recommended FROM read_parquet('data/clean/labels.parquet')
    UNION ALL SELECT 'build', build, rating, recommended FROM read_parquet('data/clean/labels.parquet')
    UNION ALL SELECT 'photo', photo, rating, recommended FROM read_parquet('data/clean/labels.parquet')
    UNION ALL SELECT 'style', style, rating, recommended FROM read_parquet('data/clean/labels.parquet')
    UNION ALL SELECT 'price', price, rating, recommended FROM read_parquet('data/clean/labels.parquet')
)
SELECT theme,
       COUNT(*) FILTER (WHERE has) AS n_with,
       COUNT(*) FILTER (WHERE NOT has) AS n_without,
       ROUND(AVG(rating) FILTER (WHERE has), 3) AS avg_rating_with,
       ROUND(AVG(rating) FILTER (WHERE NOT has), 3) AS avg_rating_without,
       ROUND(AVG(rating) FILTER (WHERE has) - AVG(rating) FILTER (WHERE NOT has), 3) AS gap_stars,
       ROUND(AVG((rating <= 2)::INT) FILTER (WHERE has), 3) AS share_1_2_star,
       ROUND(AVG(recommended) FILTER (WHERE has), 3) AS share_recommended
FROM long
GROUP BY theme
ORDER BY gap_stars;
