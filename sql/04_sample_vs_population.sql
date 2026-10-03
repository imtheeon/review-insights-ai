-- Q: Is the labelled sample representative of the full population (rating and department mix)?
WITH s AS (SELECT rating::VARCHAR AS rating, department FROM read_parquet('data/clean/labels.parquet')),
     p AS (SELECT rating::VARCHAR AS rating, department FROM read_parquet('data/clean/reviews.parquet')),
     long AS (
         SELECT 'rating' AS dimension, rating AS category, 'sample' AS src FROM s
         UNION ALL SELECT 'rating', rating, 'population' FROM p
         UNION ALL SELECT 'department', department, 'sample' FROM s
         UNION ALL SELECT 'department', department, 'population' FROM p
     ),
     c AS (SELECT dimension, category, src, COUNT(*) AS n FROM long GROUP BY ALL),
     t AS (SELECT *, n * 1.0 / SUM(n) OVER (PARTITION BY dimension, src) AS share FROM c)
SELECT dimension, category,
       MAX(n) FILTER (WHERE src = 'sample') AS n_sample,
       ROUND(MAX(share) FILTER (WHERE src = 'sample'), 4) AS share_sample,
       MAX(n) FILTER (WHERE src = 'population') AS n_population,
       ROUND(MAX(share) FILTER (WHERE src = 'population'), 4) AS share_population,
       ROUND(COALESCE(MAX(share) FILTER (WHERE src = 'sample'), 0) - MAX(share) FILTER (WHERE src = 'population'), 4) AS diff
FROM t
GROUP BY dimension, category
ORDER BY dimension, category;
