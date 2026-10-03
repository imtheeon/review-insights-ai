-- Q: Does each complaint theme's rate differ by department (departments with >= 30 sample reviews)?
WITH long AS (
    SELECT department, 'fit' AS theme, fit::INT AS x FROM read_parquet('data/clean/labels.parquet')
    UNION ALL SELECT department, 'fabric', fabric::INT FROM read_parquet('data/clean/labels.parquet')
    UNION ALL SELECT department, 'build', build::INT FROM read_parquet('data/clean/labels.parquet')
    UNION ALL SELECT department, 'photo', photo::INT FROM read_parquet('data/clean/labels.parquet')
    UNION ALL SELECT department, 'style', style::INT FROM read_parquet('data/clean/labels.parquet')
    UNION ALL SELECT department, 'price', price::INT FROM read_parquet('data/clean/labels.parquet')
)
SELECT department, theme, COUNT(*) AS n, ROUND(AVG(x), 4) AS rate
FROM long
GROUP BY department, theme
HAVING COUNT(*) >= 30  -- n counts reviews per department; small departments are too noisy
ORDER BY department, rate DESC;
