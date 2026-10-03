-- Q: What share of reviews raise each complaint theme (and any complaint)? Wilson 95% CI.
WITH long AS (
    SELECT 'fit' AS theme, fit::INT AS x FROM read_parquet('data/clean/labels.parquet')
    UNION ALL SELECT 'fabric', fabric::INT FROM read_parquet('data/clean/labels.parquet')
    UNION ALL SELECT 'build', build::INT FROM read_parquet('data/clean/labels.parquet')
    UNION ALL SELECT 'photo', photo::INT FROM read_parquet('data/clean/labels.parquet')
    UNION ALL SELECT 'style', style::INT FROM read_parquet('data/clean/labels.parquet')
    UNION ALL SELECT 'price', price::INT FROM read_parquet('data/clean/labels.parquet')
    UNION ALL SELECT 'any complaint', (fit OR fabric OR build OR photo OR style OR price)::INT
        FROM read_parquet('data/clean/labels.parquet')
),
agg AS (
    SELECT theme, COUNT(*) AS n_sample, SUM(x) AS n_with_theme, SUM(x) * 1.0 / COUNT(*) AS rate
    FROM long GROUP BY theme
),
w AS (  -- Wilson interval pieces, z = 1.96
    SELECT *, 1.96 AS z, 1 + 1.96 * 1.96 / n_sample AS d,
           1.96 * SQRT(rate*(1-rate)/n_sample + 1.96*1.96 / (4*n_sample*n_sample)) AS h
    FROM agg
)
SELECT theme, n_sample, n_with_theme,
       ROUND(rate, 4) AS rate,
       ROUND((rate + z*z / (2*n_sample) - h) / d, 4) AS ci_low,
       ROUND((rate + z*z / (2*n_sample) + h) / d, 4) AS ci_high,
       ROUND(h / d, 4) AS margin_of_error
FROM w
ORDER BY theme = 'any complaint' DESC, rate DESC;
