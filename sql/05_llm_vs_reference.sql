-- Q: How well do the LLM labels agree with the 100 reference labels (written by Claude, the AI assistant that built this repo, reading each review)?
WITH j AS (
    SELECT l.tone AS l_tone, h.tone AS h_tone,
           l.fit::INT AS l_fit, h.fit AS h_fit, l.fabric::INT AS l_fabric, h.fabric AS h_fabric,
           l.build::INT AS l_build, h.build AS h_build, l.photo::INT AS l_photo, h.photo AS h_photo,
           l.style::INT AS l_style, h.style AS h_style, l.price::INT AS l_price, h.price AS h_price
    FROM read_parquet('data/clean/labels.parquet') l
    JOIN read_csv('data/reference_labels.csv') h USING (review_id)
),
long AS (
    SELECT 'fit' AS theme, h_fit AS h, l_fit AS l FROM j
    UNION ALL SELECT 'fabric', h_fabric, l_fabric FROM j
    UNION ALL SELECT 'build', h_build, l_build FROM j
    UNION ALL SELECT 'photo', h_photo, l_photo FROM j
    UNION ALL SELECT 'style', h_style, l_style FROM j
    UNION ALL SELECT 'price', h_price, l_price FROM j
),
b AS (  -- binary themes: counts and marginal rates
    SELECT theme, COUNT(*) AS n, AVG((h = l)::INT) AS po,
           SUM(h) AS h_pos, SUM(l) AS l_pos, SUM(h * l) AS both_pos,
           AVG(h) AS ph, AVG(l) AS pl
    FROM long GROUP BY theme
),
tone_pe AS (  -- 3-class tone: pe = sum over classes of p_reference * p_llm
    SELECT SUM(a.p * COALESCE(c.p, 0)) AS pe
    FROM (SELECT h_tone AS k, COUNT(*) * 1.0 / (SELECT COUNT(*) FROM j) AS p FROM j GROUP BY h_tone) a
    LEFT JOIN (SELECT l_tone AS k, COUNT(*) * 1.0 / (SELECT COUNT(*) FROM j) AS p FROM j GROUP BY l_tone) c
      ON a.k = c.k
),
res AS (
    SELECT theme, n, po AS agreement, h_pos, l_pos, both_pos,
           both_pos * 1.0 / NULLIF(l_pos, 0) AS prec,
           both_pos * 1.0 / NULLIF(h_pos, 0) AS rec,
           (po - (ph*pl + (1-ph)*(1-pl))) / NULLIF(1 - (ph*pl + (1-ph)*(1-pl)), 0) AS kappa
    FROM b
    UNION ALL  -- tone is multi-class: no positives/precision/recall
    SELECT 'tone', COUNT(*), AVG((l_tone = h_tone)::INT), NULL, NULL, NULL, NULL, NULL,
           (AVG((l_tone = h_tone)::INT) - ANY_VALUE(pe)) / NULLIF(1 - ANY_VALUE(pe), 0)
    FROM j, tone_pe
    UNION ALL
    SELECT 'all six themes match', COUNT(*),
           AVG((l_fit = h_fit AND l_fabric = h_fabric AND l_build = h_build
                AND l_photo = h_photo AND l_style = h_style AND l_price = h_price)::INT),
           NULL, NULL, NULL, NULL, NULL, NULL
    FROM j
)
SELECT theme, n, ROUND(agreement, 3) AS agreement, h_pos AS reference_positives, l_pos AS llm_positives,
       both_pos AS both_positive, ROUND(prec, 3) AS precision, ROUND(rec, 3) AS recall, ROUND(kappa, 3) AS kappa
FROM res
ORDER BY theme = 'all six themes match', theme = 'tone', theme;
