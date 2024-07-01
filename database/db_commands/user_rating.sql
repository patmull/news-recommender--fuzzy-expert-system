SELECT p.slug AS slug, r.value AS rating
FROM posts p
JOIN ratings r ON r.post_id = p.id
WHERE r.user_id = 3185;