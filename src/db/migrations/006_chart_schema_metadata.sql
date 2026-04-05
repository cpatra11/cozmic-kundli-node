CREATE OR REPLACE FUNCTION jsonb_dasha_depth(node jsonb)
RETURNS INTEGER
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
  child jsonb;
  child_depth integer;
  deepest_child integer := 0;
BEGIN
  IF node IS NULL OR jsonb_typeof(node) <> 'object' THEN
    RETURN 0;
  END IF;

  IF NOT (node ? 'periods') OR jsonb_typeof(node->'periods') <> 'object' THEN
    RETURN 0;
  END IF;

  FOR child IN SELECT value FROM jsonb_each(node->'periods') LOOP
    child_depth := jsonb_dasha_depth(child);
    IF child_depth > deepest_child THEN
      deepest_child := child_depth;
    END IF;
  END LOOP;

  RETURN deepest_child + 1;
END;
$$;

ALTER TABLE rag_api_sources
  ADD COLUMN IF NOT EXISTS chart_schema_version TEXT NOT NULL DEFAULT 'mahadasha-first',
  ADD COLUMN IF NOT EXISTS dasha_depth INTEGER NOT NULL DEFAULT 1,
  ADD COLUMN IF NOT EXISTS dasha_period_key TEXT;

ALTER TABLE charts
  ADD COLUMN IF NOT EXISTS chart_schema_version TEXT NOT NULL DEFAULT 'mahadasha-first',
  ADD COLUMN IF NOT EXISTS dasha_depth INTEGER NOT NULL DEFAULT 1,
  ADD COLUMN IF NOT EXISTS dasha_period_key TEXT;

WITH source_schema AS (
  SELECT
    id,
    COALESCE(
      raw_payload->'chart'->'dasha',
      raw_payload->'dasha',
      chart_snapshot->'chart'->'dasha',
      chart_snapshot->'dasha'
    ) AS dasha_node,
    jsonb_dasha_depth(
      COALESCE(
        raw_payload->'chart'->'dasha',
        raw_payload->'dasha',
        chart_snapshot->'chart'->'dasha',
        chart_snapshot->'dasha'
      )
    ) AS dasha_depth_value
  FROM rag_api_sources
)
UPDATE rag_api_sources AS src
SET
  dasha_depth = COALESCE(NULLIF(source_schema.dasha_depth_value, 0), 1),
  chart_schema_version = CASE
    WHEN COALESCE(NULLIF(source_schema.dasha_depth_value, 0), 1) > 1 THEN 'legacy-deep-dasha'
    ELSE 'mahadasha-first'
  END,
  dasha_period_key = NULLIF(
    COALESCE(
      source_schema.dasha_node->>'period_key',
      source_schema.dasha_node->>'periodKey',
      source_schema.dasha_node->>'key'
    ),
    ''
  )
FROM source_schema
WHERE src.id = source_schema.id;

WITH chart_schema AS (
  SELECT
    id,
    COALESCE(chart_data->'chart'->'dasha', chart_data->'dasha') AS dasha_node,
    jsonb_dasha_depth(COALESCE(chart_data->'chart'->'dasha', chart_data->'dasha')) AS dasha_depth_value
  FROM charts
)
UPDATE charts AS chart_row
SET
  dasha_depth = COALESCE(NULLIF(chart_schema.dasha_depth_value, 0), 1),
  chart_schema_version = CASE
    WHEN COALESCE(NULLIF(chart_schema.dasha_depth_value, 0), 1) > 1 THEN 'legacy-deep-dasha'
    ELSE 'mahadasha-first'
  END,
  dasha_period_key = NULLIF(
    COALESCE(
      chart_schema.dasha_node->>'period_key',
      chart_schema.dasha_node->>'periodKey',
      chart_schema.dasha_node->>'key'
    ),
    ''
  )
FROM chart_schema
WHERE chart_row.id = chart_schema.id;
