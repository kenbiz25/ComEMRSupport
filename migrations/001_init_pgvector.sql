
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS kb_chunks (
  id BIGSERIAL PRIMARY KEY,
  namespace TEXT NOT NULL,
  content TEXT NOT NULL,
  metadata JSONB,
  embedding vector(384)
);

CREATE INDEX IF NOT EXISTS kb_chunks_embedding_idx
ON kb_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS kb_chunks_namespace_idx
ON kb_chunks (namespace);
