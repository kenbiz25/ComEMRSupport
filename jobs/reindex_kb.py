
# jobs/reindex_kb.py
from core.indexing.pipeline import IndexingPipeline
from config.settings import settings

if __name__ == "__main__":
    # Reindex the configured KB directory (or explicit 'kb' folder)
    indexer = IndexingPipeline(kb_dir="kb")
    total = indexer.reindex_all()
    print(f"Reindexed chunks: {total} (namespace={settings.KB_NAMESPACE}, dim depends on {settings.EMBED_MODEL})")
