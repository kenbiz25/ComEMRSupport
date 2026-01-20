
# jobs/reindex_kb.py
from core.indexing.pipeline import IndexingPipeline
from config.settings import settings

if __name__ == "__main__":
    indexer = IndexingPipeline()
    total = indexer.reindex_folder("kb")
    print(f"Reindexed chunks: {total} (namespace={settings.KB_NAMESPACE}, dim depends on {settings.EMBED_MODEL})")
