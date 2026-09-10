from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KNOWLEDGE_SOURCE = (
    REPOSITORY_ROOT
    / "skills/ascendc/ascendc-translator/references/AscendC_knowledge"
)
DEFAULT_KNOWLEDGE_STORE = REPOSITORY_ROOT / "knowledge_store"
