# Embedded CANNBot knowledge base

- Knowledge base ID: `cannbot-a08c4970-cannagent-routing-v1`
- Upstream repository: `cannbot-skills`
- Upstream commit: `a08c49706e35a400d7c77e0875bc7c72a3a79012`
- License: CANN Open Software License Agreement 2.0; see
  `CANN_OPEN_SOFTWARE_LICENSE_2.0.txt` in this directory.

The `docs/` tree is a read-only, curated documentation knowledge base used by CannAgent's
in-process Skill Adapter. It intentionally excludes CANNBot workflows, scripts, evals,
CMake files, execution commands, and complete project templates. Static knowledge base text
is `Level 1 — Documented`; only installed headers, CannAgent source, or local
compile/correctness evidence may raise a fact to Installed or Verified authority.

The knowledge base is not selected by filesystem links found inside an index document.
Every deliverable leaf section is explicitly named in `skill_mapping.yaml` and checked
against `knowledge_base_manifest.json` before the first round.
