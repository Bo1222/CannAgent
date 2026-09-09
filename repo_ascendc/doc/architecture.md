# AscendC Kernel Agent Semantic Knowledge Architecture

## Purpose

This document describes the target architecture of the AscendC Kernel Agent after migration.

The goal is to replace:

Official Documentation
    |
    v
LLM reads markdown
    |
    v
Prompt reasoning

with:

Official Documentation
    |
    v
Knowledge Compiler
    |
    v
Versioned Semantic Knowledge Store
    |
    v
Context-aware Knowledge Router
    |
    v
Planner / Generator / Validator


## Two-plane Architecture


# Offline Knowledge Plane


CANN Official Documentation

        |

        v

Document Parser

        |

        v

Normalized Document

        |

        v

Atomic Fact Extraction

        |

        v

Fact Validation

        |

        v

Knowledge Cards

        |

        v

Snapshot + Index


Responsibilities:

- Parse official CANN documents
- Extract verified API semantics
- Store provenance
- Build versioned knowledge snapshots
- Provide deterministic retrieval


Offline is executed when:

- CANN version changes
- SDK changes
- Documentation updates


It is NOT part of every kernel generation task.


# Online Kernel Agent Plane


Kernel Task

        |

        v

Task Analyzer

        |

        v

Knowledge Router

        |

        v

Knowledge Bundle

        |

        +-------------+
        |             |
        v             v

     Planner      Generator


        |

        v

Semantic Validator


        |

        v

Runtime Execution


        |

        v

Structured Failure


        |

        v

Knowledge Router


Online executes for every kernel generation task.


## Design Principles


1. Official documentation is not runtime prompt content.

2. Atomic facts are the runtime representation of official knowledge.

3. Every fact requires provenance.

4. API semantics must be resolved by context:

- API name
- overload
- version
- SoC
- parameter structure


5. LLM should reason over verified constraints, not discover hardware semantics.


6. Runtime failures must become structured data.


7. Failed experiments must not contaminate stable baselines.


## Migration Strategy


Migration is incremental:

Phase0:
Knowledge Compiler MVP

Phase1:
Knowledge Snapshot

Phase2:
Semantic Router

Phase3:
Structured Failure

Phase4:
Semantic Validator

Phase5:
Experience Loop


Each phase requires:

- limited file scope
- tests
- validation gate
- git checkpoint