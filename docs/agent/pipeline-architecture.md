# MagNova pipeline architecture

```text
User text
  → MiniMindInterpreter
  → MaterialRequest
  → MagNovaTaskManager
  → MockPipelineMagNovaBackend
      1. MaterialGenerator
         → GeneratedMaterial[]
      2. PropertyPredictor
         → PredictedMaterial[]
      3. CandidateScreener
         → CandidateMaterial[]
  → SQLiteTaskStore
      → status
      → stage
      → progress
      → workflow_trace
      → candidates
  → Streamlit fragment
```

Replacement points:

- Mock generator → MagNova generative model;
- Mock predictor → GNN/equivariant model/DFT service;
- rule screener → unit-aware multi-objective screening service;
- mock pipeline backend → real HTTP workflow API.
