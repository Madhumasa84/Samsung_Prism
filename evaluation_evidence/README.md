# FlowContext Evaluation Evidence

Local-only evidence archive for Samsung PRISM Theme 4 evaluation.

## Structure

- `00_run_manifest/` — environment and command records
- `01_baseline/` — Phase 1 development and held-out evaluations
- `02_streaming/` — Phase 2 streaming development and held-out evaluations
- `03_multi_intent/` — Phase 3 development and held-out evaluations
- `04_session_refinement/` — Phase 4 manual replay evidence
- `05_smoke_tests/` — smoke and dense-smoke verification
- `06_indexes/` — copies of evaluation indexes used
- `07_reports/` — consolidated Theme 4 gate summaries

## Important

This folder is local evidence only and is not intended to be committed or pushed to GitHub.

Evaluation results using synthetic fixtures, provisional labels, mock providers, or local measurements must not be presented as official benchmark results.

## Current measured gates

- G2 Early Retrieval: 100% of eligible cases in development and held-out streaming evaluation.
- G3 Multi-Intent: 75% on the Phase 3 held-out evaluation.
- G4 Factual Grounding: not yet verified.
- G5 Session Refinement: formal evaluation not yet completed.
- G6 Telemetry: 100% trace completeness in the measured streaming sets.
