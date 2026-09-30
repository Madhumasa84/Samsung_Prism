# Phase 3 Multi-Intent Evidence

The Phase 3 CLI writes both development and held-out runs to the same default report path:
`reports/phase3_evaluation_realtime.json` and `.md`.

The currently archived report in `development/` is the latest held-out run and must NOT be interpreted as the development report.

Development result was observed in the terminal during this session:
- 13 development cases
- Multi-intent identification: 100%
- Target: >=70%
- Status: PASS
- Synthetic fixture / provisional labels

Held-out result:
- 7 held-out cases
- Multi-intent identification: 75% (3/4 compound cases)
- Target: >=70%
- Status: PASS
- Synthetic fixture / provisional labels

The raw development JSON/Markdown report was overwritten by the later held-out run and was not separately captured.
