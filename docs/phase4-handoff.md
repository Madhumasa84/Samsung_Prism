# Phase 4 handoff (session state, selective updates, and answer versions)

Phase 3 stops after matched evaluation and audit. This document records the
state boundary for Phase 4 follow-up interpretation, dependency-aware
selective updates, answer-version publication, and presentation-only updates.
The lightweight state, validated patch boundary, claim invalidation plan,
targeted retrieval coordinator, and governed publisher are implemented without
requiring a graph database. The final local Phase 4 evaluation is complete over
synthetic lexical/mock fixtures; semantic review, real backends, and official
validation remain explicitly unverified. Final submission production is not
started.

## Intent interface

Each intent is identified by a deterministic `intent_id` derived from the
validated source span/query/constraint content, and is scoped by
`session_id`, `utterance_id`, and `transcript_revision`. The handoff should
retain:

```json
{
  "intent_id": "intent-…",
  "ordinal": 1,
  "query": "search-ready sub-question",
  "source_span": {"start": 0, "end": 20, "text": "exact parent text"},
  "relationship": "independent",
  "constraints": [],
  "status": "active"
}
```

Constraint records remain typed and scoped. Shared constraints apply to all
eligible intents; intent-local constraints apply only to their owning intent.
Corrections create a new revision and supersede affected intent IDs when their
source span, query, or constraints change. Unchanged intent IDs may be reused
only after revision and evidence compatibility checks.

## Claim interface

An answer is a versioned set of atomic claim records rather than an opaque
string:

```json
{
  "claim_id": "claim-…",
  "answer_version": 2,
  "intent_ids": ["intent-…"],
  "claim_text": "one factual proposition",
  "supporting_chunk_ids": ["document#chunk-0000"],
  "supporting_excerpts": ["verbatim excerpt"],
  "supporting_spans": [],
  "semantic_support": "unreviewed",
  "claim_revision": 3,
  "status": "current"
}
```

Every claim must identify the intent(s) it answers and the exact evidence
dependency records used to produce it. Citation-ID validity and excerpt
provenance remain deterministic checks; semantic support requires a separately
governed reviewer. Unsupported or superseded claims should be retained in the
audit history, marked non-current, and excluded from the current answer.

## Evidence-dependency interface

For each intent, persist the dependency graph and retrieval provenance:

```text
intent_id
  -> prerequisite intent IDs / relation
  -> backend rankings and query revision
  -> selected chunk IDs and excerpts
  -> fusion and assembly decisions
  -> claims that consumed this evidence
```

Dependency edges are `depends_on`, `refines`, or `compares_with`. A dependent
retrieval must record the prerequisite evidence revision it consumed. Evidence
is current only when its session, utterance, transcript revision, retrieval
revision, query fingerprint, and corpus/index ID match the current state.

`Phase4SelectiveUpdatePlan` records directly affected intents, propagated
invalidated claims, preserved claims, invalidated/reused evidence, dependency
uncertainty, retrieval reasons, targeted tasks, and discarded late-result IDs.
Reused evidence is copied to the candidate revision with an explicit
`reused_from_evidence_id`; the historical record is never silently reused as
current support.

## Revision and correction interface

Use an append-only revision record for corrections and late work:

```json
{
  "session_id": "…",
  "utterance_id": "…",
  "transcript_revision": 4,
  "retrieval_revision": 4,
  "supersedes": [3],
  "changed_intent_ids": ["intent-…"],
  "preserved_intent_ids": ["intent-…"],
  "invalidated_claim_ids": ["claim-…@v1"],
  "preserved_claim_ids": ["claim-…@v1"],
  "reused_evidence_ids": ["evidence-…"],
  "discarded_result_ids": ["retrieval-…"],
  "reason": "location constraint corrected",
  "created_at": "…"
}
```

Late retrieval or generation results may be observed for telemetry but cannot
be accepted as current after supersession. Stale acceptance is a correctness
failure. The Phase 2 scheduler's stale-result and session-isolation guards
are the regression baseline for this interface.

## Session-state interface

The selective-update coordinator exposes a session-scoped state boundary
similar to:

```text
SessionState
  session_id
  current_utterance_id
  transcript_revision
  retrieval_revision
  active_intents[]
  current_evidence[]
  current_claims[]
  current_answer_version
  answer_status
  selective_update_plans[]
  answer_versions[]
  pending_requests[]
  superseded_requests[]
  generation_status
  trace_ids[]
```

Reads and writes must include the session and utterance identity; a result from
another session is invalid even if its text, intent ID, or chunk IDs match.
State transitions should be monotonic by revision, auditable, and atomic at
the answer publication boundary. A publication must point to the exact
evidence and claim revisions it used.

## Implemented selective-update boundary

`Phase4SelectiveUpdateCoordinator` applies a validated patch, advances the
candidate session revision, and schedules only unresolved intent queries. The
existing `AsyncRetrievalScheduler` supplies bounded workers, retries, pending
work, cancellation as an optimisation, and stale-result observation. Phase 4
adds an independent-query FIFO for multiple unresolved intents; Phase 2's
latest-query coalescing remains the default.

Each task carries the candidate state/transcript/retrieval revisions. Admission
requires scheduler completion, the candidate still being current, matching
corpus/index identity, and active intent identity. A late result can therefore
be observed and audited but cannot become current evidence after a newer patch.
Removing a constraint deliberately schedules broader reconsideration; changing
an entity selection invalidates entity-bound policy, price, and availability
claims even when those fields were not named in the correction.

## Implemented answer-version boundary

`Phase4AnswerPublisher` builds a factual update from preserved claims whose
dependencies remain valid, re-evaluated or newly supported claims, and
explicit unresolved questions for affected unsupported information needs.
Claim records retain stable IDs when their material content is unchanged;
material changes receive a new ID and a recorded `modified` change. The
publication envelope records its parent version, triggering turn and patch,
constraint/evidence/claim deltas, retrieval and generation usage, provider
identity, and `completed`, `partial`, `failed`, or `superseded` status.

The store validates citation IDs and exact source provenance structurally, but
does not confuse those checks with semantic entailment. `expected_revision` is
checked while the session lock is held, and the candidate state, current claim
records, version pointer, revision record, and pending-work bookkeeping are
committed as one atomic transition. A late result is recorded as discarded and
cannot publish incompatible claims or evidence.

Presentation-only updates are separate immutable versions with
`change_kind="presentation"`. They render current usable claims for shorter,
bullet/table, or configured translation requests, preserve qualifications,
entity relationships, and citation tags, and make zero retrieval calls. A
formatting request that also asks a new factual question follows the factual
update path; an empty or unusable answer produces a targeted context request.

## Executable replay

The `phase4-replay` CLI command reads JSONL turns and runs the actual lexical
retriever, evidence assembly, selective scheduler, generation provider, and
answer publisher. The checked-in examples cover initial compound answers,
late constraints, entity corrections, partial support, formatting-only
updates, and a correction that supersedes an older generation. Replay output
includes proposed patches, plans, versions, usage counters, and discarded
late results. Examples are labelled synthetic and use the mock provider unless
real provider configuration is supplied.

The completed matched evaluation is recorded in
[`../reports/phase4_evaluation.md`](../reports/phase4_evaluation.md) and
[`../reports/phase4_evaluation.json`](../reports/phase4_evaluation.json), with
the pending claim sheet at
[`../reports/phase4_evaluation_claim_review.csv`](../reports/phase4_evaluation_claim_review.csv).
The redacted real-backend attempt is
[`../reports/phase4_real_e2e.json`](../reports/phase4_real_e2e.json). The
22-case suite keeps development, diagnostic/regression, and untouched held-out
roles separate; historical Phase 3 reports are preserved.

## Still deferred

The lightweight state, follow-up interpretation, selective retrieval, answer
publication, and presentation boundary described by this interface is
implemented in [docs/phase4-session-state.md](phase4-session-state.md). The
local evaluation is complete, but persistence/retention, clarification-answer
resolution, independently reviewed semantic entailment, real-model execution,
and official benchmark validation remain deferred or `NOT VERIFIED`.
