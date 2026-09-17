# Phase 4 session state and follow-up interpretation

Phase 4 provides a bounded in-memory state boundary, a validated follow-up
patch interpreter, dependency-aware invalidation, selective retrieval for
unresolved information needs, and an atomic answer-version publication
boundary. Presentation-only revisions use the current valid answer without
starting factual retrieval.

## Implemented contracts

The contracts are in `flowcontext.contracts` under
`flowcontext.phase4.v1`:

- `Phase4SessionState` stores the active topic/task, active intents, complete
  constraint history plus current constraint IDs, evidence provenance and
  corpus/index identity, claim records and dependencies, immutable answer
  publication envelopes, revision history, pending clarification/update
  records, request bookkeeping, generation status, and trace IDs.
- `Phase4ConstraintRecord` wraps the existing typed
  `DecompositionConstraint` and records its origin turn. `corpus_fact` is
  literally `False`; a user constraint or inherited reference is not a
  corpus-supported fact.
- `Phase4EvidenceRecord` wraps the existing `EvidencePassage` and binds it to
  session, utterance, transcript/retrieval revisions, corpus, and index. A
  reused passage receives a new candidate-bound record with an explicit
  `reused_from_evidence_id`; the old record remains historical.
- `Phase4ClaimRecord` wraps the existing `FactualClaim`, records supporting
  evidence IDs, and accepts explicit dependencies on evidence, constraints,
  entities, and other claims. `semantic_support_status` defaults to
  `unreviewed`; a valid chunk ID, score, or excerpt does not set it to
  `supported`.
- `Phase4AnswerVersion` is an immutable publication envelope with parent and
  triggering-turn metadata, evidence and claim deltas, retrieval/generation/
  presentation usage, unresolved questions, and a completion status. The
  store only publishes a copied answer and never edits an earlier publication.
- `Phase4ProposedPatch` contains the exact `base_revision`, the next proposed
  revision, operation type, typed operations, reference resolutions, provider
  identity, and decision trace. A clarification patch cannot contain semantic
  mutation fields.
- `Phase4SelectiveUpdatePlan` records direct and propagated impact, preserved
  and invalidated claim/evidence IDs, dependency uncertainty, retrieval
  reasons, targeted tasks, and discarded results.
- `Phase4SelectiveRetrievalTask` binds each query to a candidate state,
  transcript, and retrieval revision. Its scheduler request ID, attempts,
  evidence IDs, terminal status, and errors are retained in the plan.
- `Phase4ReplayTurn` and `Phase4ReplayResult` make the actual retrieval,
  synthesis, publication, and supersession path replayable from JSONL. Replay
  output labels synthetic corpus data and mock-provider execution explicitly.

The main API is:

```python
from flowcontext.phase4 import (
    Phase4AnswerPublisher,
    Phase4FollowUpInterpreter,
    Phase4SessionStore,
)

store = Phase4SessionStore(max_sessions=32, max_records_per_session=256)
state = store.create_session("session-1", initial_plan=plan)
patch = store.propose_follow_up(
    "session-1",
    "Actually, make it 40 attendees in Mumbai.",
    utterance_id="utterance-2",
    turn_index=2,
)
state = store.apply_patch(patch)
```

For retrieval, use the coordinator after the state patch:

```python
from flowcontext.phase4 import Phase4SelectiveUpdateCoordinator

coordinator = Phase4SelectiveUpdateCoordinator(store, retriever)
plan = coordinator.apply_patch(patch)
state = await coordinator.execute_plan(plan)
publisher = Phase4AnswerPublisher(store, corpus, generation_provider=provider)
publication = await publisher.generate_and_publish(
    session_id="session-1",
    expected_revision=state.state_revision,
)
```

`propose_follow_up` is non-mutating. `apply_patch` rejects a stale base
revision, so callers cannot silently apply an interpretation to a later
session state. `clear(session_id)` and `clear_all()` are explicit; the store
has no persistence, cross-session profile, or process-wide conversational
cache. The store evicts the least-recently-used session at `max_sessions` and
bounds per-session historical collections.

`state_revision` advances for every accepted transition. The transcript and
retrieval revisions advance only for semantic changes; clarification and
formatting turns change the state boundary but do not pretend that the
underlying retrieval query changed.

`Phase4SessionStore.apply_patch` computes the selective plan atomically. The
coordinator then uses the existing bounded `AsyncRetrievalScheduler`; it
searches the full configured corpus/index for each changed information need.
It does not restart unchanged intent queries. Multiple unresolved intents use
the scheduler's bounded independent-work queue, while a newer candidate
supersedes all older work for that session.

## Answer-version publication

`Phase4AnswerPublisher` is the governed boundary between candidate evidence and
user-visible state. A factual publication is assembled from three explicit
parts: preserved claims whose dependencies are still valid, claims
re-evaluated or newly supported from the candidate evidence, and unresolved
questions/uncertainty for anything that could not be supported. Retrieval rank,
valid citation IDs, and exact source excerpts are retained as structural and
provenance checks; they are not semantic support by themselves.

Publication validates the candidate corpus/index and citation provenance, then
checks `expected_revision` while holding the session lock. The new immutable
version, current claim records, revision record, pointer, and pending-work
updates are committed together. If the revision changed, the result is
recorded as `superseded` and cannot publish a mixture of old and new claims.
Unchanged claims retain their stable IDs. Material changes receive a new claim
ID and a `modified` change record; added, removed, and preserved claims are
also recorded against the parent version.

If generation or targeted retrieval fails, unaffected claims remain available,
affected claims are represented as unresolved, and the resulting version is
marked `partial` or `failed`. The prior version remains inspectable only as
historical; it is not silently presented as the answer to the new request.
Version history includes the parent version, triggering turn and patch,
constraint/evidence/claim deltas, retrieval and generation usage, and terminal
status.

Presentation-only requests (`shorter`, bullet/table formatting, or configured
translation) publish a `change_kind="presentation"` version from the current
usable claims. They preserve claim IDs, qualifications, entity relationships,
and citation tags, and record zero retrieval/generation calls. A formatting
request combined with a new factual question is classified as a factual
follow-up and cannot use this path. If there is no usable answer, the system
asks for context instead of inventing a formatted answer.

## Executable replay

The replay harness exercises the real lexical retriever, evidence assembly,
existing selective scheduler, generation provider, and publication store:

```bash
PYTHONPATH=src python3 -m flowcontext.cli phase4-replay \
  --turns examples/replay/phase4-initial-compound.jsonl \
  --index artifacts/fixture-lexical-index.json --backend lexical \
  --output /tmp/phase4-initial.json

PYTHONPATH=src python3 -m flowcontext.cli phase4-replay \
  --turns examples/replay/phase4-formatting.jsonl \
  --index artifacts/fixture-lexical-index.json --backend lexical

PYTHONPATH=src python3 -m flowcontext.cli phase4-replay \
  --turns examples/replay/phase4-race.jsonl \
  --index artifacts/fixture-lexical-index.json --backend lexical --race
```

The examples use synthetic documents and the mock generation provider unless
real provider settings are explicitly configured. The JSON result exposes
proposed patches, selective plans, answer versions, discarded stale results,
and retrieval/generation counters; the formatting step demonstrates zero
retrieval calls.

## Traceable patch examples

For an initial question such as “Which Hall Alpha in Pune can host an event?”:

```text
Also require a projector for at least 40 attendees on 2026-10-02.
  classification: add_constraint
  base_revision: 0 -> proposed_revision: 1
  operations: quantity(at least 40 attendees), date(2026-10-02), other(projector…)
  origin: user_follow_up, turn 2, corpus_fact=False
```

```text
Actually, make it 50 attendees in Mumbai.
  classification: replace_constraint
  operations: old quantity/location records -> new quantity/location records
  intent: old intent is superseded; the revised intent receives a new ID
```

```text
What is its cancellation policy?
  classification: add_question
  reference: its -> Hall Alpha
  resolution: exactly one active entity candidate
  inherited entity: origin=inherited_context, corpus_fact=False
```

With Hall Alpha and Hall Beta both active, “What is that venue’s cancellation
policy?” produces `clarification_required` and a question naming both
candidates. Applying that patch records the pending clarification and leaves
active intents and constraints unchanged. A topic switch creates a new active
intent set and clears old current constraints, so old-topic constraints cannot
leak into the new task. A formatting request creates a non-retrieval pending
update; publication then creates a presentation child version while preserving
the parent claim IDs and provenance.

For a local correction, the selective plan is the traceable boundary:

```text
change attendee count on intent-A
  direct intent: intent-A
  preserved: claim-B, evidence-B
  invalidated: claim-A, evidence-A
  targeted query: revised intent-A query
```

If the revised constraints are structurally covered by current evidence, the
plan has no retrieval task and rebinds that evidence for reassessment. If a
constraint is removed, the valid answer set is broader, so a task is created
even when the old passage looks lexically similar. Changing an entity selection
also invalidates claims that depend on its policies, prices, availability, or
other entity-bound fields.

## Phase 3 diagnostic boundary

The three Phase 4 diagnostic cases corresponding to the known Phase 3 failures
remain marked as diagnostic/regression cases;
their labels and historical scores are preserved. The failure trace was:

- The held-out single catering question was split during decomposition. That
  created two retrieval branches, which let unrelated evidence reach fusion
  and synthesis; the extra intent was correctly counted by the evaluation
  metric. This was application/decomposition behavior, not a label change.
- The partial and unsupported requests exposed a weak lexical-evidence path:
  retrieval returned a candidate, then mock evidence assembly/generation
  treated the candidate, citation ID, or excerpt as enough to state a claim.
  The fix is conservative support gating and explicit unsupported-intent
  uncertainty in the Phase 3 application path; it is not a change to labels or
  denominators. Semantic entailment and real-provider quality remain
  unevaluated.

The preserved historical outcomes remain in
`reports/phase3_evaluation_realtime.{md,json}`. The current diagnostic output
is recorded in `reports/phase4_diagnostic_regression.{md,json}`; the final
Phase 4 matched report is in `reports/phase4_evaluation.{md,json}` and its
untouched cases are in `data/evaluation/phase4_untouched.jsonl`.
`data/evaluation/phase4_generalization.jsonl` is retained as an earlier
reserved asset and is not part of the final 22-case suite.

## Provider execution labels

The default interpreter is labelled `rule_based` and uses the existing
validated offline decomposition helper. If a
`StructuredMultiIntentDecomposer` is supplied, its existing provider
configuration is used for intent/constraint materialization. The patch is
labelled `mock_provider` for the existing mock provider and `real_provider`
for an OpenAI-compatible provider. A real-provider failure is not replaced by
mock behavior. Mock-provider tests demonstrate contract plumbing only, not
real-model interpretation quality.

## Limits and deliberate non-goals

- Publication and presentation are implemented through the governed publisher;
  final semantic entailment still depends on the optional verifier/provider.
- Translation uses a configured presentation provider and preserves the
  current claim payload, but translation quality is not established by the
  synthetic tests.
- Reference resolution is intentionally conservative and currently uses
  active typed entity context and entity claim dependencies. It asks instead
  of guessing when there is no unique candidate.
- Constraint matching is structural. It preserves negation, units,
  quantities, dates, and entity strings but does not establish corpus truth.
- `semantic_support_status` remains `unreviewed` unless a separately governed
  verifier explicitly supplies a verdict. Retrieval rank, citation validity,
  and excerpt validity are provenance checks, not entailment.
- State and scheduler runtimes are process-local and bounded. Persistence,
  retention policy, clarification-answer resolution, semantic entailment
  review, and cross-session identity are intentionally deferred. Cancellation
  remains an optimisation; revision checks are the correctness boundary.

The Phase 3 held-out failures remain historical diagnostic/regression cases in
the existing reports; their scores are not rewritten by Phase 4 state tests.
The final local evaluation remains synthetic/lexical/mock and leaves semantic
review, real-model execution, and official validation `NOT VERIFIED`.
