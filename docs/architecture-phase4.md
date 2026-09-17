# Phase 4 architecture and dependency invalidation

Phase 4 adds bounded, process-local session state around the existing
decomposition, retrieval, scheduler, synthesis, and provenance contracts. It
does not add a graph database, a cross-session profile, or a persistent
conversation cache.

## Runtime flow

```text
follow-up text
    -> Phase4FollowUpInterpreter
    -> Phase4ProposedPatch(base_revision = current state revision)
    -> Phase4SessionStore.apply_patch (candidate revision)
    -> dependency analysis and invalidation plan
    -> existing AsyncRetrievalScheduler for unresolved intent queries
    -> revision-checked evidence admission
    -> Phase4AnswerPublisher atomic publication
```

The presentation path branches after patch validation:

```text
reformat-only patch -> current usable answer -> presentation version
mixed format + fact -> clarification; no formatting-only shortcut
```

## State and identity

`Phase4SessionState` is the complete session boundary. It contains the active
topic/task, active and historical intents, typed user constraints, corpus and
index identity, evidence records, claim records, immutable answer versions,
the current-version pointer, pending clarification/update/request records,
selective plans, and trace/revision history.

User constraints are represented by `Phase4ConstraintRecord` with
`corpus_fact=false`. A value such as `40 attendees`, a date, a negation, or a
venue name is therefore retained as a requested condition, not promoted to a
fact merely because a retrieval result contains the same text.

Evidence is bound to `session_id`, `corpus_id`, `index_id`, utterance and
transcript/retrieval revisions. Claims point to evidence records and may also
depend on constraints, entities, or other claims. The dependency graph is a
small explicit adjacency structure in `Phase4ClaimDependency`; no external
graph store is needed at this scale.

## Candidate patch and invalidation

Every semantic follow-up is proposed against an exact state revision. The
store rejects a stale base revision, unknown target, inactive constraint, or
non-fresh replacement intent. Applying a patch creates a candidate revision;
it does not mutate the historical intent/constraint records in place.

The invalidation pass proceeds in four layers:

1. Direct intent changes come from the patch target/replacement map. A shared
   constraint change expands the direct set to every active intent.
2. Evidence whose intent or dependency binding no longer matches is marked
   superseded. Structurally compatible passages are rebound as new evidence
   records with `reused_from_evidence_id`; the old record remains historical.
3. Claims attached to changed intents, changed constraints/entities, invalid
   evidence, or missing dependency identities are invalidated. Missing
   dependency coverage is conservative: the answer portion is unresolved and
   the reason is recorded.
4. Claim-to-claim edges are walked to a fixed point. A summary/comparison
   claim cannot survive when a claim it depends on is invalidated.

Entity replacements propagate to active branches that explicitly select the
same entity. This matters for downstream price, policy, availability, and
capacity claims: a valid old citation is not current support for a replacement
entity. Constraint removal is deliberately broad because removing a filter can
add valid answers that were not in the old evidence set.

The resulting `Phase4SelectiveUpdatePlan` records directly affected intents,
invalidated intents/claims/evidence, preserved claims, reused evidence,
retrieval reasons/tasks, dependency uncertainty, and discarded result IDs.

## Selective retrieval and revision protection

Only unresolved information needs become targeted tasks. A task stores the
candidate state, transcript, and retrieval revisions and searches the full
corpus index for that changed need. “Selective” describes the changed query
scope, not a restriction to previously seen chunks.

The existing bounded scheduler provides concurrency, retries, timeout, and
cancellation. Cancellation is only an optimisation. The scheduler controller
advances its candidate revision before superseding old work, and evidence
admission checks the session revision, task revision, active intent, corpus,
and index. A late result is recorded as discarded and cannot publish. The
publisher applies the same current-revision check atomically while holding the
session lock.

## Answer versions and presentation

The publisher builds a factual version from preserved valid claims,
re-evaluated/new claims, and explicit unresolved information needs. An
unchanged claim keeps its stable `claim_id`; a material change receives a new
ID and a `modified` change record. Each version records its parent, triggering
turn/patch, constraint and evidence deltas, claim changes, usage, provider
execution label, status, and unresolved questions.

Failed updates do not make the old version current for the new request. They
retain unaffected claims, mark affected content unresolved, and leave the old
answer available only through version history. Formatting versions use the
current valid claims and citations, perform zero corpus retrieval, add no
facts, and preserve qualifications. Citation ID/excerpt checks are structural
provenance checks; semantic entailment is a separate review step.

## Evaluation boundary

The Phase 4 suite is a synthetic engineering evaluation with 22 cases: 13
development cases, three previously inspected diagnostic/regression cases,
and six untouched held-out generalisation cases. Related variants remain in
their family split. The matched arms use the same interpretation patch,
synthetic corpus/index, lexical retriever/top-k, mock provider, generation
configuration, and transcript. Arm A performs a complete updated-task search
and regeneration; arm B performs dependency-aware selective updates.

The machine-readable and Markdown results are in
[`../reports/phase4_evaluation.json`](../reports/phase4_evaluation.json) and
[`../reports/phase4_evaluation.md`](../reports/phase4_evaluation.md). The
claim sheet is intentionally pending human semantic review. The real-backend
attempt and redacted configuration are in
[`../reports/phase4_real_e2e.json`](../reports/phase4_real_e2e.json).

## Limits

- State and scheduler runtimes are process-local and bounded; no persistence,
  retention service, or cross-session profile is provided.
- Rule-based follow-up interpretation and the repository mock provider test
  contract behavior, not real-model quality.
- Exact source excerpts and valid citation IDs do not prove semantic support;
  human claim review is still required.
- The official corpus, labels, benchmark runner, organiser API, and a
  configured real generation backend are unavailable in this workspace.
- This work stops at Phase 4 evaluation and handoff; final submission
  production is intentionally out of scope.
