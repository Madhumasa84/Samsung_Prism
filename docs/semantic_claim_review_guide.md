# Semantic Claim Review Guide

## Purpose

This guide provides instructions for conducting independent human semantic claim review for FlowContext. The goal is to verify that emitted factual claims are semantically entailed by their cited corpus passages, not just structurally linked.

## Distinction: Structural vs. Semantic Validity

### Structural Citation Validity (Automated)
- Citation IDs resolve to valid chunk IDs in the corpus
- Source locations and excerpts are correctly preserved
- No broken or missing citations
- This is automatically checked by the system

### Semantic Claim Support (Human Review Required)
- The claim meaning is logically entailed by the cited passage
- No inferences beyond what the passage explicitly states
- No factual contradictions between claim and passage
- No missing critical context that would change the claim's validity

## Review Process

### 1. Preparation
- Review the claim review sheet: `reports/phase4_evaluation_claim_review.csv`
- Each row contains: case_id, strategy, answer_version, claim_id, claim_text, supporting_chunk_ids, supporting_passages, source_locations
- Content hashes are provided for claim and supporting passages to ensure review reproducibility

### 2. Review Criteria
For each claim, assess:

**Supported**: The claim is directly entailed by the cited passage(s)
- The passage explicitly states the information in the claim
- No reasonable interpretation of the passage would contradict the claim
- The claim does not add new information not present in the passage

**Unsupported**: The claim is not entailed by the cited passage(s)
- The claim contains information not present in the passage
- The claim contradicts the passage
- The claim requires inference beyond what the passage provides
- The passage is insufficient to support the claim

**Uncertain**: The support status is ambiguous
- The passage partially supports the claim but is missing critical context
- Multiple interpretations are possible
- The claim spans multiple passages with conflicting information

### 3. Review Template
For each claim, document:
- Claim text and cited passages
- Assessment: supported/unsupported/uncertain
- Reasoning: brief explanation of the decision
- Confidence level: high/medium/low

### 4. Quality Assurance
- Two independent reviewers when possible
- Resolve disagreements through discussion
- Document all review decisions with reasoning
- Maintain version control of review sheets

## Current Status

### Completed Reviews
- **Codex manual claim review**: 89/89 synthetic claims reviewed
- Result: 100% support rate on synthetic fixture data
- Limitation: This is not independent human ground truth
- Location: `reports/phase4_evaluation_claim_review.csv`

### Pending Reviews
- **Independent human review**: NOT COMPLETED
- Required for: Official competition validation
- Scope: All emitted claims across official evaluation corpus
- Status: Awaiting official corpus and independent reviewers

## Template for Independent Review

When official corpus becomes available, use this template:

```csv
case_id,strategy,answer_version,claim_id,claim_text,supporting_chunk_ids,supporting_passages,semantic_support_verdict,reviewer,review_notes,confidence_level,review_date
```

### Reviewer Guidelines
1. **Reviewers**: Use "human_reviewer" or specific reviewer IDs
2. **Verdicts**: Use only "supported", "unsupported", or "uncertain"
3. **Notes**: Provide specific reasoning for each decision
4. **Confidence**: Rate your certainty in the assessment
5. **Date**: Record when the review was completed

## Integration with Evaluation Pipeline

The evaluation pipeline (`src/flowcontext/phase4_evaluation.py`) automatically:
- Ingests review sheets with matching content hashes
- Calculates semantic support rate from verified reviews
- Updates `capability_status["semantic_support"]` to PASS when threshold met
- Preserves review provenance in evaluation reports

## Thresholds

- **Target**: 85% semantic support rate (competition requirement)
- **Current synthetic fixture result**: 100% (89/89 supported)
- **Official validation**: PENDING independent human review on official corpus

## Common Pitfalls

1. **Confusing structural validity with semantic support**: Valid citations do not guarantee semantic entailment
2. **Over-inferring**: Assuming the passage supports claims that require additional context
3. **Missing contradictions**: Not identifying when claims contradict cited passages
4. **Reviewer bias**: Allowing prior knowledge to influence assessment beyond passage content

## Resources

- Current claim review sheet: `reports/phase4_evaluation_claim_review.csv`
- Evaluation reports: `reports/phase4_evaluation.md`, `reports/phase4_evaluation.json`
- Label review status: `data/evaluation/phase4_label_review_status.json`
- Architecture documentation: `docs/architecture-phase4.md`