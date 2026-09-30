# AI disclosure addendum

Date: 30 September 2026. Project: FlowContext. Team: Makar.

The accompanying [AI usage disclosure](AI_Usage_Disclosure.docx) is an unchanged export of the team-provided Google document. This addendum records subsequent work and corrects descriptions that no longer match the submitted repository. It does not alter the representative's signature or assert a new human review.

- AI assistance contributed to the Streamlit interface, its professional wording, Ollama compatibility, setup documentation, video integration, verification, and final submission packaging. The original form's statement that no material AI UI/UX contribution was used does not describe this final version.
- The lexical retriever uses unique query-term overlap (`lexical_overlap`), rather than BM25. Dense retrieval uses MiniLM embeddings; hybrid retrieval uses RRF with k=60. Historical prompts mentioning BM25 are preserved in the original form as prompt history.
- The final runtime verification passed 169 automated tests. Earlier counts in the disclosure and assistance log describe earlier revisions.
- Real local generation was tested using Ollama with `qwen2.5:3b`. The adapter works, but the small model did not satisfy every scenario check. Scenario A initially timed out and completed on retry with incomplete intent coverage; B and D also failed coverage checks. C, E and F passed their smoke checks. These are local synthetic-fixture observations, not official benchmark results.
- Automated citation and excerpt checks do not establish independent semantic review. AI-assisted claim reviews and provisional labels must not be presented as independent human ground truth.

See the [repository assistance log](../../AI_ASSISTANCE_LOG.md), [verification summary](verification.json), and committed evaluation reports for scope and limitations. The committed presentation incorporates corrected retrieval terminology, scenario mapping, test count, and evaluation boundaries; the Google authoring version may differ.
