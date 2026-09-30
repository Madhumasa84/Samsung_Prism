"""Local Ollama wire protocol and hosted-provider credential boundaries."""
import io
import json
from unittest import mock

import pytest

from flowcontext.config import Settings
from flowcontext.contracts import EvidencePassage, GenerationConfig, GenerationRequest
from flowcontext.generation import (
    GenerationUnavailable, GenerationProviderError, OllamaGenerationProvider,
    OpenAICompatibleGenerationProvider, generation_provider_for_settings,
)


def local_settings(**overrides):
    return Settings(generation_backend='openai_compatible', generation_provider='ollama',
                    generation_model='qwen2.5:3b',
                    generation_base_url=overrides.pop('generation_base_url', 'http://127.0.0.1:11434/v1'),
                    **overrides)


def test_ollama_schema_and_no_credential_forwarding(monkeypatch):
    provider = generation_provider_for_settings(local_settings())
    assert isinstance(provider, OllamaGenerationProvider)
    answer = {'answer_text': 'Venue B hosts 40 attendees.', 'factual_claims': [],
              'uncertainty': 'Fixture response.', 'answer_version': 1}
    envelope = {'choices': [{'message': {'content': json.dumps(answer)}}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 8, 'total_tokens': 18}}
    response = io.BytesIO(json.dumps(envelope).encode())
    response.headers = {}
    response.status = 200
    passage = EvidencePassage(chunk_id='chunk-1', source_location='fixture://doc',
                              text='Venue B hosts 40 attendees.', rank=1, score=1, retrieval_method='lexical_overlap')
    # A hosted API key must neither be required nor sent to Ollama.
    monkeypatch.setenv('FLOWCONTEXT_GENERATION_API_KEY', 'hosted-secret')
    with mock.patch('flowcontext.generation.urlopen', return_value=response) as network:
        result = provider._generate_once(GenerationRequest(query='Which venue?', passages=[passage]))
    request = network.call_args.args[0]
    assert request.full_url == 'http://127.0.0.1:11434/v1/chat/completions'
    assert not request.has_header('Authorization')
    payload = json.loads(request.data)
    assert payload['model'] == 'qwen2.5:3b'
    assert payload['response_format']['type'] == 'json_schema'
    schema = payload['response_format']['json_schema']['schema']
    assert set(schema['properties']) == set(answer)
    assert 'verification_audit' not in schema['properties']
    assert json.loads(result.raw_text) == answer
    assert result.usage.total_tokens == 18


@pytest.mark.parametrize('url', ['https://remote.example/v1', 'http://localhost.remote.example/v1',
                                 'http://user:password@localhost:11434/v1'])
def test_local_provider_rejects_nonlocal_or_credential_bearing_urls(url):
    with pytest.raises(ValueError, match='loopback'):
        generation_provider_for_settings(local_settings(generation_base_url=url))


def test_ollama_works_without_an_api_key(monkeypatch):
    monkeypatch.delenv('FLOWCONTEXT_GENERATION_API_KEY', raising=False)
    assert generation_provider_for_settings(local_settings())._authorization_headers() == {}


def test_hosted_provider_still_requires_credentials(monkeypatch):
    monkeypatch.delenv('FLOWCONTEXT_GENERATION_API_KEY', raising=False)
    config = GenerationConfig(backend='openai_compatible', provider='hosted', model='test',
                              base_url='https://example.invalid/v1', api_key_env='FLOWCONTEXT_GENERATION_API_KEY')
    with pytest.raises(GenerationUnavailable):
        OpenAICompatibleGenerationProvider(config=config)._authorization_headers()


def test_compact_prompt_preserves_passages_intent_bindings_and_repair_feedback():
    provider = generation_provider_for_settings(local_settings())
    passage = EvidencePassage(chunk_id='venue', source_location='fixture://venue',
                              text='Venue B hosts 40 attendees.', rank=1, score=1, retrieval_method='lexical_overlap')
    catering = passage.model_copy(update={'chunk_id': 'catering', 'text': 'Vegetarian catering is available.'})
    intents = [{'intent_id': 'i1', 'query': 'Venue for 40 attendees?', 'constraints': [],
                'relationship': 'independent', 'source_span': {'start': 0, 'end': 10}}]
    request = GenerationRequest(query='Venue and catering?', passages=[passage],
                                intent_queries=['Venue for 40 attendees?', 'Catering?'],
                                decomposed_intents=intents, shared_constraints=[{'city': 'Pune'}],
                                intent_evidence={'i1': [passage, passage], 'i2': [catering]},
                                repair_feedback='Use only supplied citation IDs.')
    content = json.loads(provider._request_payload(request)['messages'][1]['content'])
    assert content['retrieved_passages'] == [{'chunk_id': 'venue', 'text': passage.text},
                                            {'chunk_id': 'catering', 'text': catering.text}]
    assert content['evidence_by_intent'] == {'i1': ['venue'], 'i2': ['catering']}
    assert content['intents'][0]['intent_id'] == 'i1'
    assert content['intents'][0]['query'] == intents[0]['query']
    assert content['shared_constraints'] == request.shared_constraints
    assert content['repair_feedback'] == request.repair_feedback
    schema = provider._request_payload(request)['response_format']['json_schema']['schema']
    claim = schema['$defs']['FactualClaim']
    assert 'intent_ids' in claim['required']
    assert claim['properties']['intent_ids']['items']['enum'] == ['i1', 'i2']
    assert claim['properties']['supporting_chunk_ids']['items']['enum'] == ['venue', 'catering']


def test_conflicting_text_for_one_citation_id_is_rejected():
    provider = generation_provider_for_settings(local_settings())
    passage = EvidencePassage(chunk_id='venue', source_location='fixture://venue', text='Original evidence.',
                              rank=1, score=1, retrieval_method='lexical_overlap')
    conflict = passage.model_copy(update={'text': 'Conflicting evidence.'})
    request = GenerationRequest(query='Venue?', passages=[passage], intent_evidence={'i1': [conflict]})
    with pytest.raises(GenerationProviderError, match='conflicting passage'):
        provider._request_payload(request)
