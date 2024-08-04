import os
from typing import Dict

import pytest
from langchain_core.language_models import BaseLanguageModel, FakeListLLM
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from llm_data_extraction.chains import pre_process_chain, extract_chain, validate_chain, main_chain, field_chain
from llm_data_extraction.templates import SOAP_EN_INSTRUCTIONS


def get_fake_llm(response: str) -> BaseLanguageModel:
    return FakeListLLM(responses=[response])


def get_test_llm(api_key: str) -> BaseLanguageModel:
    return ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0.0, max_tokens=200, tags=["test_llm"])


@pytest.mark.fast
def test_pre_process_chain():
    input = {
        "request_id": 1234,
        "in": {
            "transcript": [
                {
                    "speaker": "Doctor",
                    "message": "Hello! How are you today?"
                },
                {
                    "speaker": "Patient",
                    "message": "Good afternoon, Doctor. I'm doing okay, thanks."
                },
            ],
            "language": "english"
        },
        "out": "soap_en"
    }

    res = RunnableLambda(pre_process_chain).invoke(input)
    print(res)
    assert res.get("transcript") == """[{"speaker": "Doctor", "message": "Hello! How are you today?"}, {"speaker": "Patient", "message": "Good afternoon, Doctor. I\'m doing okay, thanks."}]"""
    assert res.get("language") == "english"
    assert res.get("template") == "soap_en"


@pytest.mark.fast
def test_extract_chain():
    inputs = {
        "transcript": "<dummy transcript>",
        "instructions": "<dummy instructions>",
        "language": "<dummy language>"
    }
    expected = "<dummy extracted>"
    res = extract_chain(get_fake_llm(expected)).invoke(inputs)
    assert expected == res


@pytest.mark.fast
def test_field_chain():
    inputs = {
        "transcript": "<dummy transcript>",
        "language": "<dummy language>"
    }
    field = "<dummy field>"
    instructions = "<dummy instructions>"
    extract_llm = get_fake_llm("<dummy extracted>")
    validate_llm = get_fake_llm("true")
    resolve_llm = get_fake_llm("<dummy resolved>")
    res = field_chain(field, instructions, extract_llm, validate_llm, resolve_llm).invoke(inputs)

    assert "<dummy extracted>" == res


@pytest.mark.fast
def test_field_chain_invalid():
    inputs = {
        "transcript": "<dummy transcript>",
        "language": "<dummy language>"
    }
    field = "<dummy field>"
    instructions = "<dummy instructions>"
    extract_llm = get_fake_llm("<dummy extracted>")
    validate_llm = get_fake_llm("false")
    resolve_llm = get_fake_llm("<dummy resolved>")
    res = field_chain(field, instructions, extract_llm, validate_llm, resolve_llm).invoke(inputs)

    assert "<dummy resolved>" == res


@pytest.mark.slow
def test_main_chain():
    api_key = os.getenv("OPENAI_API_KEY")
    extract_llm = get_test_llm(api_key)
    validate_llm = get_test_llm(api_key)
    resolve_llm = get_test_llm(api_key)
    inputs = get_test_input()

    res = main_chain(extract_llm, validate_llm, resolve_llm).invoke(inputs)
    for field in SOAP_EN_INSTRUCTIONS:
        assert field in res


def get_test_input() -> Dict:
    return {
        "request_id": 1234,
        "in": {
            "transcript": [
                {
                    "speaker": "Doctor",
                    "message": "Hello! How are you today?"
                },
                {
                    "speaker": "Patient",
                    "message": "Good afternoon, Doctor. I'm doing okay, thanks."
                },
                {
                    "speaker": "Doctor",
                    "message": "Glad to hear that. What brings you in today? Any specific concerns or complaints?"
                },
                {
                    "speaker": "Patient",
                    "message": "Well, I've been having some headaches and a bit of a sore throat for the past few days."
                },
                {
                    "speaker": "Doctor",
                    "message": "I'm sorry to hear that. Have the headaches been persistent, or do they come and go?"
                },
                {
                    "speaker": "Patient",
                    "message": "They come and go, mostly in the afternoon. The sore throat is more constant, though."
                },
                {
                    "speaker": "Doctor",
                    "message": "Any other symptoms, like a fever or cough?"
                },
                {
                    "speaker": "Patient",
                    "message": "No fever, but I do have a slight cough, nothing major."
                },
                {
                    "speaker": "Doctor",
                    "message": "Alright. Have you been experiencing any stress or lack of sleep recently?"
                },
                {
                    "speaker": "Patient",
                    "message": "A bit stressed, I guess. Work has been hectic, and I've not been sleeping well."
                },
                {
                    "speaker": "Doctor",
                    "message": "That could be contributing to your headaches. Let's do a quick examination. I'll check your throat and take your temperature."
                },
                {
                    "speaker": "Patient",
                    "message": "Sure, go ahead."
                },
                {
                    "speaker": "Doctor",
                    "message": "Your throat looks a little inflamed. No signs of infection, but I'll prescribe something to help with the soreness. For the headaches, it might be good to manage stress and ensure you're getting enough rest."
                },
                {
                    "speaker": "Patient",
                    "message": "Okay, thank you. What should I take for the sore throat?"
                },
                {
                    "speaker": "Doctor",
                    "message": "I'll prescribe a mild pain reliever and a throat lozenge. Make sure to stay hydrated and try to rest as much as you can."
                },
                {
                    "speaker": "Patient",
                    "message": "Got it. Anything else I should do?"
                },
                {
                    "speaker": "Doctor",
                    "message": "If your symptoms persist for more than a week or worsen, please come back for a follow-up. Otherwise, these measures should help. Take care and feel better soon!"
                },
                {
                    "speaker": "Patient",
                    "message": "Thanks, Doctor. I'll follow your advice."
                },
                {
                    "speaker": "Doctor",
                    "message": "You're welcome. Have a good day!"
                }
            ],
            "language": "english"
        },
        "out": "soap_en"
    }
