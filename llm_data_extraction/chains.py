import json
from typing import Dict, Callable

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

from llm_data_extraction.log_callback_handler import LogCallbackHandler
from llm_data_extraction.prompts import EXTRACT_PROMPT, VALIDATE_PROMPT
from llm_data_extraction.templates import TEMPLATE_INSTRUCTIONS


# look at it as fast in-house model optimised for domain-specific structured extraction
def get_extract_llm(api_key: str) -> BaseLanguageModel:
    return ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.0, max_tokens=200,
                      callbacks=[LogCallbackHandler()], tags=["extract_llm"])


# fast model to detect hallucinations (cost is proportional to generated tokens)
def get_validate_llm(api_key: str) -> BaseLanguageModel:
    return ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0.0, max_tokens=2,
                      callbacks=[LogCallbackHandler()], tags=["validate_llm"])


# slow, powerful and expensive model to use for fallback when the extract model failed to do it job
def get_resolve_llm(api_key: str) -> BaseLanguageModel:
    return ChatOpenAI(api_key=api_key, model="gpt-4o", temperature=0.0, max_tokens=300,
                      callbacks=[LogCallbackHandler()], tags=["resolve_llm"])


def pre_process_chain(input_: Dict) -> Runnable:
    """
    Extracts and flattens necessary input fields
    Expects: request body
    Returns: request_id, language, template
    """
    return RunnableParallel(
        transcript=lambda x: json.dumps(x["in"]["transcript"]),
        request_id=lambda x: x["request_id"],
        language=lambda x: x["in"]["language"],
        template=lambda x: x["out"],
    )


def extract_chain(llm: BaseLanguageModel) -> Runnable:
    """
    Extracts template field from transcript according instruction and translated to language.
    Expects: instructions, transcript, language
    :param llm: LLM for extraction
    :return: Extracted field text.
    """
    return ChatPromptTemplate.from_template(EXTRACT_PROMPT) | llm | StrOutputParser()


def validate_chain(llm: BaseLanguageModel):
    """
    Validates if the extracted template field data was correctly extracted from the transcript following according
    to the instructions.
    Expects: language, extracted, instructions, transcript
    :param llm: LLM for validation
    :return: 'true' or 'false'
    """
    return ChatPromptTemplate.from_template(VALIDATE_PROMPT) | llm | StrOutputParser()


def resolve_chain(llm) -> Callable[[Dict], Runnable]:
    """
    Resolves invalidly extracted field information by using more powerful LLM for extraction.
    Expects: same as the extract_chain
    :param llm: LLM for validation
    :return:
    """
    def _resolve_chain(input_: Dict) -> Runnable:
        if input_["valid"] == "true":
            return RunnableLambda(lambda x: x["extracted"])
        else:
            return extract_chain(llm)

    return _resolve_chain


def field_chain(field: str, instructions: str, extract_llm: BaseLanguageModel, validate_llm: BaseLanguageModel,
                resolve_llm: BaseLanguageModel) -> Runnable:
    """
    Builds single field extraction chain. Includes steps for results validation and wrong result resolution.
    So teh flow is: prepare_inputs -> extract -> validate -> resolve -> prepare_result.
    Uses different LLM in sake of cost considerations.
    For every field it calculates the following structure:
    {
        "field": "Field name as in teh template"
        "instructions": "Field extraction instructions"
        "extracted": "Extracted field information (using the extraction LLM)"
        "valid": "true if the extracted information is  valid and false otherwise"
        "resolved": "Resolved incorrectly extracted information or copy of extracted information"
    }
    Only the contents of the resolved field is propagated further to the template_chain.
    :param field: field name as in the template
    :param instructions: field extraction instructions as in template
    :param extract_llm: LLM for extraction
    :param validate_llm: LLM for validation
    :param resolve_llm: LLM for resolving possible errors and hallucinations
    :return: Runnable chain
    """
    return RunnablePassthrough()\
        .assign(field=lambda x: field)\
        .assign(instructions=lambda x: instructions)\
        .assign(extracted=extract_chain(extract_llm))\
        .assign(valid=validate_chain(validate_llm))\
        .assign(resolved=resolve_chain(resolve_llm)) | RunnableLambda(lambda x: x["resolved"])


def template_chain(extract_llm: BaseLanguageModel, validate_llm: BaseLanguageModel,
                   resolve_llm) -> Callable[[Dict], Runnable]:
    """
    Assembles template chain from template fields instructions.
    :param extract_llm: LLM for extraction
    :param validate_llm: LLM for validation
    :param resolve_llm: LLM for errors resolution
    :return: Runnable chain
    """
    def _template_chain(input_: Dict) -> Runnable:
        template = input_["template"]
        field_instructions = TEMPLATE_INSTRUCTIONS[template]
        field_chains = {field: field_chain(field, field_instructions[field], extract_llm, validate_llm, resolve_llm)
                        for field in field_instructions}
        return RunnableParallel(field_chains).assign(request_id=lambda x: input_["request_id"])
    return _template_chain


def main_chain(extract_llm: BaseLanguageModel, validate_llm: BaseLanguageModel,
               resolve_llm: BaseLanguageModel) -> Runnable:
    """
    Assembles the main chain.
    :param extract_llm: LLM for extraction
    :param validate_llm: LLM for validation
    :param resolve_llm: LLM for resolving possible extraction errors
    :return: Runnable chain
    """
    return RunnableLambda(pre_process_chain) | template_chain(extract_llm, validate_llm, resolve_llm)
