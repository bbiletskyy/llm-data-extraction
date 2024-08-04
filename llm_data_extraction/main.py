import os
from typing import Dict

from fastapi import FastAPI
from langserve import add_routes

from llm_data_extraction.chains import main_chain, get_extract_llm, get_validate_llm, get_resolve_llm

api_key = os.getenv("OPENAI_API_KEY")

chain = main_chain(get_extract_llm(api_key), get_validate_llm(api_key), get_resolve_llm(api_key))

app = FastAPI(title="LLM structural extraction")

add_routes(app=app, runnable=chain, input_type=Dict, output_type=Dict)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
