# llm-data-extraction 
LLM based structured data extraction.

Given a transcript of a dialog and a template of data fields to be extracted from the dialog.
A dialog between a doctor and a patient is used as an example. During the dialog in between a small talk the doctor asks 
the patient about their complains and prescribes a treatment. 

OpenAI LLMs extract complain and treatment based on the extraction template.  
The extraction template contains data field name and description (extraction instructions). 
Another LLM verifies extracted data for correctness and for possible hallucinations. 
And yet another LLM resolves incorrectly extracted data if needed.

Different LLMs are used to address latency, costs and efficiency: 
* extraction LLM is meant to be a small and fast in-house LLM optimised for domain-specific tasks
* validation LLM is meant to be fast general purpose LLM producing only few tokens and capable of assessing results of the extraction LLM
* resolving LLM is meant to be a general purpose powerful, slow and expensive model, which is launched only when the extraction model did not manage to do it's job. 
The resolving LLM used the same prompt as the extraction LLM. The outputs of the resolving LLM are used for training the extraction LLM.                     


## How to run 
1. Run `docker build -t llm-data-extraction .`
2. Run `docker run -d -p 8000:8000 -e OPENAI_API_KEY=<your open ai key> llm-data-extraction`
3. Run `curl http://localhost:8000/invoke -X POST -H "Content-Type: application/json" --data @request_1.json`

## How to test



There are 2 types of tests: `fast` and `slow`. 
* `fast` tests use fake LLMs with predefined responses. They are meant to test the setup of chains.
 
```
pytest -m fast
``` 
* `slow` low latency tests using real LLM (require an OpenAI api key) 
 
```
export OPENAI_API_KEY=your-api-key
pytest -m slow
```

  

