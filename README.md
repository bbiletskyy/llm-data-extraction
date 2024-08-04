# llm-data-extraction 
LLM based structured data extraction.

Given a transcript of a dialog and a template of data fields to be extracted from the dialog.
A dialog between a doctor and a patient is used as an example. During the dialog in between a small talk the doctor asks 
the patient about their complains and prescribes a treatment. 

OpenAI LLMs extract complain and treatment fields specified in the extraction template. 
Instead of using 
[LangChain's built in capability to returning structured](https://python.langchain.com/v0.2/docs/how_to/structured_output/), 
data we use custom setup due to several reasons:
1. relatively large and slow models are capable of generating structured outputs, while we would like to use small 
and fast in-house model for extraction;
2. because of latency latency considerations we want to extract template fields in parallel;
3. we would like to train our in-house extraction model, which is easier to do on a field level 
rather then on the whole document;
4. we would like to generate results in streaming mode.
  
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
2. Run `docker run -d -p 8000:8000 -e OPENAI_API_KEY=your-api-key llm-data-extraction`
3. Run `curl http://localhost:8000/invoke -X POST -H "Content-Type: application/json" --data @request_1.json`

## How to test
There are 2 types of tests: `fast` and `slow`. 
* `fast` tests use fake LLMs with predefined responses. They are meant to test the setup of chains.
* `slow` low latency tests using real LLM (require an OpenAI api key)

To run tests:
1. Install development dependencies to your project environment ```pip install -e .\[dev\]```
2. Run `fast` tests: ```pytest -m fast```
3. Run `slow` tests:
```
export OPENAI_API_KEY=your-api-key
pytest -m slow
```

  

