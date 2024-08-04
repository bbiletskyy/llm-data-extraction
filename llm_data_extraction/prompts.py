EXTRACT_PROMPT = """Given a transcript of a dialog between a doctor and a patient, you need to extract {instructions}
<transcript>
{transcript}
</transcript>
Return results in {language} language. 
Return the summary only. Make sure the summary contains relevant information only. 
If necessary information was missing or was impossible to extract then return 'NA'.
"""

VALIDATE_PROMPT = """Given a transcript of a dialog between a doctor and a patient, information extraction instructions and the extracted information in {language} language. 
Validate for correctness, completeness and possible hallucinations of the extracted information against the extraction instructions and the dialog transcript.

<extracted_information>
{extracted}
</extracted_information>

<extraction_instructions>
{instructions}
</extraction_instructions>

<transcript>
{transcript}
</transcript>

If extracted information is valid and has no hallucinations then return 'true', else return 'false'. Return only 'true' or 'false'.
"""