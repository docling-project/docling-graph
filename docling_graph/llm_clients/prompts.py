def get_prompt(markdown_content: str, schema_json: str, is_partial: bool = False) -> str:
    """
    Generates the appropriate prompt based on whether we expect
    a full or partial JSON response.
    """
    if is_partial:
        return f"""
        Based *only* on the document page text below, extract any information
        you find that matches the Pydantic JSON schema. It is expected that
        you will only find partial data.
        
        Return a valid JSON object. Your response MUST be a JSON object.

        --- DOCUMENT PAGE ---
        {markdown_content}
        --- END PAGE ---

        --- PYDANTIC JSON SCHEMA ---
        {schema_json}
        --- END SCHEMA ---
        """
    else:
        return f"""
        Based *only* on the *entire document text* below, return a valid JSON object
        that strictly adheres to the given Pydantic JSON schema.
        
        Return a valid JSON object. Your response MUST be a JSON object.

        --- DOCUMENT ---
        {markdown_content}
        --- END DOCUMENT ---

        --- PYDANTIC JSON SCHEMA ---
        {schema_json}
        --- END SCHEMA ---
        """
