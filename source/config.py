class Config:

    '''
    contains the text cleaning regexes and the corresponding substring replacements
    '''
    REGEX_REPLACEMENTS_DICT = {
            r"(\b|^)/v1/\S*" : " <ENDPOINT> ", # the endpoint
            r" +" : " ", #replacing multiple spaces with single space
            r"\d+" : " <NUMBER> ", # replacing the numbers with tags
            
    }

    '''
    the id of the sentences transformer pre-trained model to use
    '''
    SENTENCE_TRANSFORMER_MODEL_ID = "all-MiniLM-L6-v2"


    '''
    the tfidf configuration applied to each of the clean versions of the documentation sentences
    '''
    TFIDF_VEC_CONF = {
        "min_df" : 1,
        "ngram_range" : (1,2)
    }

    '''
    the number of most similar documentations returned
    '''
    NUM_OF_MOST_SIM_DOCS_RETURNED = 5

    '''
    the path to the local location where the database documentations embeddings are saved
    '''
    DATABASE_EMBDEDDINGS_PATH = "../data"

    '''
    the ID of the openai foundation model used
    '''
    OPENAI_MODEL_USED = "gpt-4.1"

    '''
    the content prompt format used for the standardization and restructuring of the input documentation
    '''
    CONTENT_PROMPT_FORMAT = "Input Text: {}"

    '''
    the system prompt used
    '''
    SYSTEM_PROMPT = """
    You are an expert in API documentation analysis. Given an input document, perform the following tasks:
1. Restructure the document into the following standardized sections
- What is this API?: A clear description of the API’s purpose, what kind of data it handles, and its core functionality.
- Why is it useful?: Benefits of using the API, such as automation, personalization, efficiency, or business value.
- Who is it for?: Intended audience or users of the API (e.g., developers, analysts, business teams).
- How does it work?: A step-by-step explanation of how the API functions, including onboarding, authentication, token usage, and data flow.
- API Endpoints: A list or table of endpoints with details like endpoint name, URL, HTTP method (GET/POST), and function.
- Example API Call: A sample API request including method, headers, endpoint, and expected response format.
2. Validate each section for completeness
For each section:
- Indicate whether it is present or missing.
- If present, assess whether it includes all expected elements.
- Provide detailed feedback and suggestions for improvement, such as:
  • 'The API Endpoints section is present but missing the URL field.'
  • 'The Example API Call is present but lacks authorization headers.'
  • 'The ‘How does it work?’ section could be improved by adding token flow details.'
3. Return the result as a JSON object with two attributes
- "standardized_document": A markdown document with section names as paragraph titles and extracted content as paragraph contents.
- "missing_information_feedback": A markdown document with section names as paragraph names and feedback about missing or incomplete content as paragraph contents. Include suggestions where applicable.
Here is an example of input and output.
Input text: 
The API helps with customer data. It gives access to things like demographics, product info, and how customers interact. Works for both personal and business customers. Can be used with personalization tools and analytics.
It can be used to make customer journeys better, don’t need to manually look up data, and marketing teams can make faster decisions. Also helps with targeting and segmentation and it can be used by dev teams, analysts, and business people who want to build stuff or understand customers.
To get started, you need to register your app and get credentials. Then you use OAuth to get a token. After that, you can call the API and get data. The responses are in JSON.
There are endpoints for personal and commercial profiles. One is /v1/personalCustomerProfile/cin and the other is /v1/nonPersonalCustomerProfile/bin. Both use POST and give back product data.
Here’s how you call it:
POST /v1/personalCustomerProfile/cin HTTP/1.1
Host: api.example.com
Authorization: Bearer <access_token>
Content-Type: application/json
x-fapi-interaction-id: <unique-id>
{
"cin": "123456789"
}

Model output example: 
{
{
  "standardized_document": 
    "### What is this API?": "The API helps with customer data. It gives access to things like demographics, product info, and how customers interact. Works for both personal and business customers. Can be used with personalization tools and analytics.",
    ### Why is it useful? 
    You can make customer journeys better, don’t need to manually look up data, and marketing teams can make faster decisions. Also helps with targeting and segmentation.
    ### Who is it for 
    Dev teams, analysts, and business people who want to build stuff or understand customers.
    ### How does it work? 
    To get started, you need to register your app and get credentials. Then you use OAuth to get a token. After that, you can call the API and get data. The responses are in JSON.
    ### API Endpoints 
    There are endpoints for personal and commercial profiles. One is /v1/personalCustomerProfile/cin and the other is /v1/nonPersonalCustomerProfile/bin. Both use POST and give back product data.
    ### Example API Call 
    POST /v1/personalCustomerProfile/cin HTTP/1.1\nHost: api.example.com\nAuthorization: Bearer <access_token>\nContent-Type: application/json\nx-fapi-interaction-id: <unique-id>\n{\n  \"cin\": \"123456789\"\n}"
  "missing_information_feedback": 
    "### What is this API? 
    ✓ Present
    Covers purpose and supported data types. Could be improved by explicitly stating the API name and its core functionality.
    ### Why is it useful? 
    ✓ Present. 
    Includes benefits like personalization and segmentation. Could be improved by linking benefits to specific use cases.
    ### Who is it for? 
    ✓ Present 
    Clearly identifies target users. Could be improved by using more formal terminology (e.g., 'business teams' instead of 'business people')
    ### How does it work?
    ✓ Present 
    Includes onboarding, token usage, and response format. Could be improved by adding more detail on authentication flow and error handling.
    ### API Endpoints 
    ✓ Present
    Includes endpoint names, URLs, and methods. Could be improved by formatting as a table and adding descriptions for each endpoint.
    ### Example API Call
    ✓ Present 
    Includes method, headers, and payload. Could be improved by specifying expected response format and status codes."
  }
}
"""

    
    

    

    

