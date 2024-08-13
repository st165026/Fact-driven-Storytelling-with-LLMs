from bs4 import BeautifulSoup
import requests, json
import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluation.evaluation_pyramid import device
from config.config import bing_api_key


def fetch_documents(query, api_key):
    """
    Uses the Bing Search API to fetch documents related to a given query.

    Args:
    query (str): The search query string.
    api_key (str): The API key for Bing Search API.

    Returns:
    list: A list of dictionaries containing information about each search result.
    """
    endpoint = 'https://api.bing.microsoft.com/v7.0/search'
    headers = {'Ocp-Apim-Subscription-Key': api_key}
    params = {'q': query, 'textDecorations': True, 'textFormat': 'HTML'}

    response = requests.get(endpoint, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()['webPages']['value']
    else:
        print(f"Request failed with status code：{response.status_code}")
        return []



def download_webpage(url):
    """
        Downloads the content of a webpage given its URL.

        Args:
        url (str): URL of the webpage to download.

        Returns:
        str or None: The text content of the webpage, or None if download fails.
        """
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return None



def extract_evidence_from_web(question, api_key):
    # Load the Flan-T5-Large model
    model5 = "chentong00/propositionizer-wiki-flan-t5-large"
    tokenizer5 = AutoTokenizer.from_pretrained(model5)
    model5 = AutoModelForSeq2SeqLM.from_pretrained(model5).to(device)

    # Fetch documents from Bing Search API
    documents = fetch_documents(question, bing_api_key)
    if not documents:
        print("No documents found.")
        return

    # Process each document
    for doc in documents:
        url = doc['url']
        title = doc['name']
        print(f"Title: {title}")
        print(f"URL: {url}")
        # Download the full webpage content
        html_content = download_webpage(url)
        if not html_content:
            continue

        # Extract the main content from the HTML
        main_content = extract_main_content(html_content)
        #print(f"Main Content Extracted: {main_content[:500]}...")  # Show first 500 characters of content

        # Extract propositions, including the query in the input text
        propositions = extract_propositions(question, main_content, model5, tokenizer5)
        #print(f"Propositions: {json.dumps(propositions, indent=2)}\n")

    return propositions

def extract_main_content(html_content):
    """Extract the main article content from HTML using BeautifulSoup"""
    soup = BeautifulSoup(html_content, 'html.parser')
    # Assume the main content is within <article> tags or the most common <p> tags within a div
    article = soup.find('article') or soup.find('div', {'class': 'article'})
    if not article:
        article = soup  # If specific tags not found, fallback to entire soup
    text = article.get_text(separator=' ', strip=True)
    return text

def extract_propositions(question, text, model, tokenizer):
    """Extract propositions from text using a pre-trained model and considering the query"""
    input_text = f"Title: {question}. Section: . Content: {text}"
    #input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    input_ids = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).input_ids.to(device)

    outputs = model.generate(input_ids, max_new_tokens=512)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        prop_list = json.loads(output_text)
    except json.JSONDecodeError:
        prop_list = [output_text]  # Fallback to raw output if JSON parsing fails
    return prop_list