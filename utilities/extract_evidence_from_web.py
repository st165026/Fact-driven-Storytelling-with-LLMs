from bs4 import BeautifulSoup
import requests
import spacy

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

def print_search_results(results):
    """
    Prints the title and URL of each search result.

    Args:
    results (list): A list of dictionaries, each representing a search result.
    """
    for result in results:
        url = result.get('url')
        title = result.get('name')
        print(f"Title: {title}")
        print(f"Link: {url}")


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

def extract_evidence(text):
    """
    Extracts evidence from text using named entity recognition via SpaCy.

    Args:
    text (str): The text from which to extract entities as evidence.

    Returns:
    list: A list of strings, each a sentence containing a recognized named entity.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    evidence = []
    for ent in doc.ents:
        if ent.label_ in ["DATE", "ORG", "PERSON", "GPE", "NORP", "CARDINAL", "QUANTITY"]:
            sentence = ent.sent.text.strip()
            if sentence not in evidence:
                evidence.append(sentence)
    return evidence

def extract_evidence_from_web(question, api_key, max_char_length=2000):
    """
        Extracts evidence from the web related to a query by first fetching documents and then extracting
        relevant textual evidence from these documents.

        Args:
        question (str): The search query to fetch documents for.
        api_key (str): The Bing API key.
        max_char_length (int): The maximum character length for combined evidence to extract.

        Returns:
        list: A list of evidences extracted from the documents.
        """
    documents = []
    current_length = 0
    results = fetch_documents(question, api_key)
    print_search_results(results)
    if not results:
        print("No related documentation was found.")
        return documents

    for result in results[:8]:
        if current_length >= max_char_length:
            break  # Stop processing new documents if the character limit has been reached
        url = result.get('url')
        title = result.get('name')
        #print(f"Handling of articles: {title} ({url})")
        html_content = download_webpage(url)
        if not html_content:
         #   print("Unable to download web content.") # Some articles require membership to view, so Evidence cannot be extracted.
            continue

        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        evidence = extract_evidence(text)
        for ev in evidence:
            ev_length = len(ev)
            if current_length + ev_length > max_char_length:
                break  # If adding this evidence would exceed the length limit, stop adding it
            documents.append(ev)
            current_length += ev_length
        #print(f"From {title} extracted evidences:")
        #for e in documents[-len(evidence):]:
        #    print("-", e)
        #print()

    return documents