import json
import datetime
import requests
from bs4 import BeautifulSoup
import re
from typing import Any, Callable, Set, Dict, List, Optional


def ask_knowledge_agent(query: str) -> str:
    """
    Request information about a query that may not be directly available on Wikipedia or requires general knowledge. Use this for complex questions or when you need information beyond Wikipedia. Use this action for relationships between entities or topics, or when the entity for this action is a combination of entities or topics.

    Args:
        query (str): The search query to find relevant information

    Returns:
        str: The knowledge information as a JSON string
    """
    # Mock knowledge base response
    knowledge_response = {
        "result": f"Mock knowledge base result for query: {query}",
        "confidence": 0.85,
    }
    return json.dumps(knowledge_response)


def search_wiki(query: str) -> str:
    """
    Retrieve content from a Wikipedia page based on a search query.
    Request information about a specific topic, entity, or fact from Wikipedia. Use this for direct lookups about entities, events, or topics that have Wikipedia articles, one important condition for this action is that the entity for this action must be a single entity or topic and no combination of entities or topics.

    Args:
        query (str): The search query or page title to retrieve

    Returns:
        str: The extracted content from the Wikipedia page
    """
    # Clean the query and prepare it for URL
    search_term = query.replace(" ", "_")
    search_term = search_term.capitalize()
    print("SEARCHING WIKIPEDIA FOR: ", search_term)

    # Try to directly access the page first
    url = f"https://en.wikipedia.org/wiki/{search_term}"

    # Send GET request
    response = requests.get(url)

    # If direct access fails, try search
    if response.status_code != 200:
        # Search Wikipedia API
        search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={search_term}&limit=1&namespace=0&format=json"
        search_response = requests.get(search_url)

        if search_response.status_code != 200:
            return "Error: Could not search Wikipedia"

        search_data = search_response.json()

        # Check if results found
        if len(search_data[1]) == 0:
            return f"No Wikipedia page found for '{query}'"

        # Get the first result's URL
        page_title = search_data[1][0]
        url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"

        # Try again with the search result
        response = requests.get(url)

        if response.status_code != 200:
            return f"Error: Could not retrieve Wikipedia page for '{query}'"

    # Parse HTML content
    soup = BeautifulSoup(response.text, "html.parser")

    # Get the page title
    # page_title = soup.find("h1", {"id": "firstHeading"}).text

    # Find the main content div
    content = soup.find("div", {"id": "mw-content-text"})
    if not content:
        return "Error: Could not find content section"

    # Get all paragraphs
    paragraphs = content.find_all("p")

    # Clean and combine text
    # clean_text = f"Wikipedia article: {page_title}\n\n"

    for para in paragraphs:
        # Convert to text
        text = para.get_text()

        # Remove reference numbers like [1], [2], etc.
        text = re.sub(r"\[\d+\]", "", text)

        # Remove extra whitespace
        text = " ".join(text.split())

        # print(text[:2000])

    return text.strip()


hotpotqa_functions: Set[Callable[..., Any]] = {
    ask_knowledge_agent,
}

fever_functions: Set[Callable[..., Any]] = {
    search_wiki,
    ask_knowledge_agent,
}
