import os
import re
import argparse
import requests
from bs4 import BeautifulSoup
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import StateGraph, END

# -------------------- CONFIG --------------------

# Get the API key from the environment variable
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")

# Configure Gemini
genai.configure(api_key=api_key)

# -------------------- HELPERS --------------------

def gemini_chat(messages: list) -> str:
    """
    Accepts a list of message objects like SystemMessage and HumanMessage,
    concatenates their contents, and sends it to Gemini.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = "\n".join([m.content for m in messages])
    response = model.generate_content(prompt)
    return response.text.strip()

# Define dummy replacements for SystemMessage and HumanMessage
class SystemMessage:
    def __init__(self, content):
        self.content = content

class HumanMessage:
    def __init__(self, content):
        self.content = content

# -------------------- GRAPH STATE --------------------

class GraphState(TypedDict):
    company: str
    search_results: str
    website: str
    email: str
    phone: str
    products: str

# -------------------- NODES --------------------

search = DuckDuckGoSearchResults()

def search_company(state):
    company_name = state["company"]
    query = f"{company_name} official website"
    results = search.run(query)
    return {"search_results": results}

def extract_website(state):
    search_results = state["search_results"]
    messages = [
        SystemMessage(content="You are an assistant that helps identify official websites."),
        HumanMessage(content=f"From the following search results, find the official website URL of the company:\n\n{search_results}")
    ]
    website_response = gemini_chat(messages)

    # Debug: Print raw response
    print(f"Gemini response: {website_response}")

    # Extract a proper URL from Gemini’s full sentence
    match = re.search(r'https?://[^\s"\']+', website_response)
    website = match.group(0).strip() if match else "Invalid URL"

    # Sanitize the URL by removing unwanted characters like '**' or trailing punctuation
    website = re.sub(r'[\*\.,]+$', '', website)

    # Debug: Print sanitized URL
    print(f"Extracted website: {website}")

    return {"website": website}


def get_contact_info(state):
    website = state["website"]

    if not website.startswith("http"):
        return {"email": "Invalid website URL", "phone": "Invalid website URL"}

    def fetch_and_parse(url):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                return soup.get_text(separator="\n")
            else:
                print(f"⚠️ Failed to fetch {url} - Status Code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error fetching {url}: {e}")
        return None

    # Try multiple common paths for contact information
    text = None
    for suffix in ["/contact", "/contacts", "/about", "/support", "/help"]:
        contact_url = website.rstrip("/") + suffix
        text = fetch_and_parse(contact_url)
        if text:
            print(f"🔍 Parsed text from {contact_url}:\n{text[:500]}")  # Log first 500 characters
            break

    # Fallback to homepage
    if not text:
        text = fetch_and_parse(website)

    if not text:
        return {"email": "Error fetching site", "phone": "Error fetching site"}

    # Use regex to find potential contact info
    email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    phone_match = re.search(r"(\+?\d[\d\-\(\) ]{7,}\d)", text)

    email = email_match.group(0) if email_match else "Not found"
    phone = phone_match.group(0) if phone_match else "Not found"

    return {
        "email": email,
        "phone": phone
    }


def get_products(state):
    company = state["company"]
    query = f"{company} key pharmaceutical products or late stage clinical trials"
    results = search.run(query)

    messages = [
        SystemMessage(content="You are an assistant that extracts product names from company info."),
        HumanMessage(content=f"From the following search results, list 3 key pharmaceutical products marketed or in late-stage clinical trials by {company}:\n\n{results}")
    ]
    products = gemini_chat(messages)
    return {"products": products}

# -------------------- GRAPH --------------------

def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("search", search_company)
    builder.add_node("extract_website", extract_website)
    builder.add_node("contact_info", get_contact_info)
    builder.add_node("extract_products", get_products)

    builder.set_entry_point("search")
    builder.add_edge("search", "extract_website")
    builder.add_edge("extract_website", "contact_info")
    builder.add_edge("contact_info", "extract_products")
    builder.add_edge("extract_products", END)

    return builder.compile()

# -------------------- CLI --------------------

def main():
    print("🔍 Pharma Lookup - Powered by Gemini Flash")
    company = input("Enter a pharmaceutical company name: ").strip()

    if not company:
        print("❌ Please enter a valid company name.")
        return

    app = build_graph()
    result = app.invoke({"company": company})

    print("\n🔍 Pharma Info Lookup Result:")
    print("=" * 40)
    print("🏢 Company:", company)
    print("🔗 Website:", result.get("website"))
    print("📧 Email:", result.get("email"))
    print("📞 Phone:", result.get("phone"))
    print("💊 Key Products:\n", result.get("products"))
    print("=" * 40)


if __name__ == "__main__":
    main()
