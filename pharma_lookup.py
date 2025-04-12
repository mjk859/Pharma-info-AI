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
    email_context: str
    phone: str
    phone_context: str
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
                return soup
            else:
                print(f"⚠️ Failed to fetch {url} - Status Code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error fetching {url}: {e}")
        return None

    # Collect all pages and parse them
    suffixes = ["/contact", "/contacts", "/about", "/support", "/help"]
    pages = []

    for suffix in suffixes:
        url = website.rstrip("/") + suffix
        soup = fetch_and_parse(url)
        if soup:
            print(f"🔍 Parsed HTML from {url}")
            pages.append(soup)

    # Fallback to homepage
    if not pages:
        fallback = fetch_and_parse(website)
        if fallback:
            print(f"🔍 Fallback to homepage: {website}")
            pages.append(fallback)

    if not pages:
        return {"email": "Error fetching site", "phone": "Error fetching site"}

    # Initialize
    email, email_context = "Not found", "No context found"
    phone, phone_context = "Not found", "No context found"

    for soup in pages:
        # Remove scripts and styles
        for script in soup(["script", "style"]):
            script.decompose()

        # Search for emails
        email_match = soup.find(string=re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"))
        if email_match:
            email = email_match.strip()
            container = email_match.find_parent("div") or email_match.find_parent()
            email_context = container.get_text(separator="\n", strip=True) if container else "No context found"
            break

    for soup in pages:
        # Same cleaning
        for script in soup(["script", "style"]):
            script.decompose()

        phone_match = soup.find(string=re.compile(r"(\+?\d[\d\-\(\) ]{7,}\d)"))
        if phone_match:
            phone = phone_match.strip()
            container = phone_match.find_parent("div") or phone_match.find_parent()
            phone_context = container.get_text(separator="\n", strip=True) if container else "No context found"
            break

    return {
        "email": email,
        "email_context": email_context,
        "phone": phone,
        "phone_context": phone_context
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

def print_results(company_name, official_url, email, email_context, phone, phone_context, products):
    print("\n🔍 Pharma Lookup - Powered by Gemini Flash\n")

    print(f"🔹 Company Name: {company_name.title()}")
    print(f"🔗 Official Website: {official_url}\n")

    print("📞 Contact Information:")
    print("───────────────────────")
    if phone != "Not found":
        print(f"{phone_context.strip()}\n📞 {phone.strip()}\n")
    else:
        print("📞 Phone: Not found\n")

    if email != "Not found":
        print("📨 Email Contacts:")
        print("───────────────────────")
        print(f"📧 {email}")
        print(f"📄 Context: {email_context}\n")
    else:
        print("📨 Email: Not found\n")

    if products:
        print("💊 Key Products:")
        print("───────────────────────")
        for i, prod in enumerate(products, 1):
            print(f"{i}. {prod}")
    else:
        print("💊 Key Products: None found")

    print("\n" + "━" * 60 + "\n")


def main():
    print("🔍 Pharma Lookup - Powered by Gemini Flash")
    company = input("Enter a pharmaceutical company name: ").strip()

    if not company:
        print("❌ Please enter a valid company name.")
        return

    app = build_graph()
    result = app.invoke({"company": company})

    # Extract results from the graph output
    official_url = result.get("website", "Not found")
    email = result.get("email", "Not found")
    email_context = result.get("email_context", "No context found")
    phone = result.get("phone", "Not found")
    phone_context = result.get("phone_context", "No context found")
    products = result.get("products", "").split("\n") if result.get("products") else []

    # Call the print_results function to display the output
    print_results(company, official_url, email, email_context, phone, phone_context, products)


if __name__ == "__main__":
    main()
