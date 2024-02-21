import streamlit as st
import json
import requests
from bs4 import BeautifulSoup
from langchain.llms import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
import xml.etree.ElementTree as ET
import PyPDF2
from pdfminer.high_level import extract_text
import urllib.parse
from collections import Counter
# Load personas
# Initial API Key Input
st.title("Knowledge assistant  research")
api_key = st.text_input("Enter your OpenAI API key", type="password")
fda_api_key = st.text_input("Enter your FDA API key", type="password")
def validate_api_key(api_key):
    test_endpoint = "https://api.openai.com/v1/models"  # A simple endpoint for validation
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.get(test_endpoint, headers=headers)
        return response.status_code == 200  # Returns True if API key is valid
    except requests.RequestException:
        return False


        


def validate_fda_api_key(fda_api_key):
    test_endpoint = "https://api.fda.gov/drug/event.json"  # Replace with a relevant FDA endpoint
    params = {
        'api_key': fda_api_key,
        'limit': 1  # Request minimal data
    }

    try:
        response = requests.get(test_endpoint, params=params)
        return response.status_code == 200  # True if API key is valid
    except requests.RequestException:
        return False
def load_personas():
    with open(r"Clinical_research_associate_persona.json") as file:
        persona1 = json.load(file)
    with open(r"MedicalWriter_persona.json") as file:
        persona2 = json.load(file)
    return persona1, persona2

# Select persona based on user choice
def select_persona(persona_choice):
    if persona_choice == 'Clinical Research Associate':
        return persona1
    elif persona_choice == 'Medical Writer':
        return persona2

# Format persona context
def format_persona_context(persona):
    internet_browsing_capability = "You have the capability to surf through the internet for information." if persona['bot'].get('internet_browsing', False) else ""
    persona_template = """
        You are {name}. {whoami}
        {conversationwith}
        {traits}
        Goal of this conversation for you: {goal}
        Skills: {skills}
        {internet_browsing_capability}
        Reply based on conversation history provided in 'Context:'
        Reply with prefix '{chatname}:'
        Respond with {responselength} words max.
        """
    return persona_template.format(internet_browsing_capability=internet_browsing_capability, **persona['bot'])
# Function to fetch adverse event data
def fetch_adverse_events(fda_api_key,drug_name, therapeutic_area):
    fda_api_key = fda_api_key  # Replace with your actual API key
    base_url = "https://api.fda.gov/drug/event.json"

    # Construct the query with proper encoding
    query = {
        'search': f'patient.drug.medicinalproduct:"{drug_name}" AND patient.drug.drugindication:"{therapeutic_area}"',
        'limit': 10,
        'api_key': fda_api_key
    }
    query_encoded = urllib.parse.urlencode(query, quote_via=urllib.parse.quote)

    # Combine base URL with encoded query
    full_url = f"{base_url}?{query_encoded}"

    try:
        print("Making request to:", full_url)  # For debugging
        response = requests.get(full_url, timeout=30)
        response.raise_for_status()
        return response.json().get('results', [])
    except requests.exceptions.HTTPError as err:
        st.error(f"HTTP Error: {err}")
        st.text(f"Response status code: {response.status_code}")
        st.text(f"Response content: {response.text}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    return []
def parse_openfda_response(data):
    parsed_data = []
    for item in data:
        report_id = item.get('safetyreportid', 'N/A')
        reactions = [reaction.get('reactionmeddrapt', 'N/A') for reaction in item.get('patient', {}).get('reaction', [])]
        drugs = item.get('patient', {}).get('drug', [])
        for drug in drugs:
            drug_info = {
                'report_id': report_id,
                'medicinal_product': drug.get('medicinalproduct', 'N/A'),
                'drug_indication': drug.get('drugindication', 'N/A'),
                'reactions': reactions
            }
            parsed_data.append(drug_info)
    return parsed_data
def aggregate_data_for_summary(parsed_data):
    drug_counts = Counter()
    indication_counts = Counter()
    reaction_counts = Counter()

    for entry in parsed_data:
        drug_counts[entry['medicinal_product']] += 1
        indication_counts[entry['drug_indication']] += 1
        reactions = entry.get('reactions', [])
        for reaction in reactions:
            reaction_counts[reaction] += 1

    return drug_counts, indication_counts, reaction_counts

def generate_basic_summary(drug_counts, indication_counts, reaction_counts):
    summary = "Adverse Event Report Summary:\n"
    summary += "\nMost Common Drugs:\n"
    for drug, count in drug_counts.most_common(5):
        summary += f"- {drug}: {count} reports\n"
    summary += "\nMost Common Indications:\n"
    for indication, count in indication_counts.most_common(5):
        summary += f"- {indication}: {count} reports\n"
    summary += "\nMost Common Reactions:\n"
    for reaction, count in reaction_counts.most_common(5):
        summary += f"- {reaction}: {count} occurrences\n"
    return summary
def fetch_pubmed_data(query):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = f"{base_url}esearch.fcgi?db=pubmed&term={query}&retmax=5"
    search_response = requests.get(search_url)
    search_results = ET.fromstring(search_response.content)
    id_list = [id_.text for id_ in search_results.findall('.//IdList/Id')]
    details_url = f"{base_url}efetch.fcgi?db=pubmed&id={','.join(id_list)}&retmode=xml"
    details_response = requests.get(details_url)
    details_results = ET.fromstring(details_response.content)
    articles = []
    for article in details_results.findall('.//PubmedArticle'):
        title = article.find('.//ArticleTitle').text
        authors_list = article.findall('.//AuthorList/Author')
        authors = ', '.join([author.find('LastName').text + ' ' + author.find('ForeName').text for author in authors_list if author.find('LastName') is not None and author.find('ForeName') is not None])
        pub_date = article.find('.//PubDate/Year').text
        abstract = article.find('.//Abstract/AbstractText').text
        article_link = "https://pubmed.ncbi.nlm.nih.gov/" + article.find('.//PMID').text
        articles.append({
            "title": title,
            "authors": authors,
            "publication_date": pub_date,
            "abstract": abstract,
            "link": article_link
        })
    return articles
def fetch_article_content(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"Failed to fetch article content: HTTP status code {response.status_code}")
            return ""

        soup = BeautifulSoup(response.content, 'html.parser')

        # Locate the abstract section
        abstract_section = soup.find('div', class_='abstract')  # Adjust the class name as per the actual page structure

        if abstract_section is not None:
            abstract_text = abstract_section.text
            return abstract_text.strip()
        else:
            st.error("Failed to find the abstract in the page.")
            return ""
    except Exception as e:
        st.error(f"Failed to fetch article content: {e}")
        return ""
# Function to extract text content from a URL
def get_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all("p")
    content = "\n".join(paragraph.get_text() for paragraph in paragraphs)
    return content
def extract_text_from_pdf_miner(uploaded_file):
    try:
        text = extract_text(uploaded_file)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF with pdfminer.six: {e}")
        return ""
# Chatbot response function with retry logic
@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(2))
def get_chatbot_response(message):
    try:
        llm_response = st.session_state.conversation.run(message)
        return llm_response
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Initialize session state for chatbot
def initialize_session_state():
    if "history_cra" not in st.session_state:
        st.session_state.history_cra = []
    if "history_mw" not in st.session_state:
        st.session_state.history_mw = []
    if "selected_articles" not in st.session_state:
        st.session_state.selected_articles = []
    if "articles" not in st.session_state:
        st.session_state.articles = []
    if "webpage_content" not in st.session_state:
        st.session_state.webpage_content = ''
    if "pdf_content" not in st.session_state:
        st.session_state.pdf_content = ''
    if "conversation" not in st.session_state:
        #api_key = st.text_input("Enter your OpenAI API key")  # Input for API key
        if api_key:
            llm = OpenAI(
                temperature=0,
                model_name="gpt-3.5-turbo",
                api_key=api_key  # Use the entered API key
            )
            st.session_state.conversation = ConversationChain(
                llm=llm,
                memory=ConversationSummaryMemory(llm=llm)
            )
persona1, persona2 = load_personas()

# Streamlit UI

#api_key = st.text_input("Enter your OpenAI API key",type="password")
if api_key and fda_api_key:
    if validate_api_key(api_key) and validate_fda_api_key(fda_api_key):
        # Proceed with the rest of the app
        st.success("API Keys are valid.")
# User selection
        user_type = st.radio("Select your role:", ("Medical Writer", "Clinical Research Associate"))
        selected_persona = select_persona(user_type)
        if user_type == "Medical Writer":
            # Initialize session state for selected titles and contents
            if 'selected_titles' not in st.session_state:
                st.session_state.selected_titles = []
            if 'articles' not in st.session_state:
                st.session_state.articles = []
            if 'webpage_content' not in st.session_state:
                st.session_state.webpage_content = ''
            if 'pdf_content' not in st.session_state:
                st.session_state.pdf_content = ''

            # Option for content source
            content_source = st.radio("Select content source:", ("PubMed Library", "Insert URL", "Upload PDF"))

            # PubMed Library
            if content_source == "PubMed Library":
                query = st.text_input("Enter a therapeutic condition to search on PubMed:")
                if query and st.button("Fetch Articles"):
                    st.session_state.articles = fetch_pubmed_data(query)
                if st.session_state.articles:
                    article_titles = [article["title"] for article in st.session_state.articles]
                    selected_titles = st.multiselect("Select articles:", article_titles )#default=st.session_state.selected_titles)
                    if st.button("Confirm Selection"):
                        st.session_state.selected_titles = selected_titles
                        selected_articles_links = [article['link'] for article in st.session_state.articles if article['title'] in selected_titles]
                        st.session_state.selected_articles_content = [fetch_article_content(link) for link in selected_articles_links]
                        articles_context = "\n\n".join(st.session_state.selected_articles_content)
                        initial_context = "Based on the following context, start the discussion:\n\n" + articles_context
                        initial_bot_message = get_chatbot_response(initial_context)
                        if initial_bot_message:
                            st.session_state.history_mw.append("Bot: " + initial_bot_message.strip())
                            st.text_area("Conversation", value="\n".join(st.session_state.history_mw), height=300, key="chat_area_pubmed")

            # Insert URL
            elif content_source == "Insert URL":
                input_url = st.text_input("Enter the URL:")
                if st.button("Fetch Content from URL"):
                    webpage_content = get_text(input_url)
                    st.session_state.webpage_content = webpage_content
                    st.text_area("Extracted Content", webpage_content, height=150)
                    url_context = "Based on the following context, start the discussion:\n\n" + webpage_content
                    initial_bot_message = get_chatbot_response(url_context)
                    if initial_bot_message:
                        st.session_state.history_mw.append("Bot: " + initial_bot_message.strip())
                        st.text_area("Conversation", value="\n".join(st.session_state.history_mw), height=300, key="chat_area_webpage_content")

            # Upload PDF
            elif content_source == "Upload PDF":
                uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
                if uploaded_file is not None:
                    pdf_content = extract_text_from_pdf_miner(uploaded_file)
                    st.session_state.pdf_content = pdf_content
                    st.text_area("Extracted Content", pdf_content, height=150)
                    pdf_context = "Based on the following context, start the discussion:\n\n" + pdf_content
                    initial_bot_message = get_chatbot_response(pdf_context)
                    if initial_bot_message:
                        st.session_state.history_mw.append("Bot: " + initial_bot_message.strip())
                        st.text_area("Conversation", value="\n".join(st.session_state.history_mw), height=300, key="chat_area_pdf")
            # Chatbot interaction for user input
        elif user_type == "Clinical Research Associate":
            st.subheader("Search for Adverse Drug Events")
            drug_query = st.text_input("Enter the drug name")
            therapeutic_area_query = st.text_input("Enter the therapeutic area")

            if st.button("Fetch and Summarize Adverse Events"):
                response = fetch_adverse_events(fda_api_key,drug_query, therapeutic_area_query)  # Fetch data
                parsed_data = parse_openfda_response(response)  # Parse data
                drug_counts, indication_counts,reaction_counts = aggregate_data_for_summary(parsed_data)  # Aggregate data
                summary = generate_basic_summary(drug_counts, indication_counts,reaction_counts)  # Generate summary
                st.text(summary)  # Display summary
        user_input = st.text_input("Ask a question to the chatbot:", key="user_input")
        if st.button("Chat with Chatbot"):

            if user_type=="Medical Writer":
                current_history_mw = st.session_state.history_mw
                message = selected_persona['human']['chatname'] + ": " + user_input
                # Determine the context based on the content source
                full_context = ""
                if content_source == "PubMed Library" and st.session_state.selected_articles:
                    full_context = "\n\n".join([f"Article Title: {article['title']}\nAbstract: {article['abstract']}" for article in st.session_state.articles if article['title'] in st.session_state.selected_titles])
                elif content_source == "Insert URL":
                    full_context = st.session_state.webpage_content
                elif content_source == "Upload PDF":
                    full_context = st.session_state.pdf_content

                full_context = message + "\n\n" + full_context  # Append user input to the context

                # Get and display chatbot response
                bot_response = get_chatbot_response(full_context)
                if bot_response:
                    current_history_mw.append(message)  # User's question
                    current_history_mw.append("Bot: " + bot_response.strip())  # Bot's response
                    st.text_area("Conversation", value="\n".join(current_history_mw), height=300, key="chat_area")

                    st.session_state.history_mw = current_history_mw
            elif user_type == "Clinical Research Associate":
                current_history_cra = st.session_state.history_cra
                message = selected_persona['human']['chatname'] + ": " + user_input
                bot_response = get_chatbot_response(message)
                if bot_response:
                    current_history_cra.append(message)  # User's question
                    current_history_cra.append("Bot: " + bot_response.strip())  # Bot's response
                    st.text_area("Conversation", value="\n".join(current_history_cra), height=300, key="chat_area")

                    st.session_state.history_cra = current_history_cra  # Update conversation history

        # Display results and chatbot conversation
        st.write("Results and chatbot conversation will be displayed here.")

        # Initialize session state for chatbot
        initialize_session_state()
    else:
        st.error("Invalid API Key(s). Please enter valid keys to proceed.")
else:
    st.warning("Please enter the API keys")
