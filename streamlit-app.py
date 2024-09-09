import os
import instructor
import asyncio
import re
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from enum import Enum
from typing import List, Dict
import pandas as pd
import streamlit as st
import time

# Initialize OpenAI client
api_key = st.secrets["OPENAI_API_KEY"]
client = AsyncOpenAI(api_key=api_key)

# Print instructor file location for debugging
st.write(f"Instructor file location: {instructor.__file__}")

# Updated pricing based on input and output tokens -price for 1000 tokens
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.00060},
    "gpt-4o-mini-2024-07-18": {"input": 0.00015, "output": 0.00060},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-2024-08-06": {"input": 0.0025, "output": 0.01},
    "gpt-4o-2024-05-13": {"input": 0.005, "output": 0.015}
}

def calculate_cost(total_prompt_tokens, total_completion_tokens, model_name):
    input_cost_rate = MODEL_PRICING[model_name]["input"]
    output_cost_rate = MODEL_PRICING[model_name]["output"]

    prompt_cost = (total_prompt_tokens / 1000) * input_cost_rate
    completion_cost = (total_completion_tokens / 1000) * output_cost_rate
    total_cost = prompt_cost + completion_cost

    return {
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "total_cost": total_cost
    }

# Define Pydantic data models
class KeywordClassification(BaseModel):
    category: str
    subcategory: str
    explanation: str = Field(description="Brief explanation for the classification")

# System prompt for keyword classification
SYSTEM_PROMPT = """
You are an AI assistant for an SEO team.
Your role is to analyze keywords and classify them into appropriate categories and subcategories to help our team organize and prioritize their SEO efforts.
Your tasks:
1. Categorize the keyword into the most appropriate category from the provided list.
2. Assign the keyword to the most appropriate subcategory within the chosen category.
3. Give a brief explanation for your classification choice.
Respond in the following format:
Category: [chosen category]
Subcategory: [chosen subcategory]
Explanation: [brief explanation]
Remember:
- Be objective and base your analysis on the keyword provided and the client information given.
- The explanation should be concise but informative, highlighting why the keyword fits the chosen category and subcategory.
- Consider the client's industry, target audience, and business goals in your classification.
"""

# Load categories and subcategories from the uploaded CSV file
def load_categories(file):
    categories_df = pd.read_csv(file)
    categories = {}
    for index, row in categories_df.iterrows():
        category = row['Categ']
        subcategory = row['Sous_categ']
        if category not in categories:
            categories[category] = []
        if subcategory not in categories[category]:
            categories[category].append(subcategory)
    return categories

# Classification logic using OpenAI model
async def classify_keyword(keyword: str, categories: Dict[str, List[str]], client_info: str, model_name: str) -> tuple:
    try:
        categories_str = "\n".join([f"{cat}: {', '.join(subcats)}" for cat, subcats in categories.items()])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Client Information:\n{client_info}\n\nKeyword: {keyword}\nCategories and Subcategories:\n{categories_str}"}
        ]

        response = await client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        content = response.choices[0].message.content

        # Extract usage information
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        # Use regex to extract category, subcategory, and explanation
        category_match = re.search(r'Category: (.+)', content)
        subcategory_match = re.search(r'Subcategory: (.+)', content)
        explanation_match = re.search(r'Explanation: (.+)', content)

        if not all([category_match, subcategory_match, explanation_match]):
            raise ValueError("Could not parse the AI response correctly")

        category = category_match.group(1).strip()
        subcategory = subcategory_match.group(1).strip()
        explanation = explanation_match.group(1).strip()

        classification = KeywordClassification(category=category, subcategory=subcategory, explanation=explanation)
        return classification, prompt_tokens, completion_tokens, total_tokens
    except Exception as e:
        st.error(f"Error classifying keyword '{keyword}': {str(e)}")
        return None, 0, 0, 0

async def process_keywords(keywords, categories, client_info, model_name, progress_bar, progress_text):
    results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    for i, keyword in enumerate(keywords, 1):
        classification, prompt_tokens, completion_tokens, tokens = await classify_keyword(keyword, categories, client_info, model_name)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_tokens += tokens
        if classification:
            results.append({
                'keyword': keyword,
                'category': classification.category,
                'subcategory': classification.subcategory,
                'explanation': classification.explanation
            })

        # Update progress
        progress_bar.progress(i / len(keywords))
        progress_text.text(f"Processed {i}/{len(keywords)} keywords")

    return results, total_prompt_tokens, total_completion_tokens, total_tokens

def main():
    st.title("SEO Keyword Classification")

    # Model selection
    model_name = st.selectbox(
        "Select OpenAI model",
        options=[
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-05-13"
        ],
        index=0,
        help="Choose the OpenAI model to use for classification. Different models have different costs and capabilities."
    )

    # Client Information Input
    st.subheader("Client Information")
    client_info = st.text_area("Enter important information about the client (e.g., industry, target audience, business goals):",
                               help="This information will help guide the AI in classifying the keywords.")

    # Category Input Option
    category_option = st.radio(
        "How would you like to input categories and subcategories?",
        ("Upload CSV", "Enter Manually")
    )

    categories = {}

    if category_option == "Upload CSV":
        # File upload for Categories and Subcategories CSV
        st.subheader("Upload Categories and Subcategories CSV")
        categories_file = st.file_uploader("Choose a CSV file for Categories and Subcategories", type="csv")

        if categories_file is not None:
            categories = load_categories(categories_file)
            st.write("Categories and subcategories loaded successfully!")

            # Display the loaded categories and subcategories
            st.subheader("Loaded Categories and Subcategories")
            for category, subcategories in categories.items():
                st.write(f"**{category}**")
                for subcategory in subcategories:
                    st.write(f"- {subcategory}")
                st.write("")  # Add a blank line between categories
    else:
        # Manual input for categories and subcategories
        st.subheader("Enter Categories and Subcategories")
        num_categories = st.number_input("Number of categories", min_value=1, value=3)
        for i in range(num_categories):
            cat = st.text_input(f"Category {i+1}")
            if cat:
                subcats = st.text_input(f"Subcategories for {cat} (comma-separated)")
                categories[cat] = [subcat.strip() for subcat in subcats.split(',') if subcat.strip()]

    # File upload for Keywords CSV
    st.subheader("Upload Keywords CSV")
    keywords_file = st.file_uploader("Choose a CSV file for Keywords", type="csv")

    if keywords_file is not None and categories:
        df = pd.read_csv(keywords_file)
        num_keywords = len(df)  # Get the total number of keywords
        st.write(f"Uploaded file contains {num_keywords} keywords")

        # Dynamically set max sample size based on the number of keywords in the uploaded file
        sample_size = st.slider("Select sample size", min_value=1, max_value=num_keywords, value=min(10, num_keywords))

        # Select a sample of the keywords based on the sample size
        sample_df = df.sample(n=sample_size, random_state=42)
        keywords = sample_df['keywords'].tolist()

        if st.button("Classify Keywords") and categories and client_info:
            st.write(f"Processing {len(keywords)} keywords using {model_name}")
            st.write(f"This will require approximately {len(keywords)} API calls")

            progress_bar = st.progress(0)
            progress_text = st.empty()

            start_time = time.time()
            results, total_prompt_tokens, total_completion_tokens, total_tokens = asyncio.run(process_keywords(keywords, categories, client_info, model_name, progress_bar, progress_text))
            end_time = time.time()

            # Create results dataframe
            results_df = pd.DataFrame(results)

            # Display results
            st.write("\nClassification Results:")
            st.dataframe(results_df)

            # Display summary
            st.write("\nClassification Summary:")
            st.write(results_df.groupby(['category', 'subcategory']).size().unstack(fill_value=0))

            # Calculate and display cost
            cost_breakdown = calculate_cost(total_prompt_tokens, total_completion_tokens, model_name)
            st.write(f"\nTotal prompt tokens: {total_prompt_tokens}")
            st.write(f"Total completion tokens: {total_completion_tokens}")
            st.write(f"Total tokens: {total_tokens}")
            st.write(f"Prompt tokens cost: ${cost_breakdown['prompt_cost']:.6f}")
            st.write(f"Completion tokens cost: ${cost_breakdown['completion_cost']:.6f}")
            st.write(f"Estimated total cost for all {len(keywords)} API calls: ${cost_breakdown['total_cost']:.6f}")

            st.write(f"\nTotal API calls made: {len(results)}")
            st.write(f"Total processing time: {end_time - start_time:.2f} seconds")

            # Option to download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="keyword_classification_results.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()
