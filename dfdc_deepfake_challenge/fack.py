import streamlit as st
import requests

# Function to check the news URL with Google Fact Check API
def check_fact_with_google_factcheck(news_url: str, api_key: str):
    endpoint = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "key": api_key,
        "query": news_url,  # Change from 'url' to 'query'
    }

    try:
        response = requests.get(endpoint, params=params)
        
        # Check if the API key is invalid by looking at the status code
        if response.status_code == 403 or response.status_code == 401:
            return "❌ Invalid API key or insufficient permissions.", []

        # Check for other potential errors
        if response.status_code != 200:
            return f"❌ Error: Received unexpected status code {response.status_code}", []

        data = response.json()

        if "claims" not in data:
            return "✅ No fact checks found for this URL. It might be unverified but not flagged.", data

        results = []
        for claim in data["claims"]:
            text = claim.get("text", "No claim text.")
            claimant = claim.get("claimant", "Unknown source")
            rating = claim.get("claimReview", [{}])[0].get("textualRating", "No rating")
            review_url = claim.get("claimReview", [{}])[0].get("url", "")

            results.append({
                "claim": text,
                "claimant": claimant,
                "rating": rating,
                "review_url": review_url
            })

        return "⚠️ Fact checks found for this URL.", results

    except Exception as e:
        return f"❌ Error occurred: {str(e)}", []

# Streamlit UI
st.set_page_config(page_title="Google Fact Checker", page_icon="🔍")
st.title("🔍 Google News Fact Checker")

api_key = st.text_input("Enter your Google Fact Check API Key", type="password")
news_url = st.text_input("Enter the News Article URL")

if st.button("Check News Validity"):
    if not api_key or not news_url:
        st.warning("Please provide both the API key and news URL.")
    else:
        status, checks = check_fact_with_google_factcheck(news_url, api_key)
        st.markdown(f"### {status}")

        # Show the fact check results, if any
        for idx, c in enumerate(checks):
            with st.expander(f"🧾 Claim {idx + 1}"):
                st.markdown(f"**Claim:** {c['claim']}")
                st.markdown(f"**Claimant:** {c['claimant']}")
                st.markdown(f"**Rating:** `{c['rating']}`")
                st.markdown(f"[🔗 View Full Review]({c['review_url']})")
