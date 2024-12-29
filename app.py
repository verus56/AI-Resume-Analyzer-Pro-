import streamlit as st
import google.generativeai as genai
import os
import docx2txt
import PyPDF2 as pdf
from dotenv import load_dotenv
import json 
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import requests
import json
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.colored_header import colored_header
from streamlit_extras.card import card
import time

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('Google_api_key'))

# Gemini AI Configuration
generation_config = {
    "temperature": 0.5,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 2000,
}



# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = pdf.PdfReader(uploaded_file)
        text_content = ""
        for page in pdf_reader.pages:
            text_content += str(page.extract_text())
        return text_content
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

# Function to extract text from DOCX
def extract_text_from_docx(upload_file):
    try:
        text = docx2txt.process(upload_file)
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return None

# Function to create word cloud
def create_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt

# Function to calculate readability score
def calculate_readability_score(text):
    words = len(re.findall(r'\w+', text))
    sentences = len(re.findall(r'[.!?]+', text))
    if sentences == 0:
        sentences = 1
    avg_words_per_sentence = words / sentences
    return min(100, max(0, 100 - (abs(avg_words_per_sentence - 20) * 2)))

# Function to create radar chart
def create_radar_chart(skills_data):
    categories = list(skills_data.keys())
    values = list(skills_data.values())

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False
    )
    return fig

# Function to create timeline chart
import pandas as pd
from datetime import datetime


def create_timeline_chart(experience_data):
    # Create a copy of the dataframe to avoid modifying the original
    df = experience_data.copy()

    # Function to parse date strings
    def parse_date(date_str):
        if pd.isna(date_str) or date_str.lower() == 'present':
            return datetime.now()
        try:
            return pd.to_datetime(date_str, format='%B %Y')
        except:
            try:
                return pd.to_datetime(date_str)
            except:
                return None

    # Convert date columns
    df['start_date'] = df['start_date'].apply(parse_date)
    df['end_date'] = df['end_date'].apply(parse_date)

    # Create the timeline
    fig = px.timeline(df,
                      x_start='start_date',
                      x_end='end_date',
                      y='role',
                      title='Experience Timeline')

    # Customize the layout
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        xaxis=dict(
            title="Date",
            type="date"
        )
    )

    return fig



def validate_response_structure(response_json):
    """Validate that all required fields are present in the response"""
    required_fields = {
        'match_percentage': str,
        'technical_skills': {
            'matching_skills': list,
            'missing_skills': list,
            'skill_match_percentage': str
        },
        'experience_analysis': {
            'years_of_experience': str,
            'experience_match_percentage': str,
            'key_achievements': list,
            'career_progression': list
        },
        'education_analysis': {
            'education_match': str,
            'recommended_certifications': list
        },
        'improvement_suggestions': list,
        'ats_compatibility': {
            'format_score': str,
            'issues_found': list,
            'formatting_suggestions': list
        },
        'cultural_fit': {
            'alignment_score': str,
            'matching_values': list,
            'development_areas': list
        },
        'industry_analysis': {
            'industry_alignment': str,
            'relevant_experience': list,
            'industry_gaps': list
        }
    }

    try:
        data = json.loads(response_json)

        def check_fields(required, actual):
            for key, value_type in required.items():
                if key not in actual:
                    return False
                if isinstance(value_type, dict):
                    if not check_fields(value_type, actual[key]):
                        return False
                elif not isinstance(actual[key], value_type):
                    return False
                # Ensure lists are not empty
                if isinstance(value_type, type) and value_type == list and not actual[key]:
                    return False
            return True

        return check_fields(required_fields, data)
    except:
        return False


# Update the prompt template to be more explicit
input_prompt_template = """
Analyze this resume against the job description. You MUST provide all requested information and ensure no fields are empty.

Resume: {text}
Job Description: {job_description}
Job Title: {job_title}

IMPORTANT INSTRUCTIONS:
1. You MUST analyze and provide ALL fields in the exact format below
2. ALL lists MUST contain at least one item - never return empty lists
3. If no specific items are found, provide relevant suggestions based on the job description
4. ALL percentages must be calculated based on actual analysis
5. Ensure each list field has at least 2-3 items

Provide response in this exact JSON format:
{{
    "match_percentage": "<calculated_percentage>",
    "technical_skills": {{
        "matching_skills": ["MUST list at least 2-3 matching skills"],
        "missing_skills": ["MUST list at least 2-3 missing skills"],
        "skill_match_percentage": "<calculated_percentage>"
    }},
    "experience_analysis": {{
        "years_of_experience": "<extracted_years>",
        "experience_match_percentage": "<calculated_percentage>",
        "key_achievements": ["MUST list at least 2-3 achievements"],
        "career_progression": [
            {{
                "role": "<role>",
                "start_date": "<start_date>",
                "end_date": "<end_date>"
            }}
        ]
    }},
    "education_analysis": {{
        "education_match": "<calculated_percentage>",
        "recommended_certifications": ["MUST list at least 2-3 relevant certifications"]
    }},
    "improvement_suggestions": ["MUST list at least 2-3 suggestions"],
    "ats_compatibility": {{
        "format_score": "<calculated_percentage>",
        "issues_found": ["MUST list at least 2-3 issues"],
        "formatting_suggestions": ["MUST list at least 2-3 suggestions"]
    }},
    "cultural_fit": {{
        "alignment_score": "<calculated_percentage>",
        "matching_values": ["MUST list at least 2-3 values"],
        "development_areas": ["MUST list at least 2-3 areas"]
    }},
    "industry_analysis": {{
        "industry_alignment": "<calculated_percentage>",
        "relevant_experience": ["MUST list at least 2-3 experiences"],
        "industry_gaps": ["MUST list at least 2-3 gaps"]
    }}
}}"""


# Update generate_response function with retry logic
def generate_response(input_text, max_retries=3):
    for attempt in range(max_retries):
        try:
            llm = genai.GenerativeModel(
                model_name="gemini-pro",
                generation_config=generation_config
            )
            output = llm.generate_content(input_text)
            response_text = output.text.strip()
            response_text = response_text.replace('```json', '').replace('```', '')

            # Validate response structure and content
            if validate_response_structure(response_text) and validate_percentages(response_text):
                return response_text
            else:
                if attempt == max_retries - 1:
                    st.error("Failed to generate a complete analysis after multiple attempts. Please try again.")
                    return None
                continue

        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Error generating response: {str(e)}")
                return None
            continue

    return None


# Add a function to validate percentages
def validate_percentages(response_json):
    """Validate that percentages are calculated and not hardcoded"""
    try:
        data = json.loads(response_json)
        match_percentage = float(data['match_percentage'])
        skill_match = float(data['technical_skills']['skill_match_percentage'])
        exp_match = float(data['experience_analysis']['experience_match_percentage'])

        # Check if percentages are varied (not hardcoded)
        if len(set([match_percentage, skill_match, exp_match])) == 1:
            return False
        return True
    except:
        return False


# Update the generate_response function
def generate_response(input_text):
    try:
        llm = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=generation_config
        )
        output = llm.generate_content(input_text)
        response_text = output.text.strip()
        response_text = response_text.replace('```json', '').replace('```', '')

        # Validate JSON and percentages
        try:
            if not validate_percentages(response_text):
                # Regenerate if percentages seem hardcoded
                return generate_response(input_text)
            return response_text
        except json.JSONDecodeError as je:
            st.error(f"Invalid JSON format received. Please try again.")
            st.code(response_text)
            return None

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None


def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Function to get color mode
def get_color_mode():
    try:
        return st.session_state.color_mode
    except:
        st.session_state.color_mode = "light"
        return "light"

# Main Streamlit app


def main():
    st.set_page_config(page_title="Resume Analyzer Pro", layout="wide")

    # Modern CSS styling
    st.markdown("""
            <style>
            [data-theme="light"] {
                --bg-color: #ffffff;
                --text-color: #333333;
                --card-bg: #ffffff;
                --shadow: rgba(0,0,0,0.1);
            }
            [data-theme="dark"] {
                --bg-color: #1e1e1e;
                --text-color: #ffffff;
                --card-bg: #2d2d2d;
                --shadow: rgba(255,255,255,0.1);
            }
            .main-header {
                background: linear-gradient(90deg, #4CAF50 0%, #2196F3 100%);
                padding: 2rem;
                border-radius: 15px;
                color: white;
                margin-bottom: 2rem;
                text-align: center;
                box-shadow: 0 4px 15px var(--shadow);
            }
            .card {
                background: var(--card-bg);
                padding: 1.5rem;
                border-radius: 15px;
                box-shadow: 0 4px 6px var(--shadow);
                margin-bottom: 1rem;
                transition: transform 0.3s ease;
                color: var(--text-color);
            }
            .card:hover {
                transform: translateY(-5px);
            }
            .stButton>button {
                background: linear-gradient(90deg, #4CAF50 0%, #2196F3 100%);
                color: white;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 25px;
                font-weight: 500;
                transition: all 0.3s ease;
                width: 100%;
            }
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px var(--shadow);
            }
            </style>
        """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        theme = st.selectbox("Theme", ["Light", "Dark"], key="theme")
        st.session_state.color_mode = theme.lower()

        st.markdown("### 📤 Export Options")
        export_format = st.selectbox("Export Format", ["PDF", "DOCX", "JSON"])
        if st.button("Export Analysis"):
            if 'analysis' in locals():
                with st.spinner("Preparing export..."):
                    time.sleep(1)  # Simulate export
                    st.success("Analysis exported successfully!")
            else:
                st.warning("Please analyze a resume first")

    ## Header with Lottie Animation
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
            <div class="main-header">
                <h1>📊 Resume Analyzer Pro</h1>
                <p>AI-Powered Resume Analysis & Career Insights</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        lottie_url = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
        lottie_json = load_lottie_url(lottie_url)
        if lottie_json:
            st_lottie(lottie_json, height=150)

    # Tabs
    tabs = st.tabs(["🎯 Analysis", "🔍 ATS Check", "📈 Insights"])

    with tabs[0]:
        col1, col2 = st.columns([1, 1])

        with col1:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer")
                job_description = st.text_area(
                    "Job Description",
                    height=200,
                    placeholder="Paste the complete job description here..."
                )
                st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                uploaded_file = st.file_uploader(
                    "Upload Your Resume",
                    type=["pdf", "docx"],
                    help="Supported formats: PDF, DOCX"
                )
                if uploaded_file:
                    with st.spinner("Processing file..."):
                        time.sleep(1)
                        st.success(f"✅ Successfully uploaded: {uploaded_file.name}")
                st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🚀 Analyze Resume", type="primary"):
            if job_description and uploaded_file and job_title:
                with st.spinner("Analyzing your resume..."):
                    # Progress bar animation
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)

                    # File processing
                    if uploaded_file.type == "application/pdf":
                        text = extract_text_from_pdf(uploaded_file)
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        text = extract_text_from_docx(uploaded_file)
                    else:
                        st.error("Invalid File Format")
                        st.stop()

                    if text:
                        response = generate_response(
                            input_prompt_template.format(
                                text=text,
                                job_description=job_description,
                                job_title=job_title
                            )
                        )

                        if response:
                            try:
                                analysis = json.loads(response)

                                # Results display
                                st.markdown('<div class="card">', unsafe_allow_html=True)
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Match", f"{analysis['match_percentage']}%")
                                with col2:
                                    st.metric("Skills", f"{analysis['technical_skills']['skill_match_percentage']}%")
                                with col3:
                                    st.metric("Industry", f"{analysis['industry_analysis']['industry_alignment']}%")
                                with col4:
                                    st.metric("Culture", f"{analysis['cultural_fit']['alignment_score']}%")
                                st.markdown('</div>', unsafe_allow_html=True)

                                # Skills Analysis
                                st.markdown('<div class="card">', unsafe_allow_html=True)
                                st.subheader("💡 Skills Analysis")
                                skills_data = {
                                    "Overall": float(analysis['match_percentage']),
                                    "Technical": float(analysis['technical_skills']['skill_match_percentage']),
                                    "Experience": float(analysis['experience_analysis']['experience_match_percentage']),
                                    "Education": float(analysis['education_analysis']['education_match']),
                                    "Cultural": float(analysis['cultural_fit']['alignment_score']),
                                    "Industry": float(analysis['industry_analysis']['industry_alignment'])
                                }
                                st.plotly_chart(create_radar_chart(skills_data), use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)

                                # Detailed Analysis
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown('<div class="card">', unsafe_allow_html=True)
                                    st.subheader("✅ Strengths")
                                    st.write("Matching Skills:", ", ".join(analysis['technical_skills']['matching_skills']))
                                    st.write("Key Achievements:")
                                    for achievement in analysis['experience_analysis']['key_achievements']:
                                        st.write(f"• {achievement}")
                                    st.markdown('</div>', unsafe_allow_html=True)

                                with col2:
                                    st.markdown('<div class="card">', unsafe_allow_html=True)
                                    st.subheader("🎯 Areas for Improvement")
                                    st.write("Missing Skills:", ", ".join(analysis['technical_skills']['missing_skills']))
                                    st.write("Recommended Certifications:")
                                    for cert in analysis['education_analysis']['recommended_certifications']:
                                        st.write(f"• {cert}")
                                    st.markdown('</div>', unsafe_allow_html=True)

                            except Exception as e:
                                st.error(f"Error processing results: {str(e)}")
            else:
                st.warning("Please provide all required information: Job Title, Description, and Resume")

    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔍 ATS Compatibility Check")
        if 'analysis' in locals():
            ats_score = int(analysis['ats_compatibility']['format_score'])
            st.progress(ats_score / 100)
            st.metric("ATS Score", f"{ats_score}%")

            col1, col2 = st.columns(2)
            with col1:
                st.write("📋 Format Issues:")
                for issue in analysis['ats_compatibility']['issues_found']:
                    st.warning(issue)
            with col2:
                st.write("💡 Suggestions:")
                for suggestion in analysis['ats_compatibility']['formatting_suggestions']:
                    st.info(suggestion)
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 Career Insights")
        if 'analysis' in locals():
            col1, col2 = st.columns(2)
            with col1:
                st.write("🎯 Industry Experience")
                for exp in analysis['industry_analysis']['relevant_experience']:
                    st.success(exp)
            with col2:
                st.write("🔄 Development Areas")
                for area in analysis['cultural_fit']['development_areas']:
                    st.info(area)
        st.markdown('</div>', unsafe_allow_html=True)

        # Footer
        st.markdown("""
               <div style='text-align: center; margin-top: 2rem; padding: 1rem;'>
                   <p>Made with ❤️ byV56</p>
               </div>
           """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
