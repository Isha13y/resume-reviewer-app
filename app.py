import streamlit as st
import os
from dotenv import load_dotenv
import PyPDF2
import pdfplumber
from docx import Document
import io
import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
import time
import random

# LangChain imports
try:
    from langchain_anthropic import ChatAnthropic
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Load environment variables
load_dotenv('config.env')

# Configure page
st.set_page_config(
    page_title="LLM Resume Reviewer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class ResumeFeedback:
    """Structure for resume feedback"""
    overall_score: int
    strengths: List[str]
    weaknesses: List[str]
    missing_keywords: List[str]
    improvement_suggestions: List[str]
    section_feedback: Dict[str, str]
    improved_resume: Optional[str] = None

class ResumeAnalyzer:
    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-20250514", optimize_for_free_tier: bool = False):
        """
        Initialize the Resume Analyzer with LangChain Claude client.
        
        Args:
            api_key (str): Claude API key (from UI or env file)
            model (str): Claude model to use
            optimize_for_free_tier (bool): Use optimizations (less relevant for Claude)
        """
        # Use provided API key or fall back to environment variable
        self.api_key = api_key or os.getenv('RAKUTEN_AI_GATEWAY_KEY')
        self.model = model
        self.optimize_for_free_tier = optimize_for_free_tier
        
        # Set token limits - Claude has higher limits than OpenAI free tier
        if self.optimize_for_free_tier:
            self.max_tokens_analysis = 2000  # Claude is more generous
            self.max_tokens_improvement = 1500
            self.temperature = 0.1
        else:
            self.max_tokens_analysis = 4000  # Claude can handle more tokens
            self.max_tokens_improvement = 3000
            self.temperature = 0.1  # Claude works well with low temperature
        
        # Validate API key
        if not self.api_key or len(self.api_key.strip()) < 10:
            st.error("⚠️ Claude API key not provided!")
            st.info("📝 Please:\n1. Enter your Claude API key in the sidebar, OR\n2. Configure it in config.env file\n3. Ensure your API key is valid")
            st.stop()
        
        # Initialize LangChain Claude client
        try:
            self.llm = ChatAnthropic(
                model_name=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens_analysis,
                # anthropic_api_url="your_api_url_here",  # Configure your API endpoint
                anthropic_api_key="test",  # Required placeholder for setup
                default_headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=60.0,
                max_retries=3
            )
            
                
        except Exception as e:
            st.error(f"❌ Failed to initialize Claude client: {str(e)}")
            st.info("💡 Please check:\n- Claude API key is valid\n- You have access to Claude models\n- Internet connection is stable")
            st.stop()
    
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file"""
        try:
            # Try with pdfplumber first (better formatting)
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
                if text.strip():
                    return text
        except:
            pass
        
        try:
            # Fallback to PyPDF2
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting PDF text: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, file_content: bytes) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(io.BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting DOCX text: {str(e)}")
            return ""
    
    def _make_api_call_with_retry(self, messages, max_tokens=None):
        """
        Make API call with exponential backoff retry logic for rate limiting.
        """
        max_retries = 3  # Claude API is usually more stable
        base_delay = 1
        max_delay = 30
        
        for attempt in range(max_retries):
            try:
                # Update max_tokens for this specific call
                if max_tokens:
                    self.llm.max_tokens = max_tokens
                
                # Add small random delay to avoid thundering herd
                if attempt > 0:
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    st.info(f"⏳ API rate limited. Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(delay)
                
                response = self.llm.invoke(messages)
                return response.content
                
            except Exception as e:
                error_str = str(e).lower()
                
                if "rate limit" in error_str or "429" in error_str or "too many requests" in error_str:
                    if attempt == max_retries - 1:
                        st.error("❌ Claude API rate limit exceeded after all retries")
                        raise Exception(f"API rate limit exceeded: {str(e)}")
                    continue
                elif "unauthorized" in error_str or "403" in error_str or "401" in error_str:
                    st.error("🔑 Claude API authentication issue")
                    raise Exception(f"Authentication error: {str(e)}")
                elif "overloaded" in error_str or "503" in error_str:
                    if attempt == max_retries - 1:
                        st.error("⚠️ Claude API is temporarily overloaded")
                        raise Exception(f"API overloaded: {str(e)}")
                    continue
                else:
                    # Other errors, don't retry
                    raise e
        
        raise Exception("All retry attempts failed")
    
    def analyze_resume(self, resume_text: str, job_role: str, job_description: str = "") -> ResumeFeedback:
        """Analyze resume using Claude AI"""
        
        # Claude can handle longer content, but still truncate if extremely long
        if len(resume_text) > 8000:  # Claude has higher limits
            resume_text = resume_text[:8000] + "... [truncated for processing]"
            st.warning("📝 Resume was truncated due to length")
        
        job_context = f"Role: {job_role}"
        if job_description and len(job_description) < 1000:  # Claude can handle longer job descriptions
            job_context += f"\nJob Description: {job_description[:1000]}"
        
        # Claude works well with detailed prompts
        system_prompt = """You are an expert HR professional and career counselor with 15+ years of experience in resume analysis and recruitment. 
        Analyze resumes with focus on ATS compatibility, industry standards, and job-specific requirements. Always return valid JSON."""
        
        user_prompt = f"""
        Analyze the following resume for the specified job role and provide comprehensive, actionable feedback.

        {job_context}
        
        RESUME TO ANALYZE:
        {resume_text}

        Please provide a detailed analysis in the following JSON format:
        {{
            "overall_score": <score from 1-100>,
            "strengths": [<list of key strengths>],
            "weaknesses": [<list of areas needing improvement>],
            "missing_keywords": [<important keywords/skills missing for this role>],
            "improvement_suggestions": [<specific actionable suggestions>],
            "section_feedback": {{
                "contact_info": "<feedback on contact information>",
                "summary": "<feedback on professional summary/objective>",
                "experience": "<feedback on work experience section>",
                "education": "<feedback on education section>",
                "skills": "<feedback on skills section>",
                "formatting": "<feedback on overall formatting and structure>"
            }}
        }}

        Focus on:
        1. Relevance to the {job_role} position
        2. ATS (Applicant Tracking System) compatibility
        3. Quantified achievements and impact
        4. Proper formatting and structure
        5. Industry-specific keywords and skills
        6. Professional tone and clarity
        
        Return ONLY the JSON response, no additional text.
        """
        
        try:
            # Use retry logic with appropriate token limit
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            content = self._make_api_call_with_retry(messages, self.max_tokens_analysis)
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                feedback_data = json.loads(json_str)
            else:
                feedback_data = json.loads(content)
            
            return ResumeFeedback(
                overall_score=feedback_data.get('overall_score', 0),
                strengths=feedback_data.get('strengths', []),
                weaknesses=feedback_data.get('weaknesses', []),
                missing_keywords=feedback_data.get('missing_keywords', []),
                improvement_suggestions=feedback_data.get('improvement_suggestions', []),
                section_feedback=feedback_data.get('section_feedback', {})
            )
            
        except json.JSONDecodeError as e:
            st.error(f"❌ Error parsing Claude response: {str(e)}")
            return self._create_fallback_feedback("JSON parsing failed")
        except Exception as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower() or "429" in error_msg.lower():
                st.error(f"❌ Claude API rate limit issue: {error_msg}")
                st.info("💡 Try again in a few minutes, Claude API limits are usually temporary")
            elif "unauthorized" in error_msg.lower() or "401" in error_msg.lower() or "403" in error_msg.lower():
                st.error(f"❌ Claude API authentication issue: {error_msg}")
                st.info("💡 Please check your Claude API key in the sidebar")
            else:
                st.error(f"❌ Error analyzing resume: {error_msg}")
            return self._create_fallback_feedback(f"Analysis failed: {error_msg}")
    
    def _create_fallback_feedback(self, error_message: str) -> ResumeFeedback:
        """Create fallback feedback when analysis fails"""
        return ResumeFeedback(
            overall_score=0,
            strengths=["Unable to analyze due to technical issues"],
            weaknesses=[error_message],
            missing_keywords=[],
            improvement_suggestions=["Please try again or check your API configuration"],
            section_feedback={}
        )
    
    def generate_improved_resume(self, original_resume: str, feedback: ResumeFeedback, job_role: str) -> str:
        """Generate an improved version of the resume using Claude AI"""
        
        # Claude can handle longer content than OpenAI free tier
        if len(original_resume) > 6000:
            original_resume = original_resume[:6000] + "... [truncated]"
        
        # Claude works well with detailed instructions
        system_prompt = f"""You are an expert resume writer specializing in {job_role} positions. 
        Your task is to improve resumes while maintaining factual accuracy and enhancing ATS compatibility."""
        
        user_prompt = f"""
        Based on the following feedback, please rewrite and improve this resume for a {job_role} position.
        Focus on the improvement suggestions and address the identified weaknesses.

        ORIGINAL RESUME:
        {original_resume}

        FEEDBACK TO ADDRESS:
        - Weaknesses: {', '.join(feedback.weaknesses)}
        - Missing Keywords: {', '.join(feedback.missing_keywords)}
        - Improvement Suggestions: {', '.join(feedback.improvement_suggestions)}

        Please provide an improved version that:
        1. Maintains the original information's accuracy
        2. Improves formatting and structure
        3. Incorporates relevant keywords naturally
        4. Quantifies achievements where possible
        5. Uses strong action verbs
        6. Is ATS-friendly
        7. Follows industry best practices for {job_role} positions
        8. Uses clear, professional language
        9. Optimizes for both human readers and ATS systems
        
        Return only the improved resume text, no additional commentary or explanations.
        """
        
        try:
            # Use retry logic with appropriate token limit for resume improvement
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            content = self._make_api_call_with_retry(messages, self.max_tokens_improvement)
            return content
            
        except Exception as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower() or "429" in error_msg.lower():
                st.error(f"❌ Claude API rate limit exceeded: {error_msg}")
                st.info("💡 Try again in a few minutes, Claude rate limits are usually temporary")
                return f"Unable to generate improved resume due to API rate limits.\n\nError: {error_msg}\n\nPlease try again in a few minutes."
            elif "unauthorized" in error_msg.lower() or "401" in error_msg.lower() or "403" in error_msg.lower():
                st.error(f"❌ Claude API authentication issue: {error_msg}")
                st.info("💡 Please check your Claude API key")
                return f"Unable to generate improved resume due to authentication issues.\n\nError: {error_msg}\n\nPlease check your Claude API key in the sidebar."
            else:
                st.error(f"❌ Error generating improved resume: {error_msg}")
                return f"Unable to generate improved resume due to technical issues.\n\nError: {error_msg}\n\nPlease try again or check your configuration."

def main():
    st.title("📄 LLM-Powered Resume Reviewer")
    st.markdown("---")
    
    
    # Check if LangChain Anthropic is available
    if not LANGCHAIN_AVAILABLE:
        st.error("📦 LangChain Anthropic is required but not available. Please install langchain-anthropic.")
        st.code("pip install langchain-anthropic")
        st.stop()
    
    # Get API key from environment or use default configuration
    env_api_key = os.getenv('RAKUTEN_AI_GATEWAY_KEY')
    api_key = env_api_key if env_api_key and len(env_api_key.strip()) > 10 else None
    
    # Set default configuration values
    optimize_for_free_tier = False
    generate_improved = True
    detailed_feedback = True
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Job role input
        st.subheader("🎯 Target Job Role")
        job_role = st.text_input(
            "Enter target job role/position:",
            placeholder="e.g., Data Scientist, Product Manager, Software Engineer",
            help="Specify the job role you're targeting for tailored feedback"
        )
        
        # Model selection
        st.subheader("🤖 Claude Model")
        model_choice = st.selectbox(
            "Select Claude Model:",
            options=[
                "claude-sonnet-4-20250514",
                "claude-3-5-sonnet-20241022", 
                "claude-3-7-sonnet-20250219"
            ],
            index=0,
            help="Different Claude models with varying capabilities"
        )
        
        # Job description input
        st.subheader("📋 Job Description (Optional)")
        job_description = st.text_area(
            "Paste job description:",
            placeholder="Paste the job description here for more targeted feedback...",
            height=200,
            help="Adding a job description will provide more specific, targeted feedback"
        )
        
        if not job_description:
            job_description = ""
    
    # Main content area
    st.header("📤 Resume Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Upload Resume File")
        uploaded_file = st.file_uploader(
            "Choose your resume file",
            type=['pdf', 'txt', 'docx'],
            help="Supported formats: PDF, TXT, DOCX (Max 10MB)"
        )
    
    with col2:
        st.subheader("📝 Or Paste Resume Text")
        resume_text_input = st.text_area(
            "Paste your resume text here:",
            height=300,
            placeholder="Paste your resume content here as an alternative to file upload..."
        )
    
    # Check for job role requirement
    if not job_role:
        st.warning("⚠️ Please enter a target job role in the sidebar to get started.")
        return
    
    
    # Validate API key before proceeding
    if not api_key or len(api_key.strip()) < 10:
        st.error("⚠️ No Claude API key found in config.env file.")
        st.info("💡 Please configure your Claude API key in config.env file to use this application.")
        return
    
    # Initialize analyzer with Claude model and API key
    try:
        analyzer = ResumeAnalyzer(
            api_key=api_key, 
            model=model_choice, 
            optimize_for_free_tier=optimize_for_free_tier
        )
    except Exception as e:
        st.error(f"❌ Failed to initialize Claude analyzer: {str(e)}")
        return
    
    # Process resume
    resume_text = ""
    
    if uploaded_file:
        with st.spinner("📖 Extracting text from file..."):
            file_content = uploaded_file.read()
            
            if uploaded_file.type == "application/pdf":
                resume_text = analyzer.extract_text_from_pdf(file_content)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                resume_text = analyzer.extract_text_from_docx(file_content)
            else:  # text file
                resume_text = str(file_content, "utf-8")
                
    elif resume_text_input.strip():
        resume_text = resume_text_input.strip()
    
    if resume_text:
        # Show processing info
        estimated_tokens = len(resume_text) // 4 + len(job_description) // 4 + 1000  # Claude rough estimate
        st.info(f"📊 Estimated tokens for Claude analysis: ~{estimated_tokens}")
        if estimated_tokens > 6000 and optimize_for_free_tier:
            st.warning("⚠️ Large resume detected. Using conservative mode to manage token usage.")
        
        # Add generate button
        st.markdown("---")
        if st.button("🚀 Analyze Resume", type="primary", help="Click to start the resume analysis with Claude"):
            # Analyze resume with Claude
            with st.spinner("🤖 Analyzing your resume with Claude..."):
                start_time = time.time()
                feedback = analyzer.analyze_resume(resume_text, job_role, job_description)
                analysis_time = time.time() - start_time
            
            st.success(f"✅ Claude analysis completed in {analysis_time:.1f}s")
            
            # Display results
            display_results(feedback, resume_text, job_role, generate_improved, detailed_feedback, analyzer)
    else:
        st.info("👆 Upload a file or paste your resume text above to get started.")

def display_results(feedback: ResumeFeedback, original_resume: str, job_role: str, 
                   generate_improved: bool, detailed_feedback: bool, analyzer: ResumeAnalyzer):
    """Display analysis results"""
    
    # Overall score
    st.metric("Overall Score", f"{feedback.overall_score}/100")
    
    # Progress bar for score
    progress_color = "green" if feedback.overall_score >= 80 else "orange" if feedback.overall_score >= 60 else "red"
    st.progress(feedback.overall_score / 100)
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📝 Detailed Feedback", "✨ Improved Resume", "📊 Keywords"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💪 Strengths")
            for strength in feedback.strengths:
                st.success(f"✅ {strength}")
        
        with col2:
            st.subheader("⚠️ Areas for Improvement")
            for weakness in feedback.weaknesses:
                st.warning(f"⚠️ {weakness}")
        
        st.subheader("🚀 Improvement Suggestions")
        for i, suggestion in enumerate(feedback.improvement_suggestions, 1):
            st.info(f"{i}. {suggestion}")
    
    with tab2:
        if detailed_feedback and feedback.section_feedback:
            for section, feedback_text in feedback.section_feedback.items():
                if feedback_text:
                    st.subheader(f"📋 {section.replace('_', ' ').title()}")
                    st.write(feedback_text)
                    st.markdown("---")
    
    with tab3:
        if generate_improved:
            with st.spinner("✨ Generating improved resume..."):
                start_time = time.time()
                improved_resume = analyzer.generate_improved_resume(original_resume, feedback, job_role)
                generation_time = time.time() - start_time
            
            st.success(f"✅ Improved resume generated in {generation_time:.1f}s")
            
            st.subheader("✨ Improved Resume Version")
            st.markdown("*This is a Claude-generated improvement based on the feedback. Please review and edit as needed.*")
            
            if analyzer.optimize_for_free_tier:
                st.info("💡 Conservative mode: Resume was optimized for token efficiency while maintaining quality")
            else:
                st.info("🎯 Claude's advanced reasoning provides high-quality resume improvements")
            
            improved_text = st.text_area(
                "Improved Resume:",
                value=improved_resume,
                height=400,
                help="You can edit this improved version before copying"
            )
            
            # Download button for improved resume
            st.download_button(
                label="📥 Download Improved Resume",
                data=improved_text,
                file_name=f"improved_resume_{job_role.replace(' ', '_').lower()}.txt",
                mime="text/plain"
            )
        else:
            st.info("Enable 'Generate improved resume' in the sidebar to see an AI-improved version.")
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔑 Missing Keywords")
            if feedback.missing_keywords:
                for keyword in feedback.missing_keywords:
                    st.error(f"❌ {keyword}")
            else:
                st.success("✅ No critical keywords missing!")
        
        with col2:
            st.subheader("📊 Keyword Analysis")
            if feedback.missing_keywords:
                df = pd.DataFrame({
                    'Missing Keywords': feedback.missing_keywords,
                    'Priority': ['High'] * len(feedback.missing_keywords)
                })
                st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()
