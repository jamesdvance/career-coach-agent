from typing import TypedDict, List, Optional
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
import re
import os
import csv
from datetime import datetime


class CompanyInfo(TypedDict):
    is_public: bool
    latest_funding_round: Optional[str]
    total_funding_raised: Optional[str]
    is_profitable: Optional[bool]
    founded_year: Optional[str]
    employee_count: Optional[str]


class JobAnalysisState(TypedDict):
    job_url: str
    raw_content: str
    job_title: str
    company_name: str
    key_skills: List[str]
    company_info: CompanyInfo
    error: str
    model_vendor: str


def check_auth_token(model_vendor: str) -> Optional[str]:
    """Check for required authentication tokens in environment variables"""
    if model_vendor == "anthropic":
        token = os.getenv("ANTHROPIC_API_KEY")
        if not token:
            return "Missing ANTHROPIC_API_KEY environment variable"
    elif model_vendor == "openai":
        token = os.getenv("OPENAI_API_KEY")
        if not token:
            return "Missing OPENAI_API_KEY environment variable"
    else:
        return f"Unsupported model vendor: {model_vendor}"
    
    return None


def fetch_job_posting(state: JobAnalysisState) -> JobAnalysisState:
    """Fetch job posting content from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(state["job_url"], headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        text_content = soup.get_text(separator=' ', strip=True)
        
        return {
            **state,
            "raw_content": text_content,
            "error": ""
        }
    except Exception as e:
        return {
            **state,
            "error": f"Failed to fetch job posting: {str(e)}"
        }


def extract_job_details(state: JobAnalysisState) -> JobAnalysisState:
    """Extract job title, company, and key skills using LLM"""
    if state.get("error"):
        return state
    
    try:
        model_vendor = state.get("model_vendor", "anthropic")
        
        # Check for authentication token
        auth_error = check_auth_token(model_vendor)
        if auth_error:
            return {
                **state,
                "error": auth_error
            }
        
        if model_vendor == "anthropic":
            llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
        elif model_vendor == "openai":
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        else:
            raise ValueError(f"Unsupported model vendor: {model_vendor}")
        
        prompt = f"""
        Analyze the following job posting content and extract:
        1. Job title
        2. Company name
        3. Key skills (list of 5-10 most important technical/professional skills)
        
        Job posting content:
        {state["raw_content"][:3000]}
        
        Respond in this exact format:
        TITLE: [job title]
        COMPANY: [company name]
        SKILLS: [skill1, skill2, skill3, ...]
        """
        
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content
        
        # Parse the response
        title_match = re.search(r'TITLE:\s*(.+)', content)
        company_match = re.search(r'COMPANY:\s*(.+)', content)
        skills_match = re.search(r'SKILLS:\s*(.+)', content)
        
        job_title = title_match.group(1).strip() if title_match else "Unknown"
        company_name = company_match.group(1).strip() if company_match else "Unknown"
        
        skills = []
        if skills_match:
            skills_text = skills_match.group(1).strip()
            skills = [skill.strip() for skill in skills_text.split(',')]
        
        return {
            **state,
            "job_title": job_title,
            "company_name": company_name,
            "key_skills": skills
        }
    
    except Exception as e:
        return {
            **state,
            "error": f"Failed to extract job details: {str(e)}"
        }


def research_company(state: JobAnalysisState) -> JobAnalysisState:
    """Research company financial information using web search and LLM analysis"""
    if state.get("error"):
        return state
    
    company_name = state.get("company_name", "")
    if not company_name or company_name == "Unknown":
        return {
            **state,
            "company_info": {
                "is_public": False,
                "latest_funding_round": None,
                "total_funding_raised": None,
                "is_profitable": None,
                "founded_year": None,
                "employee_count": None
            }
        }
    
    try:
        model_vendor = state.get("model_vendor", "anthropic")
        
        # Check for authentication token
        auth_error = check_auth_token(model_vendor)
        if auth_error:
            return {
                **state,
                "error": auth_error
            }
        
        if model_vendor == "anthropic":
            llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
        elif model_vendor == "openai":
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        else:
            raise ValueError(f"Unsupported model vendor: {model_vendor}")
        
        # Search for company information
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        search_queries = [
            f"{company_name} funding rounds venture capital",
            f"{company_name} company profile financials public private",
            f"{company_name} founded when established year"
        ]
        
        search_results = []
        for query in search_queries:
            try:
                # Use Google search (this is a simplified approach)
                search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
                response = requests.get(search_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    # Extract search result snippets
                    snippets = soup.find_all('span', attrs={'data-ved': True})
                    for snippet in snippets[:3]:  # Take first 3 snippets
                        text = snippet.get_text(strip=True)
                        if len(text) > 50:  # Only meaningful text
                            search_results.append(text)
            except:
                continue
        
        # Combine search results
        combined_info = " ".join(search_results[:10])  # Limit to avoid token limits
        
        prompt = f"""
        Analyze the following information about {company_name} and extract company financial details.
        
        Information:
        {combined_info[:2000]}
        
        Please extract and provide:
        1. Is the company public or private? (respond with "public" or "private")
        2. Latest funding round (e.g., "Series A", "Series B", "IPO", "Bootstrap" or "Unknown")
        3. Total funding raised (e.g., "$50M", "$1.2B" or "Unknown")
        4. Is the company profitable? (respond with "yes", "no", or "unknown")
        5. When was the company founded? (year only, e.g., "2015" or "Unknown")
        6. Approximate employee count (e.g., "100-500", "1000+" or "Unknown")
        
        Respond in this exact format:
        PUBLIC_STATUS: [public/private]
        LATEST_ROUND: [funding round or Unknown]
        TOTAL_FUNDING: [amount or Unknown]  
        PROFITABLE: [yes/no/unknown]
        FOUNDED: [year or Unknown]
        EMPLOYEES: [count range or Unknown]
        """
        
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content
        
        # Parse the response
        public_match = re.search(r'PUBLIC_STATUS:\s*(.+)', content, re.IGNORECASE)
        funding_match = re.search(r'LATEST_ROUND:\s*(.+)', content, re.IGNORECASE)
        total_match = re.search(r'TOTAL_FUNDING:\s*(.+)', content, re.IGNORECASE)
        profit_match = re.search(r'PROFITABLE:\s*(.+)', content, re.IGNORECASE)
        founded_match = re.search(r'FOUNDED:\s*(.+)', content, re.IGNORECASE)
        employee_match = re.search(r'EMPLOYEES:\s*(.+)', content, re.IGNORECASE)
        
        is_public = public_match.group(1).strip().lower() == "public" if public_match else False
        latest_round = funding_match.group(1).strip() if funding_match else None
        total_funding = total_match.group(1).strip() if total_match else None
        profitable_text = profit_match.group(1).strip().lower() if profit_match else "unknown"
        founded_year = founded_match.group(1).strip() if founded_match else None
        employee_count = employee_match.group(1).strip() if employee_match else None
        
        # Convert profitable text to boolean or None
        is_profitable = None
        if profitable_text == "yes":
            is_profitable = True
        elif profitable_text == "no":
            is_profitable = False
        
        # Clean up "Unknown" values
        if latest_round and latest_round.lower() == "unknown":
            latest_round = None
        if total_funding and total_funding.lower() == "unknown":
            total_funding = None
        if founded_year and founded_year.lower() == "unknown":
            founded_year = None
        if employee_count and employee_count.lower() == "unknown":
            employee_count = None
        
        company_info: CompanyInfo = {
            "is_public": is_public,
            "latest_funding_round": latest_round,
            "total_funding_raised": total_funding,
            "is_profitable": is_profitable,
            "founded_year": founded_year,
            "employee_count": employee_count
        }
        
        return {
            **state,
            "company_info": company_info
        }
    
    except Exception as e:
        # Return empty company info on error, don't fail the entire workflow
        return {
            **state,
            "company_info": {
                "is_public": False,
                "latest_funding_round": None,
                "total_funding_raised": None,
                "is_profitable": None,
                "founded_year": None,
                "employee_count": None
            }
        }


def create_career_coach_workflow(model_vendor: str = "anthropic") -> CompiledStateGraph:
    """Create and compile the career coach workflow"""
    workflow = StateGraph(JobAnalysisState)
    
    # Add nodes
    workflow.add_node("fetch", fetch_job_posting)
    workflow.add_node("extract", extract_job_details)
    workflow.add_node("research", research_company)
    
    # Define edges
    workflow.set_entry_point("fetch")
    workflow.add_edge("fetch", "extract")
    workflow.add_edge("extract", "research")
    workflow.add_edge("research", END)
    
    return workflow.compile()


def read_job_links(file_path: str = "links.txt") -> List[str]:
    """Read job posting URLs from a text file, one per line"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            links = [line.strip() for line in file if line.strip()]
        return links
    except FileNotFoundError:
        print(f"Error: {file_path} not found")
        return []
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def analyze_job_posting(job_url: str, model_vendor: str = "anthropic") -> JobAnalysisState:
    """Main function to analyze a job posting"""
    workflow = create_career_coach_workflow(model_vendor)
    
    initial_state: JobAnalysisState = {
        "job_url": job_url,
        "raw_content": "",
        "job_title": "",
        "company_name": "",
        "key_skills": [],
        "company_info": {
            "is_public": False,
            "latest_funding_round": None,
            "total_funding_raised": None,
            "is_profitable": None,
            "founded_year": None,
            "employee_count": None
        },
        "error": "",
        "model_vendor": model_vendor
    }
    
    result = workflow.invoke(initial_state)
    return result


def analyze_multiple_job_postings(model_vendor: str = "anthropic") -> List[JobAnalysisState]:
    """Analyze multiple job postings from links.txt file"""
    links = read_job_links()
    results = []
    
    for i, url in enumerate(links, 1):
        print(f"Analyzing job {i}/{len(links)}: {url}")
        result = analyze_job_posting(url, model_vendor)
        results.append(result)
    
    return results


def save_results_to_csv(results: List[JobAnalysisState], filename: str = None):
    """Save job analysis results to CSV file"""
    if not results:
        print("No results to save.")
        return
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"job_analysis_{timestamp}.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'job_url', 'job_title', 'company_name', 'key_skills',
            'is_public', 'latest_funding_round', 'total_funding_raised',
            'is_profitable', 'founded_year', 'employee_count', 'error'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            company_info = result.get('company_info', {})
            row = {
                'job_url': result.get('job_url', ''),
                'job_title': result.get('job_title', ''),
                'company_name': result.get('company_name', ''),
                'key_skills': ', '.join(result.get('key_skills', [])),
                'is_public': 'Yes' if company_info.get('is_public') else 'No',
                'latest_funding_round': company_info.get('latest_funding_round', ''),
                'total_funding_raised': company_info.get('total_funding_raised', ''),
                'is_profitable': ('Yes' if company_info.get('is_profitable') 
                                 else 'No' if company_info.get('is_profitable') is False 
                                 else 'Unknown'),
                'founded_year': company_info.get('founded_year', ''),
                'employee_count': company_info.get('employee_count', ''),
                'error': result.get('error', '')
            }
            writer.writerow(row)
    
    print(f"Results saved to {filename}")
    return filename


# Mock frontend interface for testing
if __name__ == "__main__":
    # Check if links.txt exists and analyze multiple job postings
    try:
        with open("links.txt", 'r') as f:
            pass  # File exists
        
        print("Analyzing multiple job postings from links.txt...")
        results = analyze_multiple_job_postings()
        
        if not results:
            print("No job links found in links.txt or file is empty.")
        else:
            # Save results to CSV
            filename = save_results_to_csv(results)
            
            # Summary
            successful = sum(1 for r in results if not r["error"])
            failed = len(results) - successful
            print(f"Summary: Analyzed {len(results)} job postings")
            print(f"Successful: {successful}, Failed: {failed}")
    
    except FileNotFoundError:
        print("links.txt not found. Running with example URL...")
        # Fallback to single job analysis
        test_url = "https://www.linkedin.com/jobs/view/4284180479/?refId=d9e84332-15e0-4302-bcae-daca4627e58e&trackingId=t0DFY6iDRJSs9%2F72fB0QFQ%3D%3D"
        result = analyze_job_posting(test_url)
        
        # Save single result to CSV
        save_results_to_csv([result])