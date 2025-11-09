import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in .env file.")

genai.configure(api_key=GEMINI_API_KEY)
# Using a fast and reliable model for this task
# Using the model from your original project
model = genai.GenerativeModel('models/gemini-2.5-pro')
# --- Flask App Setup ---
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# --- Prompts Dictionary ---
# Define all your different prompts here
PROMPTS = {
    "resume": """
    I need you to analyze how well my LaTeX resume matches a specific job description.
Rate the match from 1–10.0 based on how aligned my experience, skills, and keywords are with the job.
If the rating is below 9.5, make targeted suggestions to improve my resume to raise it above 9.9, following the rules below.

🚨 Strict Rules:
My resume is written in LaTeX code. Do not modify any LaTeX syntax or structure — replace each bullet point text line by line based on this job description. Use data-driven, results-oriented language.
When revising bullet points:
🔴 Top Priority:
Each revised bullet point must individually match the original bullet’s character length (including spaces) — not a uniform length across all bullets.
The new bullet should be within ±3 characters of the original.
It can be slightly shorter (up to 3 fewer characters) but never longer than the original.
If needed, remove filler words, secondary quantifiers, or less-critical phrases to maintain this length limit.
Always rewrite concisely without expanding sentence structure or adding unnecessary words.
Example:
Original bullet:
"Devised an AI-powered video multi-tagging system with NVIDIA NIM micro services, building ETL pipelines for video preprocessing, transcription, and multimodal entity recognition, enabling a new product that analyzed 10K+ hours of video"
→ (245 characters)
Revised bullet must also be around 242–248 characters — no more, no less.
There should be no full stop (period) in any of the bulletins.

Maintain the same starting action verb (e.g., “Developed,” “Implemented,” “Designed,” etc.).
Include at least one quantifiable metric or measurable outcome.
Do not add indentation, tabs, or spaces before \item — every bullet must begin flush-left.

Do not use any markdown, bold text, or formatting markers (no **, _, or highlighting). Output must be plain LaTeX text.
When writing “%” in LaTeX, escape it properly as "\%" so the document compiles.
Identify all keywords in the job description and naturally integrate them into the bullet points (only where relevant).
For the Media Stream AI experience section only, you are allowed to completely rewrite bullet point content to match the job description exactly — make it realistic, naturally written, and human-like.

✨ Additional Tailoring Rules (to be followed for every revision):
Your task is to revise every single bullet point under each job/role on my resume using the provided job description. Each bullet should be rewritten as an achievement-based statement, not just a task.
Each bullet must highlight:
What was done (the core duty or responsibility),
How it was done (tools, methods, collaboration, etc.),
What the impact or outcome was (quantifiable where possible).
All bullet points should be aligned with the job description and use language, keywords, and priorities that reflect the responsibilities and requirements of the target job.
If a bulletin in my resume is not relevant to the job description, rewrite it to make it relevant while keeping:
the same starting action verb,
approximately the same character length as its original (±3 characters, never longer),and the same grammatical style and tone.
SKILLS SECTION OPTIMIZATION rule
Objective: Remove skills NOT in the job description. Keep ALL skills that align with the JD.
Rules:
KEEP: Any skill mentioned in the JD (exact match, synonym, or subset/superset)
Example: JD says "React" → Keep "React.js", "React Native"
Example: JD says "AWS" → Keep "AWS Lambda", "S3", "EC2"
REMOVE: Skills with zero relevance to the JD
REMOVE: Redundant duplicates (e.g., "JavaScript" and "JS")
ADD: Skills mentioned 2+ times in JD but missing from resume
REORDER: Place JD-aligned skills first within each category
PRESERVE: Exact LaTeX syntax, category headers, and formatting
CRITICAL: If a skill appears ANYWHERE in the JD, keep it. Only remove skills with NO connection to the JD.

🏭 Industry Alignment Rule:
Whenever I provide the job description, you should:
Identify the sector/industry of the company I’m applying to.
Check whether my existing “Full Stack Storage Platform” project title aligns with that industry at least 70%.
If it does not, change the project title to something more relevant to that industry (e.g., for CVS → “Full Stack Hospital Management System” or “Full Stack Medicine Shop”).
Keep all bulletins under that project following the same rules (same verb, same length, same structure), but adapt the meaning to fit the new project title and industry context.

📥 After revisions, provide:
The revised LaTeX code. STrictly only output the LaTeX code with no extra explanation.

Input Sections:

[RESUME LATEX CODE]
(  {resume_latex} )

[JOB DESCRIPTION]
(  {job_description} )
    """,



    "cover_letter": """
    You are an expert career strategist and copywriter who specializes in writing persuasive, personalized, marketing-style cover letters that sell the candidate — not summarize their biography.
Your goal is to write a cover letter that feels specific, confident, and aligned with both the job and the company. Follow the exact process and structure outlined below.

⚙️ INPUTS I WILL PROVIDE YOU:
My Resume: i will paste below
Job Description: I will paste below
Optional: [I will give below]

🧩 STEP 1 — Analyze the Job Description
Identify the “What You’ll Do” section. Extract the top 3–5 responsibilities.
Identify the “Qualifications” section. Highlight the top 3–5 requirements or preferred skills.
Create a simple table like this:
Job Requirement	My Experience (from Resume)
Example: Partner with Product & Engineering	Led 3 cross-functional analytics projects with engineering and product teams
Example: Develop scalable code	Built and deployed production-grade ETL pipelines using Airflow & Spark
🧭 STEP 2 — Identify What to Include
Select the two strongest qualification matches from the table above.
Use the same wording as the job description to mirror their language and ATS keywords.
Each will later be turned into one body paragraph linked to a story theme (e.g., leadership, initiative, curiosity, managing conflict).
💡 STEP 3 — Research & Align Motivation
Research the company and write 2–3 sentences answering:
What is the company’s mission or key product?
Why does it interest me personally or professionally?
What values, impact, or direction do I resonate with?
You will later use this in the “Why I Want to Work Here” paragraph.
🧱 STEP 4 — Write the Cover Letter Using This Structure
(i) Who You Are, What You Want, What You Believe In
Open with 2 short sentences describing:
Your professional identity (role, years of experience)
What you care about professionally (values, goals, or motivation)
Optional: a company-specific mention
Examples:
“I’m a product-focused Software Engineer with 3 years of startup experience who loves building reliable, user-friendly applications.”
“I’m a data-driven Analyst passionate about turning raw information into business growth stories.”
(ii) Transition
Summarize your recent achievement and tie it to your excitement for this role.
Template: “Over the last [X months/years], I’ve [insert your biggest measurable contribution]. Now, I’m excited to contribute and grow at [COMPANY]. There are three things that make me the perfect fit for this position:”
(iii) Skills & Qualification Match (2 Body Paragraphs)
Pick two major skills that align with the job and structure each paragraph using this formula:
1/ Theme – Choose one story theme (e.g., taking initiative, leading people, curiosity).
2/ Context – Briefly describe the situation or project.
3/ What You Did – Be specific and use measurable impact.
4/ Why It Matters – End with what you learned or how it made you stronger.
Example: Theme: Taking Initiative “I like to go above and beyond in whatever I do. As one of the youngest data scientists at my startup, I investigated fraud analytics within our platform. Instead of just reporting findings, I created a company-wide document on fraud detection best practices, which saved hundreds of client support hours and reshaped our strategy.”
(iv) Why You Want to Work There
Choose your top two reasons from your research: one value-driven and one industry- or product-driven.
Template: “Third, I’ve been following [COMPANY] for a while and resonate with both its values and its direction. The [Insert Value] really stands out to me because [reason]. I also recently read that [Insert topical reason] — which excites me because [how it connects to your goals or skills].”
(v) Conclusion
End confidently and clearly: “I believe my experience is a strong match for [COMPANY] and this position. I’m eager to bring my [specific skills or impact] to your team and would love the opportunity to discuss how I can contribute.”
Finish with: “Thank you for your time and consideration.
Best,
[Your Full Name]”

🧠 OUTPUT EXPECTATION:
When generating the cover letter, ensure:
It’s no longer than one page (≈300–400 words).
Uses plain, confident, action-focused language (no buzzwords like “synergy” or “results-driven”).
Reads like a personal pitch, not an essay.
Is formatted cleanly with 4–5 paragraphs.
Integrates quantifiable impact where possible.

OUTPUT ONLY the final cover letter text without any extra commentary.
    
    JOB DESCRIPTION:
    {job_description}
    
    MY RESUME (for context):
    {resume_latex}
    """,
    "linkedin_note": """
    You are a networking expert.
    Write a short, friendly, and professional LinkedIn connection request note. 
    Add value by mentioning a specific reason for connecting based on the job description provided. 
    Like expressing admiration for the company's work.
    Keep it concise (under 300 characters) and avoid generic phrases.
    Output only the text without any extra commentary.
    Example- Hi Name, I wanted to connect because I'm interested in the [Role in jD] role at your company and would appreciate any help to let me get in touch with the right contact. Thank you! Prince.

    JOB DESCRIPTION:
    {job_description}
    MY RESUME (for context):
    {resume_latex}
    """,


    "cold_email": """
    I am going to send you my cold email rule — use it strictly when writing any cold email for me.
    RULE SUMMARY:
Emails must be < 90 words.
Use 2-sentence paragraphs with lots of white space.
Write like you talk — simple, human, and casual (3rd-grade reading level).
Avoid jargon, clichés, and unnecessary details.
Never include more than 1 question.
Never include links or attachments.
Focus on the prospect’s pain point or goal, not my company.
Only mention one feature or benefit that’s directly relevant.
Assume the reader will scan quickly (use bold words strategically).
Send emails as if they are personal notes, not marketing copy.
Structure to follow for every cold email:
Version 1 (Intro Email):
Hey [prospect name] – I saw that you’re [their role] and likely focused on [key goal/problem you can solve].
My company helps [target companies] [unique selling point] which leads to [specific benefit].
Would you be open to a quick intro call on [day or time]?
Version 2 (Feature Email):
Hey [prospect name] – Not sure if you’ve ever looked into [specific feature or idea relevant to them], but I thought you might find this interesting.
We worked with [relevant client] to [achieve specific result or improvement].
Just wanted to see if [their company] might be interested in checking this out — if so, we can set up a quick call!
Additional Research Insights:
Send between 7:00–9:30 AM for best open rates.
People decide in 2.7 seconds whether to read or delete — start strong.
80% will only scan — make it visually easy.

TASK:
Write me a cold email using my details and following all the rules and structure above. 

Output only the email text without any extra commentary, text formatting.

    JOB DESCRIPTION:
    {job_description}
    
    MY RESUME (for context):
    {resume_latex}
    """
    # You can add more prompts here, like "prompt_5", "prompt_6"
}


# --- Flask Routes ---

@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Handles the AI generation requests."""
    try:
        data = request.json
        resume_latex = data.get('resume_latex')
        job_description = data.get('job_description')
        prompt_type = data.get('prompt_type') # e.g., "resume", "cover_letter"

        if not resume_latex or not job_description:
            return jsonify({"error": "Resume and Job Description cannot be empty."}), 400

        # Find the master prompt from our dictionary
        master_prompt = PROMPTS.get(prompt_type)
        if not master_prompt:
            return jsonify({"error": "Invalid prompt type."}), 400
        
        # Format the chosen prompt with the user's input
        final_prompt = master_prompt.format(
            job_description=job_description,
            resume_latex=resume_latex
        )
        
        # Call the Gemini API
        response = model.generate_content(final_prompt)
        
        # Return the generated text
        return jsonify({"result": response.text})

    except Exception as e:
        app.logger.error(f"Error in /generate: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)