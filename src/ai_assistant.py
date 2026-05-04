"""
src/ai_assistant.py — Free AI Assistant for GigScore (Pollinations.ai)

Uses Pollinations.ai free API — NO API key required!
- Text: POST https://text.pollinations.ai/ (OpenAI-compatible)
- Image: GET https://image.pollinations.ai/prompt/{description}

Fully free, unlimited, no authentication needed.
"""

import requests
import json
import urllib.parse

# ─────────────────────────────────────────────────────────────────────
# System Prompt — Fine-tuned for GigScore context
# ─────────────────────────────────────────────────────────────────────
GIGSCORE_SYSTEM_PROMPT = """You are GigScore AI, an expert assistant for India's alternative credit scoring system designed for gig economy workers (Ola, Uber, Swiggy, Zomato, Urban Company drivers and delivery partners).

## Your Role
You help gig worker applicants and loan officers understand:
- What their GigScore means and how it compares to traditional CIBIL scores
- Why specific factors helped or hurt their score
- Practical, actionable steps to improve their score
- How the credit limit is calculated
- What "Approved", "Conditional", and "Declined" decisions mean

## Key Domain Knowledge
- GigScore ranges from 0-100 (higher = better creditworthiness)
- Score = round((1 - default_probability) × 100)
- Score Bands: 0-39 Very Poor (Rejected), 40-54 Poor (Conditional), 55-69 Fair (Approved), 70-84 Good (Approved), 85-100 Excellent (Max Limit)
- Credit Limits: ₹0 / ₹25,000 / ₹75,000 / ₹1,50,000 / ₹3,00,000 per band
- The model uses 44+ behavioural features: income stability, platform tenure, UPI patterns, work consistency
- Gender DPD = 0.001 (near-zero bias), Income DPD = 0.150 (economically justified)
- Model: Stacking Ensemble best AUC-ROC 0.86, Gini 0.72, KS 56.5
- LightGBM AUC 0.86, XGBoost AUC 0.86, LogReg baseline AUC 0.84
- Top feature: platform_tenure_months (SHAP = 0.762)

## Response Rules
1. Use specific numbers from the applicant's profile when available
2. Keep answers concise (2-4 sentences for simple questions, up to 6 for complex ones)
3. Always mention actionable improvement steps when discussing low scores
4. Never recommend illegal or unethical ways to manipulate the score
5. If asked about something outside credit scoring, politely redirect
6. Use ₹ for currency, refer to Indian gig platforms by name (Swiggy, Ola, etc.)
7. Explain technical concepts in plain language — the user may not be financially literate

## Tone
Helpful, warm, professional. Avoid jargon. Speak as if talking to a gig worker who may not be financially literate. Be encouraging even for low scores — always show a path to improvement."""


def build_applicant_context(session_state: dict) -> str:
    """Builds a detailed context string from the current session state."""
    if session_state.get("score") is None:
        return "No applicant has been scored yet. The user may be asking general questions about GigScore."

    pos = session_state.get("positive_reasons", [])
    neg = session_state.get("negative_reasons", [])
    tips = session_state.get("improvement_tips", [])

    pos_text = "\n".join(f"  - {r.get('reason', str(r))}" for r in pos[:5]) if pos else "  - None identified"
    neg_text = "\n".join(f"  - {r.get('reason', str(r))}" for r in neg[:5]) if neg else "  - None identified"
    tips_text = "\n".join(f"  - {t.get('tip', str(t))} ({t.get('impact', '')})" for t in tips[:5]) if tips else "  - None available"

    return f"""
=== CURRENT APPLICANT PROFILE ===
GigScore: {session_state.get('score', 'N/A')}/100
Score Band: {session_state.get('band_label', 'N/A')}
Decision: {session_state.get('decision', 'N/A')}
Decision Detail: {session_state.get('decision_detail', 'N/A')}
Recommended Credit Limit: ₹{session_state.get('credit_limit', 0):,}
Default Probability: {session_state.get('default_prob', 'N/A')}

Positive Factors (helping the score):
{pos_text}

Risk Factors (hurting the score):
{neg_text}

Improvement Recommendations:
{tips_text}
=== END PROFILE ===
"""


def ask_gigscore_ai(question: str, applicant_context: str) -> str:
    """
    Calls Pollinations.ai free text API with question + full context.
    No API key required. Uses the /openai endpoint for standard JSON.
    """
    try:
        response = requests.post(
            "https://text.pollinations.ai/openai",
            json={
                "messages": [
                    {"role": "system", "content": GIGSCORE_SYSTEM_PROMPT},
                    {"role": "system", "content": applicant_context},
                    {"role": "user", "content": question},
                ],
                "model": "openai",
                "temperature": 0.7,
                "max_tokens": 500,
            },
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        if response.status_code == 200:
            try:
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", response.text)
            except (json.JSONDecodeError, ValueError):
                text = response.text.strip()
                return text if text else "I received an empty response. Please try again."
        else:
            return f"I'm having trouble connecting right now (status {response.status_code}). Please try again in a moment."

    except requests.Timeout:
        return "The request timed out. The AI service may be busy — please try again in a few seconds."
    except requests.ConnectionError:
        return "Could not connect to the AI service. Please check your internet connection."
    except Exception as e:
        return f"Something went wrong: {str(e)[:100]}. Please try again."


# ─────────────────────────────────────────────────────────────────────
# Demo Fallback Responses (when API is unreachable)
# ─────────────────────────────────────────────────────────────────────
DEMO_RESPONSES = {
    "default": "Based on this applicant's profile, the GigScore reflects their overall creditworthiness combining income stability, platform activity, and digital financial behavior. The score is calculated using a trained LightGBM ensemble model with 37+ behavioral features. Key positive factors include consistent work patterns, while areas like income volatility or short platform tenure may reduce the score. Each factor is weighted by its SHAP importance value.",
    "improve": "To improve this score, focus on: (1) Maintaining consistent weekly work schedules — this alone can add +5 to +12 points, (2) Building longer platform tenure by staying active, (3) Using UPI consistently to strengthen your digital financial footprint, and (4) Opening a savings account if you don't have one. These are the highest-impact levers based on our SHAP analysis.",
    "loan": "For loan eligibility, GigScore maps to 5 bands: EXCELLENT (85-100) qualifies for up to ₹3,00,000, GOOD (70-84) up to ₹1,50,000, FAIR (55-69) up to ₹75,000, POOR (40-54) requires a guarantor with max ₹25,000, and VERY POOR (0-39) is declined. The credit limit also scales with monthly income within each band.",
}


def get_demo_response(question: str) -> str:
    """Returns a pre-written demo response when the AI API is unavailable."""
    q_lower = question.lower()
    if any(w in q_lower for w in ['improve', 'better', 'increase', 'raise', 'higher']):
        return DEMO_RESPONSES['improve']
    if any(w in q_lower for w in ['loan', 'eligible', 'qualify', 'limit', 'approved']):
        return DEMO_RESPONSES['loan']
    return DEMO_RESPONSES['default']


def get_ai_generated_image_url(score: int, band_label: str) -> str:
    """
    Generates a Pollinations.ai image URL for the applicant's score visualization.
    No API key needed — just a URL that returns an image.
    Returns None if the service appears unavailable.
    """
    prompts = {
        "EXCELLENT": "glowing green diamond badge with golden sparkles on dark background digital art",
        "GOOD": "bright green crystal sphere radiating light on dark background digital art",
        "FAIR": "amber compass pointing upward with warm glow on dark background digital art",
        "POOR": "orange warning signal with upward arrow on dark background digital art",
        "VERY POOR": "dim red ember with small spark of hope on dark background digital art",
    }

    prompt = prompts.get(band_label, prompts["FAIR"])
    encoded = urllib.parse.quote(prompt)
    return f"https://image.pollinations.ai/prompt/{encoded}?width=300&height=300&nologo=true&seed={score}"
