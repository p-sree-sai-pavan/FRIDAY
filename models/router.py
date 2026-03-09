import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from groq import Groq

groq_client = Groq(api_key=config.GROQ_API_KEY)

MODELS = {
    "simple":    "llama-3.1-8b-instant",
    "general":   "llama-3.3-70b-versatile",
    "tools":     "llama-3-groq-70b-tool-use",
    "vision":    "llama4-scout-17b-16e",
    "reasoning": "llama-3.3-70b-versatile",
    "fallback":  "gemini-2.5-flash"
}

def detect_task(prompt):
    classification = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",  # fast small model for classification
        messages=[{
            "role": "user",
            "content": f"""You are a task classifier. Classify the request below into exactly ONE of these categories:

simple    → basic questions, greetings, time, weather
tools     → open apps, run commands, execute anything
reasoning → complex analysis, planning, building, research  
vision    → anything about images or screen
general   → everything else

Request: {prompt}

Reply with ONE word only from the list above."""
        }],
        max_tokens=5,
        temperature=0
    )
    result = classification.choices[0].message.content.strip().lower()
    return result

def route(prompt, task_type=None):
    if task_type:
        return MODELS.get(task_type, MODELS["general"])
    
    try:
        detected = detect_task(prompt)
        return MODELS.get(detected, MODELS["general"])  # fallback to general
    except Exception as e:
        print(f"Router error: {e}")  # ADD THIS
        return MODELS["general"]  # if classification fails → use general
    
if __name__ == "__main__":
    print(route("open spotify"))
    print(route("what is python"))
    print(route("build me a web scraper"))