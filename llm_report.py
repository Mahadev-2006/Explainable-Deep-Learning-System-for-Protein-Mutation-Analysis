import subprocess

def generate_llm_report(data):

    prompt = f"""
You are a board-certified clinical molecular geneticist.

Prepare a formal mutation interpretation report for a medical specialist.

Mutation: {data["mutation"]}
Predicted ddG: {data["ddg"]:.3f} kcal/mol
Structural Interpretation: {data["stability_text"]}
Pathogenic Classification: {data["pathogenic"]}
Predicted Disease Category: {data["disease"]}
Mutation Location: {data["region"]}
Functional Hypothesis: {data["mechanism"]}
Model Interpretation Insight: {data["model_driver"]}

Requirements:
- Do NOT mention AI, computational models, or probabilities.
- Do NOT include conversational phrases.
- Do NOT say "thank you".
- Avoid generic statements like "more research is needed".
- Use concise, professional clinical language.
- Clearly distinguish structural impact from functional consequence.
- Provide a definitive clinical-style conclusion.

Format:
Paragraph style.
No bullet points.
No filler text.
"""

    process = subprocess.Popen(
        ["ollama", "run", "phi", "--temperature", "0"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8"
    )

    output, error = process.communicate(prompt)

    return output.strip()