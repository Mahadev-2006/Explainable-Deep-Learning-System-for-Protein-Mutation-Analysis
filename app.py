import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import subprocess
from captum.attr import IntegratedGradients

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Clinical Genomic Interpreter",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        text-align: center;
        border-left: 5px solid #4CAF50;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1f2937;
    }
    .metric-label {
        font-size: 14px;
        color: #6b7280;
    }
    .report-box {
        background-color: #1f2937;
        color: #f3f4f6;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        font-family: monospace;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# MODEL ARCHITECTURE (CACHED)
# =====================================================

class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, n_disease):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.path_head = nn.Linear(256, 1)
        self.dis_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_disease)
        )
        self.ddg_head = nn.Linear(256, 1)

    def forward(self, x):
        rep = self.shared(x)
        return (
            self.path_head(rep).squeeze(-1),
            self.dis_head(rep),
            self.ddg_head(rep).squeeze(-1)
        )

class ClinicalMutationInterpreter(MultiTaskModel):
    def _integrated_gradients(self, input_tensor, task="path"):
        if task == "path":
            def forward_fn(x):
                out, _, _ = self.forward(x)
                return out
        else:
            raise ValueError("Invalid task")

        ig = IntegratedGradients(forward_fn)
        return ig.attribute(input_tensor, target=None)

    def _block_importance(self, attributions):
        attr = attributions.squeeze().detach().cpu().numpy()
        return {
            "global": float(np.mean(np.abs(attr[:320]))),
            "mutation": float(np.mean(np.abs(attr[320:640]))),
            "local": float(np.mean(np.abs(attr[640:])))
        }

    def _stability_tier(self, ddg):
        abs_ddg = abs(ddg)
        if abs_ddg < 0.3: return "Minimal structural impact"
        elif abs_ddg < 1.0: return "Moderate structural perturbation"
        else: return "Significant structural destabilization"

    def _mechanism_hypothesis(self, position, seq_len, mutation_focus):
        if position <= 5:
            region_note = "N-terminal region"
            mechanism = "Potential alteration of translation initiation or early folding events."
        elif position >= seq_len - 5:
            region_note = "C-terminal region"
            mechanism = "Possible disruption of stability or protein-protein interaction interface."
        else:
            region_note = "Internal structured region"
            mechanism = "Likely perturbation of local secondary structure or binding regions."

        if mutation_focus > 0.5:
            driver = "Model interpretation indicates mutation-site dominant effect."
        else:
            driver = "Model interpretation suggests broader contextual structural influence."

        return region_note, mechanism, driver

    def get_structured_interpretation(self, input_tensor, wt, position, mut, sequence_length, disease_classes):
        self.eval()
        with torch.no_grad():
            path_pred, dis_pred, ddg_pred = self.forward(input_tensor)

        pathogenic = torch.sigmoid(path_pred).item() > 0.5
        ddg_value = ddg_pred.item()

        if pathogenic:
            disease_idx = dis_pred.argmax(1).item()
            disease_name = disease_classes[disease_idx]
        else:
            disease_name = "None"

        # Enable grads just for captum
        input_tensor.requires_grad_()
        attr = self._integrated_gradients(input_tensor, task="path")
        importance = self._block_importance(attr)
        mutation_focus = importance["mutation"] / sum(importance.values())

        stability_tier = self._stability_tier(ddg_value)
        region_note, mechanism, driver = self._mechanism_hypothesis(
            position, sequence_length, mutation_focus
        )

        return {
            "mutation": f"{wt}{position}{mut}",
            "ddg": ddg_value,
            "stability_text": stability_tier,
            "pathogenic": "Pathogenic" if pathogenic else "Likely Benign",
            "disease": disease_name,
            "region": region_note,
            "mechanism": mechanism,
            "model_driver": driver,
            "mutation_focus": mutation_focus,
            "importance": importance
        }

# =====================================================
# DATA LOADING (CACHED)
# =====================================================

@st.cache_resource
def load_data_and_model():
    # Load tensors & df
    X_test = torch.load(r"C:\SEM 4\patho\test_features.pt", map_location=DEVICE)
    train_df = pd.read_csv(r"C:\SEM 4\patho\Processed\train.csv")
    test_df = pd.read_csv(r"C:\SEM 4\patho\Processed\test.csv")
    
    disease_classes = sorted(train_df["Disease_Category"].unique())
    
    # Filter for True Positive Pathogenics for demonstration
    pathogenic_mask = (test_df["ClinicalSignificance"] == "Pathogenic")
    pathogenic_df = test_df[pathogenic_mask].reset_index(drop=False)
    
    # Load model
    model = ClinicalMutationInterpreter(
        input_dim=X_test.shape[1],
        n_disease=len(disease_classes)
    ).to(DEVICE)
    model.load_state_dict(torch.load("best_multitask_model.pt", map_location=DEVICE, weights_only=True))
    model.eval()
    
    return X_test, pathogenic_df, disease_classes, model, test_df

X_test, demo_df, disease_classes, model, test_df = load_data_and_model()

# =====================================================
# LLM GENERATION
# =====================================================

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
- Do NOT mention that You are a board-certified clinical molecular geneticist.
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
    try:
        # Popen with `--temperature 0` was silently failing on Streamlit's async loop.
        # Restoring `subprocess.run` to guarantee stdout text capture while keeping prompt.
        result = subprocess.run(
            ["ollama", "run", "phi"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        if result.returncode != 0:
            return f"Ollama execution failed.\nError details: {result.stderr.strip()}"
        
        output_text = result.stdout.strip()
        if not output_text:
            return "Ollama returned an empty response. Please check if the 'phi' model is fully downloaded."
            
        return output_text
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}\n\nPlease ensure Ollama is running locally with the 'phi' model installed."

# =====================================================
# STREAMLIT UI
# =====================================================

st.markdown("<h1 style='text-align: center;'>🧬 Clinical Genomic Interpreter</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 40px;'>An Explainable Deep Learning Framework for Protein Mutation Stability and Pathogenicity Assessment.</p>", unsafe_allow_html=True)

st.sidebar.header("Mutation Selection")

def format_mutation(idx):
    row = demo_df.loc[idx]
    gene = row.get("GeneSymbol", "Unknown")
    mut_str = f"{row['WT']}{row['Position']}{row['MUT']}"
    return f"{gene} - {mut_str}"

selected_idx = st.sidebar.selectbox("Choose Mutation", options=demo_df.index, format_func=format_mutation)
selected_original_idx = demo_df.loc[selected_idx, "index"]
selected_row = test_df.iloc[selected_original_idx]

st.sidebar.markdown("---")
st.sidebar.write("**Selected Details:**")
st.sidebar.write(f"**Gene:** {selected_row.get('GeneSymbol', 'Unknown')}")
st.sidebar.write(f"**Phenotype:** {selected_row.get('PhenotypeList', 'Unknown')}")
st.sidebar.write(f"**ClinVar Impact:** {selected_row.get('ClinicalSignificance', 'Pathogenic')}")

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    run_inference = st.button("🚀 Run & Generate Clinical Report", type="primary", use_container_width=True)

if run_inference:
    
    with st.spinner("🧠 Running Multi-Task Deep Learning Model..."):
        # Get PyTorch Tensor
        input_tensor = X_test[selected_original_idx].unsqueeze(0).to(DEVICE).clone()
        
        # Pull sequence info
        wt = selected_row["WT"]
        position = selected_row["Position"]
        mut = selected_row["MUT"]
        # Fallback length if not in df
        seq_len = len(selected_row.get("sequence", "A"*500)) 
        
        # Run Interpreter
        structured_data = model.get_structured_interpretation(
            input_tensor, wt, position, mut, seq_len, disease_classes
        )
    
    # ---------------------------
    # DISPLAY PYTORCH METRICS
    # ---------------------------
    st.subheader("Model Predictions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box" style="border-left-color: {'#ef4444' if structured_data['pathogenic'] == 'Pathogenic' else '#10b981'};">
            <div class="metric-value">{structured_data['pathogenic']}</div>
            <div class="metric-label">Classification</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-box" style="border-left-color: #3b82f6;">
            <div class="metric-value">{structured_data['disease']}</div>
            <div class="metric-label">Disease Category</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-box" style="border-left-color: #f59e0b;">
            <div class="metric-value">{structured_data['ddg']:.3f} kcal/mol</div>
            <div class="metric-label">Predicted ΔΔG Stability</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ---------------------------
    # DISPLAY XAI (EXPLAINABILITY)
    # ---------------------------
    st.subheader("Explainable AI")
    
    st.write(f"**Structural Region:** {structured_data['region']}")
    st.write(f"**Mechanism Hypothesis:** {structured_data['mechanism']}")
    st.write(f"**Captum Driver:** {structured_data['model_driver']}")
        
    st.markdown("---")
    
    # ---------------------------
    # DISPLAY OLLAMA GENERATION
    # ---------------------------
    st.subheader("Clinical report")
    
    with st.spinner("🤖 Prompting local Ollama LLM..."):
        clinical_report = generate_llm_report(structured_data)
        
    st.markdown(f"""
    <div class="report-box">
        {clinical_report}
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("👈 Select a mutation from the sidebar and click 'Run Inference' to begin.")
