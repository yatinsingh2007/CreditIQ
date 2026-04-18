# ─── Install all required packages ───────────────────────────────────────
# Run this cell once per Colab session (or whenever the runtime is reset).



import os
import getpass

# ─── Groq API Key ─────────────────────────────────────────────────────────
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = ""
print("Groq API key: SET ✅")

# ─── Model Path ───────────────────────────────────────────────────────────
# Upload dt_model.pkl via the Files panel (📁) or adjust the path below.
os.environ["DT_MODEL_PATH"] = "dt_model.pkl"   # ← update if needed
print(f"Model path  : {os.environ['DT_MODEL_PATH']} ✅")

# -- Standard library ---------------------------------------------------------
import os
import re
import ast
import json
import pickle
import traceback
from datetime import datetime, timezone

# -- Third-party --------------------------------------------------------------
import numpy as np
import pandas as pd
from groq import Groq
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# =============================================================================
# SECTION 1 -- CONFIGURATION
# All tuneable constants live here. Change them here; do not scatter magic
# numbers through the rest of the file.
# =============================================================================

# Groq models used in the pipeline.
# STRONG: for complex reasoning and tool orchestration.
# FAST: for simple planning and auditing (saves tokens/avoids 429s).
GROQ_MODEL_STRONG = "llama-3.3-70b-versatile"
GROQ_MODEL_FAST   = "llama-3.1-8b-instant"


# Path to the pickled Decision Tree model package.
# Override at runtime by setting the DT_MODEL_PATH environment variable.
MODEL_PATH = os.getenv("DT_MODEL_PATH", "dt_model_streamlit.pkl")

# Hard cap on how many tool-calling iterations the Executor may make per run.
# Prevents infinite loops if the LLM keeps calling tools without terminating.
MAX_EXECUTOR_ITERS = 8

# How many times the Reflector is allowed to request a retry before we
# give up and proceed to the Reporter regardless.
MAX_REFLECT_RETRIES = 2

# =============================================================================
# SECTION 2 -- MODULE-LEVEL CACHE VARIABLES
# These replace @lru_cache. Each is initialised to None and populated the
# first time its corresponding loader function is called. Subsequent calls
# return the cached value immediately without hitting disk or rebuilding.
# Using plain variables (not decorators) keeps the code transparent and simple.
# =============================================================================

# Holds the loaded model package dict after the first call to load_model_package().
_MODEL_PKG_CACHE = None

# Holds the ChromaDB collection after the first call to get_vector_store().
_VECTOR_STORE_CACHE = None

# =============================================================================
# SECTION 3 -- ROBUST JSON EXTRACTION
# LLMs are inconsistent. Even with response_format=json_object they may
# slip in Python-style booleans, single-quoted strings, or trailing commas.
# This section provides three layered helpers that together handle every
# observed failure mode.
# =============================================================================

def normalise_llm_json_text(text):
    """
    Coerce Python-literal syntax into valid JSON syntax.

    LLMs trained on Python source code frequently emit Python-style tokens
    rather than strict JSON. This function corrects the most common deviations:

        Python token    JSON token
        ----------------------------
        True            true
        False           false
        None            null
        'single quote'  "double quote"
        trailing ,}     }
        trailing ,]     ]

    Boolean/None replacements use word-boundary anchors so substrings like
    "TrueColor" or "NoneType" are left untouched.

    Parameters
    ----------
    text : str
        Raw string that may contain Python-literal tokens.

    Returns
    -------
    str
        A copy of text with all Python-literal tokens replaced by their
        JSON equivalents.
    """
    # Replace standalone Python boolean and None keywords only
    text = re.sub(r'\bTrue\b',  'true',  text)
    text = re.sub(r'\bFalse\b', 'false', text)
    text = re.sub(r'\bNone\b',  'null',  text)

    # Replace single-quoted strings with double-quoted strings
    text = re.sub(r"(?<!\\)'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', text)

    # Remove trailing commas before closing braces or brackets (illegal in JSON)
    text = re.sub(r',\s*([\]}])', r'\1', text)

    return text


def try_parse_json(candidate):
    """
    Attempt to parse a string as JSON using three escalating strategies.

    Strategy 1: json.loads() directly.
        Handles clean, strict JSON.

    Strategy 2: normalise_llm_json_text() then json.loads().
        Handles Python-style True/False/None, single-quoted strings,
        and trailing commas.

    Strategy 3: ast.literal_eval().
        Handles valid Python literals (dicts, lists, booleans) that are
        still not parseable as JSON after normalisation.
        Only accepts dict or list results -- rejects bare strings/numbers.

    Parameters
    ----------
    candidate : str
        The string to attempt to parse.

    Returns
    -------
    dict or list
        The successfully parsed Python object.

    Raises
    ------
    ValueError
        If all three strategies fail.
    """
    # Strategy 1: strict JSON parse
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: normalise Python literals, then JSON parse
    try:
        return json.loads(normalise_llm_json_text(candidate))
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 3: Python literal evaluator (safe -- does not execute arbitrary code)
    try:
        result = ast.literal_eval(candidate)
        if isinstance(result, (dict, list)):
            return result
    except (ValueError, SyntaxError):
        pass

    raise ValueError(
        "try_parse_json: all three strategies failed.\n"
        f"Candidate (first 200 chars): {candidate[:200]!r}"
    )


def extract_json(raw_text):
    """
    Extract and parse the first valid JSON object or array from an LLM response.

    Applies four strategies in order, stopping at the first success:

    Strategy A: Parse the full response text directly.
        Works for clean JSON and Python literals (via try_parse_json).

    Strategy B: Extract content from a markdown code fence.
        Handles ```json...```, ```python...```, plain ```...```, and
        variations with extra spaces or uppercase language tags.

    Strategy C: Scan for the first { or [ and parse from there.
        Handles responses with preamble text like "Here is the plan:"
        before the actual JSON.
        Tries whichever bracket appears first so an inner [] inside a
        dict (e.g. "gaps": []) is not confused with the root structure.

    Parameters
    ----------
    raw_text : str
        The full string returned by the LLM. May contain markdown fences,
        preamble text, trailing commentary, Python literals, or any
        combination of the above.

    Returns
    -------
    dict or list
        The parsed Python object.

    Raises
    ------
    ValueError
        If no valid JSON or Python literal can be extracted.
        The message includes the first 300 chars of raw_text for debugging.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("extract_json: LLM returned an empty response.")

    text = raw_text.strip()

    # Strategy A: parse the full text directly
    try:
        return try_parse_json(text)
    except ValueError:
        pass

    # Strategy B: extract from a markdown code fence
    fence_match = re.search(r"```[a-zA-Z\s]*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        inside = fence_match.group(1).strip()
        try:
            return try_parse_json(inside)
        except ValueError:
            pass

    # Strategy C: scan for the outermost { or [ and parse from there
    brace_pos   = text.find('{')
    bracket_pos = text.find('[')

    if brace_pos != -1 or bracket_pos != -1:
        # Try whichever opening bracket appears first in the string
        if brace_pos == -1:
            order = [('[', ']')]
        elif bracket_pos == -1:
            order = [('{', '}')]
        elif brace_pos < bracket_pos:
            order = [('{', '}'), ('[', ']')]
        else:
            order = [('[', ']'), ('{', '}')]

        for open_ch, close_ch in order:
            start = text.find(open_ch)
            if start == -1:
                continue
            end = text.rfind(close_ch)
            if end == -1 or end <= start:
                continue
            candidate = text[start:end + 1]
            try:
                return try_parse_json(candidate)
            except ValueError:
                continue

    raise ValueError(
        "extract_json: all strategies exhausted.\n"
        f"First 300 chars of raw response: {raw_text[:300]!r}"
    )

# =============================================================================
# SECTION 4 -- MODEL LOADING
# Loads the sklearn Decision Tree package from disk.
# The cache variable _MODEL_PKG_CACHE (Section 2) prevents repeated disk reads.
# =============================================================================

# Keys that MUST exist in the model package dict.
_REQUIRED_MODEL_KEYS = {"model", "scaler", "cat_cols", "feature_columns", "dt_threshold"}


def load_model_package():
    """
    Load the model package from disk and return it as a plain dict.

    On the first call the package is read from MODEL_PATH, validated, and
    stored in the module-level _MODEL_PKG_CACHE variable. On every subsequent
    call the cached dict is returned immediately without reading the file again.

    The .pkl file must contain a plain Python dict with at least these keys:
        model            -- fitted sklearn DecisionTreeClassifier
        scaler           -- fitted sklearn StandardScaler
        cat_cols         -- list of str: categorical feature column names
        feature_columns  -- list of str: all post-OHE feature column names
        dt_threshold     -- float: decision threshold (e.g. 0.35)

    Optional keys used when present:
        lr_model         -- logistic regression fallback model
        lr_threshold     -- float: lr_model threshold (default 0.50)
        dataset_info     -- dict: metadata about the training dataset
        dt_metrics       -- dict: model metrics (accuracy, roc_auc, etc.)

    Returns
    -------
    dict
        Flat dict with all model package fields.

    Raises
    ------
    FileNotFoundError
        If MODEL_PATH does not exist on disk.
    TypeError
        If the pickle file does not contain a dict.
    KeyError
        If any required key is missing from the dict.
    """
    global _MODEL_PKG_CACHE

    # Return the cached package if already loaded (replaces @lru_cache)
    if _MODEL_PKG_CACHE is not None:
        return _MODEL_PKG_CACHE

    from pathlib import Path
    path = Path(MODEL_PATH)

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{path.resolve()}'.\n"
            "Set DT_MODEL_PATH env var or place dt_model.pkl in the working directory."
        )

    with open(path, "rb") as fh:
        raw = pickle.load(fh)

    if not isinstance(raw, dict):
        raise TypeError(
            f"Expected a dict inside the pkl file, got {type(raw).__name__}."
        )

    missing = _REQUIRED_MODEL_KEYS - raw.keys()
    if missing:
        raise KeyError(f"Model package is missing required keys: {missing}")

    # Build a clean flat dict -- no wrapper object needed
    pkg = {
        "model":           raw["model"],
        "scaler":          raw["scaler"],
        "cat_cols":        raw["cat_cols"],
        "feature_columns": raw["feature_columns"],
        "dt_threshold":    float(raw["dt_threshold"]),
        "lr_model":        raw.get("lr_model"),
        "lr_threshold":    float(raw.get("lr_threshold", 0.50)),
        "dataset_info":    raw.get("dataset_info", {}),
        "dt_metrics":      raw.get("dt_metrics", {}),
    }

    _MODEL_PKG_CACHE = pkg   # store in module-level cache for reuse

    print(
        "ModelLoader -- Loaded: "
        f"type={type(pkg['model']).__name__}, "
        f"threshold={pkg['dt_threshold']}, "
        f"n_features={len(pkg['feature_columns'])}, "
        f"roc_auc={pkg['dt_metrics'].get('roc_auc')}"
    )
    return pkg


def predict_proba_default(pkg, X_scaled):
    """
    Return P(default=1) for a single pre-scaled feature row.

    Parameters
    ----------
    pkg      : dict          Model package from load_model_package().
    X_scaled : numpy.ndarray Shape (1, n_features), already StandardScaler-transformed.

    Returns
    -------
    float
        Probability that the applicant defaults. Range [0.0, 1.0].
    """
    return float(pkg["model"].predict_proba(X_scaled)[0][1])


def predict_with_threshold(pkg, proba):
    """
    Convert a probability to a binary prediction using the exported dt_threshold.

    Uses the threshold stored in the model package (e.g. 0.35) rather than
    sklearn's default of 0.50. The exported threshold was tuned on the
    validation set to maximise recall for the default class.

    Parameters
    ----------
    pkg   : dict   Model package from load_model_package().
    proba : float  P(default) from predict_proba_default().

    Returns
    -------
    int
        1 if proba >= dt_threshold (predict default -- REJECT).
        0 if proba <  dt_threshold (predict no default -- APPROVE).
    """
    return int(proba >= pkg["dt_threshold"])

# =============================================================================
# SECTION 5 -- PIPELINE STATE
# All pipeline state is kept in a plain dict created by make_state().
# Every phase reads from and writes to this dict by key name.
# =============================================================================

def make_state(applicant_data):
    """
    Create and return a fresh pipeline state dict for one PER run.

    The state dict is the single source of truth across all four phases.
    Phases write their results into specific keys; downstream phases read them.

    State layout
    ------------
    raw_input          -- dict   original applicant data from the caller
    plan               -- list or None   step list from the Planner
    execution_log      -- list   one record per tool call
    ml_output          -- dict or None   result of preprocess_and_predict
    retrieved_rules    -- list or None   accumulated RAG results
    risk_flags         -- dict or None   result of compute_risk_flags
    segment_score      -- dict or None   result of score_applicant_segment
    decision_rationale -- dict or None   result of build_decision_rationale
    reflection         -- dict or None   result of run_reflector
    final_report       -- str or None    narrative from run_reporter
    final_decision     -- str or None    "APPROVE" or "REJECT"
    audit_trail        -- list   timestamped phase-level events
    error_log          -- list   errors that did not crash the pipeline
    reflect_retries    -- int    how many reflector retries have occurred

    Parameters
    ----------
    applicant_data : dict
        Raw applicant feature dict provided by the caller.

    Returns
    -------
    dict
        Initialised state dict. Pass this into run_planner() to start.
    """
    return {
        "raw_input":          applicant_data,
        "plan":               None,
        "execution_log":      [],
        "ml_output":          None,
        "retrieved_rules":    None,
        "risk_flags":         None,
        "segment_score":      None,
        "decision_rationale": None,
        "reflection":         None,
        "final_report":       None,
        "final_decision":     None,
        "audit_trail":        [],
        "error_log":          [],
        "reflect_retries":    0,
    }


def log_event(state, phase, action, detail=None):
    """
    Append a timestamped audit event to state["audit_trail"].

    Parameters
    ----------
    state  : dict   Pipeline state dict. Mutated in place.
    phase  : str    Name of the current phase, e.g. "ORCHESTRATOR".
    action : str    Short label for the event, e.g. "phase_1_planner_start".
    detail : any    Optional extra context. Converted to str, truncated to 300 chars.
    """
    state["audit_trail"].append({
        "phase":     phase,
        "action":    action,
        "detail":    str(detail)[:300] if detail is not None else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def log_tool_call(state, tool_name, args, result, success):
    """
    Append one tool-call record to state["execution_log"].

    Called by dispatch_tool() after every tool invocation.

    Parameters
    ----------
    state     : dict   Pipeline state dict. Mutated in place.
    tool_name : str    Name of the tool that was called.
    args      : dict   Arguments passed to the tool.
    result    : any    Return value from the tool (or error message).
    success   : bool   True if the tool completed without raising an exception.
    """
    state["execution_log"].append({
        "tool":    tool_name,
        "args":    args,
        "result":  result,
        "success": success,
        "ts":      datetime.now(timezone.utc).isoformat(),
    })


def get_tools_called(state):
    """
    Return the set of tool names that completed successfully in this run.

    Parameters
    ----------
    state : dict   Pipeline state dict.

    Returns
    -------
    set of str
        e.g. {"preprocess_and_predict", "compute_risk_flags"}
    """
    return {entry["tool"] for entry in state["execution_log"] if entry["success"]}


def state_to_dict(state):
    """
    Return a JSON-serialisable snapshot of the pipeline state.

    Intended for API responses and UI consumption. The returned dict
    contains all fields from make_state() with no additions or removals.

    Parameters
    ----------
    state : dict   Pipeline state dict.

    Returns
    -------
    dict
        Flat, JSON-safe copy of the state. Pass to json.dumps() directly.
    """
    return {
        "raw_input":          state["raw_input"],
        "plan":               state["plan"],
        "ml_output":          state["ml_output"],
        "retrieved_rules":    state["retrieved_rules"],
        "risk_flags":         state["risk_flags"],
        "segment_score":      state["segment_score"],
        "decision_rationale": state["decision_rationale"],
        "reflection":         state["reflection"],
        "final_report":       state["final_report"],
        "final_decision":     state["final_decision"],
        "execution_log":      state["execution_log"],
        "audit_trail":        state["audit_trail"],
        "error_log":          state["error_log"],
    }

# =============================================================================
# SECTION 6 -- PREPROCESSING HELPERS
# Shared by preprocess_and_predict (Tool 1) and compute_risk_flags (Tool 3).
# Keeping them here ensures both tools apply identical feature engineering.
# =============================================================================

# Translates friendly/short field names to the exact column names the model
# was trained on. Keys not in this map are passed through unchanged.
_ALIAS_MAP = {
    "income":           "person_income($)",
    "age":              "person_age",
    "employment_years": "person_emp_length",
    "home_ownership":   "person_home_ownership",
    "loan_amount":      "loan_amnt($)",
    "interest_rate":    "loan_int_rate",
    "default_on_file":  "cb_person_default_on_file",
    "credit_history":   "cb_person_cred_hist_length",
}

# Dataset median/mode values used to fill missing features.
# Using training-set statistics keeps imputation stable and reproducible.
_DEFAULTS = {
    "person_age":                  27,
    "person_income($)":            48000,
    "person_home_ownership":       "RENT",
    "person_emp_length":           4.0,
    "loan_intent":                 "PERSONAL",
    "loan_amnt($)":                8000,
    "loan_int_rate":               11.0,
    "loan_percent_income":         0.17,
    "cb_person_default_on_file":   "N",
    "cb_person_cred_hist_length":  3,
}


def resolve_aliases(raw_dict):
    """
    Translate friendly field names to the internal column names the model expects.

    Keys not in _ALIAS_MAP are passed through unchanged, so callers may use
    either friendly names or internal names (or a mix of both).

    Parameters
    ----------
    raw_dict : dict
        Applicant data with any combination of friendly and internal keys.

    Returns
    -------
    dict
        New dict with all alias keys replaced by their internal equivalents.
    """
    return {_ALIAS_MAP.get(k, k): v for k, v in raw_dict.items()}


def preprocess_features(resolved_dict, pkg):
    """
    Replicate the training feature-engineering pipeline exactly.

    Applying the same steps in the same order as training is critical.
    Any deviation will cause silent prediction errors.

    Steps
    -----
    1. Fill missing features with dataset defaults from _DEFAULTS.
    2. One-hot encode categorical columns (drop_first=True, same as training).
    3. Reindex to the full post-OHE column list; fill unseen dummies with 0.
    4. StandardScaler.transform() to zero mean / unit variance.

    Parameters
    ----------
    resolved_dict : dict
        Applicant data after resolve_aliases() -- internal column names only.
    pkg : dict
        Model package from load_model_package().

    Returns
    -------
    numpy.ndarray
        Shape (1, n_features). Ready to pass into model.predict_proba().
    """
    # Step 1: complete feature row, filling gaps with training defaults
    row = {col: resolved_dict.get(col, _DEFAULTS[col]) for col in _DEFAULTS}

    # Step 2: one-hot encode -- drop_first=True must match the training call exactly
    df   = pd.DataFrame([row])
    denc = pd.get_dummies(df, columns=pkg["cat_cols"], drop_first=True)

    # Step 3: align to training columns; fill_value=0 handles unseen categories
    aln = denc.reindex(columns=pkg["feature_columns"], fill_value=0)

    # Step 4: apply the fitted StandardScaler
    return pkg["scaler"].transform(aln.values)

# =============================================================================
# SECTION 7 -- TOOL 1: preprocess_and_predict
# =============================================================================

def preprocess_and_predict(applicant_data):
    """
    Run the Decision Tree classifier on raw applicant data and return a verdict.

    This is the primary ML tool. It must be called first in every analysis
    plan because all other tools and the final decision are anchored to the
    probability it returns.

    Internal pipeline
    -----------------
    1. resolve_aliases()       -- translate friendly field names
    2. preprocess_features()   -- fill defaults, OHE, scale
    3. predict_proba_default() -- get P(default) from model
    4. Safety overrides        -- floor probabilities for extreme edge cases
    5. predict_with_threshold()-- convert probability to 0/1 using dt_threshold
    6. Confidence band         -- map probability to a named risk tier

    Safety overrides (applied before threshold classification)
    ----------------------------------------------------------
    income <= 10,000      -> P(default) floored at 0.70
    employment_years == 0 -> P(default) floored at 0.75
    These handle tail cases where training data was sparse.

    Parameters
    ----------
    applicant_data : dict
        Raw applicant features. Accepts both friendly and internal key names.

    Returns
    -------
    dict with keys:
        prediction       -- int    0 = no default predicted, 1 = default predicted
        probability      -- float  P(default) after safety overrides (4 dp)
        confidence_band  -- str    LOW_RISK | MODERATE_RISK | HIGH_RISK | VERY_HIGH_RISK
        decision         -- str    APPROVE | REJECT
        model_threshold  -- float  the dt_threshold used for classification

    On error returns:
        {"error": str, "traceback": str}
    """
    try:
        pkg   = load_model_package()
        res   = resolve_aliases(applicant_data)
        X     = preprocess_features(res, pkg)
        proba = predict_proba_default(pkg, X)

        # Safety overrides for extreme edge cases
        income  = float(res.get("person_income($)",  _DEFAULTS["person_income($)"]))
        emp_len = float(res.get("person_emp_length", _DEFAULTS["person_emp_length"]))

        if income <= 10_000:
            proba = max(proba, 0.70)   # very low income -> floor at HIGH_RISK

        if emp_len == 0:
            proba = max(proba, 0.75)   # zero employment -> floor at VERY_HIGH_RISK

        pred = predict_with_threshold(pkg, proba)

        # Map probability to a named risk tier
        if   proba < 0.30: band = "LOW_RISK"
        elif proba < 0.50: band = "MODERATE_RISK"
        elif proba < 0.75: band = "HIGH_RISK"
        else:              band = "VERY_HIGH_RISK"

        return {
            "prediction":      pred,
            "probability":     round(proba, 4),
            "confidence_band": band,
            "decision":        "REJECT" if pred == 1 else "APPROVE",
            "model_threshold": pkg["dt_threshold"],
        }

    except Exception as exc:
        return {
            "error":     f"preprocess_and_predict: {type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

# =============================================================================
# SECTION 8 -- TOOL 2: retrieve_credit_rules
# ChromaDB in-memory vector store built from a hard-coded policy knowledge base.
# _VECTOR_STORE_CACHE (Section 2) prevents rebuilding the index on every call.
# =============================================================================

# Hard-coded credit policy knowledge base.
# In production this would be loaded from a database or document store.
_CREDIT_RISK_DOCS = [
    "CREDIT POLICY 2.1 ELIGIBILITY: Age 21-65. Min monthly income 25000 salaried, "
    "40000 self-employed. Employment: 6 months salaried, 2 years business. "
    "Credit score: 650 personal, 700 home loans. Failing any = auto-reject.",

    "CREDIT POLICY 3.4 SCORE BANDS: 750-900 Excellent fast-track. "
    "700-749 Good standard underwriting. 650-699 Fair enhanced due diligence higher rate. "
    "600-649 Poor collateral or co-applicant mandatory. Below 600 auto-decline.",

    "CREDIT RISK 5.2 PD THRESHOLDS: PD<3% approve standard rate. "
    "PD 3-7% approve with conditions. PD 7-12% credit committee referral. "
    "PD 12-20% collateral mandatory LTV 60%. PD>20% decline.",

    "UNDERWRITING 7.1 INCOME STABILITY: PSU/Government low risk. MNC low-medium. "
    "Private unlisted medium. Gig high risk. Tenure <6m high risk, 6-24m medium, >24m low. "
    "Cash salary unverifiable. Salary variance >25% MoM triggers review.",

    "UNDERWRITING 7.3 DTI: DTI=Monthly Debt/Net Income. "
    "DTI<=35% low risk. 36-50% medium cap new EMI. 51-60% collateral mandatory. >60% decline. "
    "CC utilization >80% adds 5% effective DTI.",

    "CREDIT POLICY 4.1 DELINQUENCY: 0 DPD 24m clean no adjustment. "
    "1-29 DPD 1-2 times +0.5% surcharge. 30+ DPD in 12m manual review mandatory. "
    "90+ DPD or written-off auto-decline unsecured. Settled = 60 DPD treatment.",

    "CREDIT POLICY 4.7 THIN FILE NTC: <3 tradelines or <12m history = NTC. "
    "Alternative data: utility bills, rental records, GST filings, UPI transactions. "
    "Max initial loan 200000.",

    "COLLATERAL POLICY 9.1 LTV: Residential self-occupied 80%. Rented 70%. Commercial 65%. "
    "Fixed deposits 90%. Listed equity 50%. Gold hallmarked 18K+ 75%. "
    "Third-party empanelled valuer required. Re-valuation every 3 years.",

    "SECTORAL RISK 6.2: HIGH RISK real estate developers, crypto, airlines, hospitality. "
    "MEDIUM RISK retail construction small manufacturing. "
    "LOW RISK IT healthcare FMCG BFSI government entities.",

    "FRAUD INDICATORS 11.1: Address mismatch KYC/bureau. >3 hard inquiries in 30 days. "
    "Shared mobile/email unrelated applicants. Salary-TDS mismatch. "
    "Document metadata inconsistencies. Large cash deposits before income period.",

    "EARLY WARNING 12.3: Level1 EMI delayed 1-15 days 2 consecutive months or score -30pts. "
    "Level2 EMI delayed 16-30 days NACH bounce 20% salary drop. "
    "Level3 30+DPD second NACH bounce legal proceedings NPA at another lender.",

    "RBI IRAC 13.1: Standard Asset <90 days overdue. Sub-Standard NPA <12m 15% provisioning. "
    "Doubtful 1yr 25% secured+100% unsecured. Doubtful 1-3yr 40%+100%. Loss 100%. "
    "Annual RBI credit risk certification mandatory.",

    "IFRS9 ECL 14.1: ECL=PD*LGD*EAD. Stage1 no risk increase 12m ECL. "
    "Stage2 significant risk lifetime ECL. Stage3 credit-impaired lifetime ECL. "
    "Stage2 triggers: 30+DPD score drop 40pts restructuring macro overlay.",

    "REPAYMENT VINTAGE 4.4: Positive pre-closure no penalty 0DPD 24m auto-debit. "
    "Negative NACH bounce restructuring requests balance transfers revolving debt. "
    "Vintage score 0-100 feeds PD model 18% weight.",
]


def get_vector_store():
    """
    Build and return the ChromaDB in-memory vector store.

    On the first call the store is built by embedding all _CREDIT_RISK_DOCS
    strings. The collection object is then stored in _VECTOR_STORE_CACHE and
    returned on all subsequent calls without rebuilding.

    This function replaces a @lru_cache decorator with a plain global variable.

    Embedding model: all-MiniLM-L6-v2 (lightweight, fast, good quality).

    Returns
    -------
    chromadb.Collection
        Ready to call .query(query_texts=[...], n_results=N) on.
    """
    global _VECTOR_STORE_CACHE

    # Return the cached store if already built (replaces @lru_cache)
    if _VECTOR_STORE_CACHE is not None:
        return _VECTOR_STORE_CACHE

    ef     = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.Client()

    # Delete any leftover collection from a previous run (important in Colab
    # where the Python process persists across cell re-runs)
    try:
        client.delete_collection("credit_risk_kb_per")
    except Exception:
        pass

    col = client.create_collection("credit_risk_kb_per", embedding_function=ef)
    col.add(
        documents=_CREDIT_RISK_DOCS,
        ids=[f"doc_{i}" for i in range(len(_CREDIT_RISK_DOCS))],
    )

    print(f"RAG -- Indexed {len(_CREDIT_RISK_DOCS)} policy documents.")
    _VECTOR_STORE_CACHE = col   # cache for reuse
    return col


def retrieve_credit_rules(query, top_k=3):
    """
    Semantic search over the credit policy knowledge base.

    Uses cosine similarity in embedding space to retrieve the most relevant
    policy clauses for a given risk-dimension query.

    Use a query that is SPECIFIC to the applicant's risk profile. Examples:
        "high DTI rejection policy"
        "prior default bureau treatment 60 DPD"
        "thin credit file NTC alternative data"

    This tool may be called at most twice per run with different queries.

    Parameters
    ----------
    query : str
        Specific risk-dimension query string.
    top_k : int
        Number of policy clauses to retrieve. Clamped to [1, len(KB)].

    Returns
    -------
    dict with keys:
        rules      -- list of {"rule": str, "relevance": float} dicts
        query_used -- str   the exact query string sent
        count      -- int   number of rules returned

    On error returns:
        {"error": str, "rules": [], "count": 0}
    """
    try:
        store = get_vector_store()
        n     = min(max(1, top_k), len(_CREDIT_RISK_DOCS))
        res   = store.query(query_texts=[query], n_results=n)

        # ChromaDB returns L2 distances (smaller = more similar).
        # Convert to a [0,1] relevance score for readability.
        rules = [
            {"rule": doc.strip(), "relevance": round(1.0 - dist, 4)}
            for doc, dist in zip(res["documents"][0], res["distances"][0])
        ]

        return {"rules": rules, "query_used": query, "count": len(rules)}

    except Exception as exc:
        return {
            "error":  f"retrieve_credit_rules: {type(exc).__name__}: {exc}",
            "rules":  [],
            "count":  0,
        }

# =============================================================================
# SECTION 9 -- TOOL 3: compute_risk_flags
# =============================================================================

def compute_risk_flags(applicant_data):
    """
    Run deterministic hard-coded policy checks on applicant data.

    These checks encode institutional credit policy rules and operate
    independently of the ML model. They are auditable and explainable.

    Checks performed (in order)
    ----------------------------
    1. Monthly income below floor      (CRITICAL if <10k, HIGH if <25k)
    2. Employment tenure below minimum (CRITICAL if 0 yrs, HIGH if <6 months)
    3. Loan-to-income ratio too high   (CRITICAL if >60%, HIGH if >40%)
    4. Thin credit file                (MEDIUM if history <2 years)
    5. Prior default on bureau file    (HIGH -- treated as 60 DPD equivalent)
    6. Interest rate in sub-prime band (MEDIUM if rate >18%)
    7. Renter with high debt exposure  (MEDIUM if renting AND LPI >35%)

    Severity scoring
    ----------------
    CRITICAL flag = +3 points
    HIGH flag     = +2 points
    MEDIUM flag   = +1 point
    Overall: >=5 -> CRITICAL, >=3 -> HIGH, >=1 -> MEDIUM, else LOW

    Parameters
    ----------
    applicant_data : dict
        Raw applicant features. Alias names are accepted.

    Returns
    -------
    dict with keys:
        flags          -- list of {"flag": str, "detail": str, "severity": str}
        severity       -- str   CRITICAL | HIGH | MEDIUM | LOW
        severity_score -- int   total weighted score
        flag_count     -- int   number of flags raised

    On error returns:
        {"error": str, "flags": [], "severity": "UNKNOWN", "flag_count": 0}
    """
    try:
        res = resolve_aliases(applicant_data)

        income  = float(res.get("person_income($)",          _DEFAULTS["person_income($)"]))
        emp     = float(res.get("person_emp_length",          _DEFAULTS["person_emp_length"]))
        loan    = float(res.get("loan_amnt($)",               _DEFAULTS["loan_amnt($)"]))
        rate    = float(res.get("loan_int_rate",              _DEFAULTS["loan_int_rate"]))
        lpi     = float(res.get("loan_percent_income",
                                 round(loan / max(income, 1), 4)))
        dof     = str(res.get("cb_person_default_on_file",
                               _DEFAULTS["cb_person_default_on_file"])).upper()
        hist    = float(res.get("cb_person_cred_hist_length",
                                 _DEFAULTS["cb_person_cred_hist_length"]))
        home    = str(res.get("person_home_ownership",
                               _DEFAULTS["person_home_ownership"])).upper()
        monthly = income / 12.0

        flags = []
        score = 0

        # Check 1: Monthly income
        if monthly < 10_000:
            flags.append({"flag": "INCOME_BELOW_MINIMUM",
                          "detail": f"Monthly income {monthly:,.0f} is below the 10,000 hard floor",
                          "severity": "CRITICAL"})
            score += 3
        elif monthly < 25_000:
            flags.append({"flag": "INCOME_LOW",
                          "detail": f"Monthly income {monthly:,.0f} is below the 25,000 preferred minimum",
                          "severity": "HIGH"})
            score += 2

        # Check 2: Employment tenure
        if emp == 0:
            flags.append({"flag": "NO_EMPLOYMENT_HISTORY",
                          "detail": "Employment length is zero -- income cannot be verified",
                          "severity": "CRITICAL"})
            score += 3
        elif emp < 0.5:
            flags.append({"flag": "INSUFFICIENT_EMPLOYMENT_TENURE",
                          "detail": f"Employment {emp:.1f} yr is below the 6-month minimum",
                          "severity": "HIGH"})
            score += 2

        # Check 3: Loan-to-income ratio
        if lpi > 0.60:
            flags.append({"flag": "LOAN_PERCENT_INCOME_CRITICAL",
                          "detail": f"Loan is {lpi:.0%} of annual income -- exceeds 60% DTI ceiling",
                          "severity": "CRITICAL"})
            score += 3
        elif lpi > 0.40:
            flags.append({"flag": "LOAN_PERCENT_INCOME_HIGH",
                          "detail": f"Loan is {lpi:.0%} of annual income -- exceeds 40% caution threshold",
                          "severity": "HIGH"})
            score += 2

        # Check 4: Credit history length
        if hist < 2:
            flags.append({"flag": "THIN_CREDIT_FILE",
                          "detail": f"Credit history is {hist:.0f} year(s) -- NTC protocol applies",
                          "severity": "MEDIUM"})
            score += 1

        # Check 5: Prior default on bureau file
        if dof == "Y":
            flags.append({"flag": "PRIOR_DEFAULT_ON_FILE",
                          "detail": "Prior default on bureau -- treated as 60 DPD equivalent",
                          "severity": "HIGH"})
            score += 2

        # Check 6: Sub-prime interest rate
        if rate > 18.0:
            flags.append({"flag": "HIGH_INTEREST_RATE",
                          "detail": f"Interest rate {rate}% is in the sub-prime band (>18%)",
                          "severity": "MEDIUM"})
            score += 1

        # Check 7: Renter with high debt exposure
        if home == "RENT" and lpi > 0.35:
            flags.append({"flag": "RENTER_HIGH_DEBT_EXPOSURE",
                          "detail": "Renting + LPI above 35% creates dual payment pressure",
                          "severity": "MEDIUM"})
            score += 1

        if   score >= 5: overall = "CRITICAL"
        elif score >= 3: overall = "HIGH"
        elif score >= 1: overall = "MEDIUM"
        else:            overall = "LOW"

        return {
            "flags":          flags,
            "severity":       overall,
            "severity_score": score,
            "flag_count":     len(flags),
        }

    except Exception as exc:
        return {
            "error":      f"compute_risk_flags: {type(exc).__name__}: {exc}",
            "flags":      [],
            "severity":   "UNKNOWN",
            "flag_count": 0,
        }

# =============================================================================
# SECTION 10 -- TOOL 4: score_applicant_segment
# =============================================================================

# Percentile benchmarks from the 32,576-row training dataset.
# Each inner dict maps a percentile label to the metric value at that percentile.
_PEER_BENCHMARKS = {
    "income":             {"p10": 18000,  "p25": 30000, "p50": 48000, "p75": 70000,  "p90": 100000},
    "loan_amnt":          {"p10": 2500,   "p25": 5000,  "p50": 8000,  "p75": 14000,  "p90": 20000},
    "loan_int_rate":      {"p10": 7.0,    "p25": 9.5,   "p50": 11.0,  "p75": 14.0,   "p90": 18.0},
    "loan_percent_income":{"p10": 0.06,   "p25": 0.10,  "p50": 0.17,  "p75": 0.27,   "p90": 0.40},
    "emp_length":         {"p10": 0.0,    "p25": 1.0,   "p50": 4.0,   "p75": 8.0,    "p90": 10.0},
}


def percentile_rank(value, benchmarks):
    """
    Return the approximate percentile rank (0-100) of a value against benchmarks.

    Walks sorted percentile thresholds and returns the label of the first
    bracket the value falls into. Returns 99 if it exceeds all thresholds.

    Parameters
    ----------
    value      : float
        The metric value to rank.
    benchmarks : dict
        Maps percentile label strings ("p10", "p25", etc.) to numeric thresholds.

    Returns
    -------
    int
        Percentile rank: 10, 25, 50, 75, 90, or 99.
    """
    sorted_brackets = sorted(benchmarks.items(), key=lambda item: item[1])
    for label, threshold in sorted_brackets:
        rank = int(label[1:])   # "p10" -> 10
        if value <= threshold:
            return rank
    return 99


def score_applicant_segment(applicant_data):
    """
    Position the applicant against training-population peer-group benchmarks.

    Computes individual percentile ranks for five metrics, then combines them
    into a weighted composite risk score and maps it to a named segment.

    Composite score formula (higher = riskier, range 0-100)
    --------------------------------------------------------
    (100 - income_pct)  * 0.30   low income increases risk
    + int_rate_pct      * 0.25   high rate increases risk
    + dti_proxy_pct     * 0.25   high DTI increases risk
    + (100 - emp_pct)   * 0.20   short tenure increases risk

    Segment thresholds
    ------------------
    composite < 30  -> PRIME
    30-49           -> NEAR_PRIME
    50-69           -> SUBPRIME
    >= 70           -> DEEP_SUBPRIME

    Parameters
    ----------
    applicant_data : dict
        Raw applicant features. Alias names are accepted.

    Returns
    -------
    dict with keys:
        percentiles           -- dict {metric_name: percentile_rank_int}
        composite_risk_score  -- int  0 (lowest) to 100 (highest risk)
        segment               -- str  PRIME | NEAR_PRIME | SUBPRIME | DEEP_SUBPRIME
        interpretation        -- str  plain-language summary

    On error returns:
        {"error": str}
    """
    try:
        res = resolve_aliases(applicant_data)

        income = float(res.get("person_income($)",    _DEFAULTS["person_income($)"]))
        loan   = float(res.get("loan_amnt($)",        _DEFAULTS["loan_amnt($)"]))
        rate   = float(res.get("loan_int_rate",       _DEFAULTS["loan_int_rate"]))
        lpi    = float(res.get("loan_percent_income", round(loan / max(income, 1), 4)))
        emp    = float(res.get("person_emp_length",   _DEFAULTS["person_emp_length"]))

        pctls = {
            "income_pct":      percentile_rank(income, _PEER_BENCHMARKS["income"]),
            "loan_amount_pct": percentile_rank(loan,   _PEER_BENCHMARKS["loan_amnt"]),
            "int_rate_pct":    percentile_rank(rate,   _PEER_BENCHMARKS["loan_int_rate"]),
            "dti_proxy_pct":   percentile_rank(lpi,    _PEER_BENCHMARKS["loan_percent_income"]),
            "emp_length_pct":  percentile_rank(emp,    _PEER_BENCHMARKS["emp_length"]),
        }

        composite = int(
            (100 - pctls["income_pct"])       * 0.30
            + pctls["int_rate_pct"]           * 0.25
            + pctls["dti_proxy_pct"]          * 0.25
            + (100 - pctls["emp_length_pct"]) * 0.20
        )

        if   composite < 30: segment = "PRIME"
        elif composite < 50: segment = "NEAR_PRIME"
        elif composite < 70: segment = "SUBPRIME"
        else:                segment = "DEEP_SUBPRIME"

        interpretation = (
            f"Income at the {pctls['income_pct']}th percentile of the training population. "
            f"Interest rate at the {pctls['int_rate_pct']}th percentile. "
            f"DTI proxy at the {pctls['dti_proxy_pct']}th percentile. "
            f"Composite risk score {composite}/100 -- segment: {segment}."
        )

        return {
            "percentiles":          pctls,
            "composite_risk_score": composite,
            "segment":              segment,
            "interpretation":       interpretation,
        }

    except Exception as exc:
        return {"error": f"score_applicant_segment: {type(exc).__name__}: {exc}"}

# =============================================================================
# SECTION 11 -- TOOL 5: build_decision_rationale  (TERMINAL TOOL)
# =============================================================================

def build_decision_rationale(
    decision,
    risk_level,
    probability,
    segment,
    primary_factors,
    policy_citations,
    conditions,
    override_reason="",
):
    """
    Assemble the structured final decision object from all prior evidence.

    This is the TERMINAL tool -- calling it signals to the Executor loop
    that analysis is complete. Its output dict is what the Reporter uses
    to generate the narrative.

    Parameters
    ----------
    decision         : str
        Final credit decision. Must be "APPROVE" or "REJECT".
    risk_level       : str
        Risk tier. "LOW", "MODERATE", "HIGH", or "VERY_HIGH".
    probability      : float
        P(default) from the ML tool. Range [0.0, 1.0].
    segment          : str
        "PRIME", "NEAR_PRIME", "SUBPRIME", or "DEEP_SUBPRIME".
    primary_factors  : list of str
        2-4 strings describing the top factors driving this decision.
    policy_citations : list of str
        Policy section references consulted (e.g. "CREDIT RISK 5.2 PD THRESHOLDS").
    conditions       : list of str
        For APPROVE: conditions attached to the approval.
        For REJECT: remediation steps the applicant can take.
    override_reason  : str
        If a policy flag overrides the ML model decision, explain why here.
        Pass empty string if no override occurred (the default).

    Returns
    -------
    dict
        Structured decision object with all fields needed by run_reporter().

    On error returns:
        {"error": str}
    """
    try:
        ts = datetime.now(timezone.utc).isoformat()

        return {
            "decision":         decision.upper(),
            "risk_level":       risk_level.upper(),
            "probability":      round(float(probability), 4),
            "probability_pct":  f"{float(probability):.1%}",
            "segment":          segment.upper(),
            "primary_factors":  primary_factors,
            "policy_citations": policy_citations,
            "conditions":       conditions,
            "override_reason":  override_reason if override_reason else None,
            "generated_at":     ts,
            "model":            "DecisionTree (ROC-AUC 0.89, threshold=0.35)",
            "disclaimer": (
                "AI-generated analysis. Requires qualified credit officer review. "
                "Refer to RBI IRAC guidelines and internal credit policy."
            ),
        }

    except Exception as exc:
        return {"error": f"build_decision_rationale: {type(exc).__name__}: {exc}"}

# =============================================================================
# SECTION 12 -- TOOL REGISTRY AND JSON SCHEMA
# =============================================================================

# Plain dict: tool name -> callable function.
# The Executor uses this for dispatch -- no reflection or import tricks needed.
TOOL_REGISTRY = {
    "preprocess_and_predict":   preprocess_and_predict,
    "retrieve_credit_rules":    retrieve_credit_rules,
    "compute_risk_flags":       compute_risk_flags,
    "score_applicant_segment":  score_applicant_segment,
    "build_decision_rationale": build_decision_rationale,
}

# OpenAI-compatible tool schema. The LLM reads this to know what tools exist,
# what each parameter means, and which parameters are required.
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "preprocess_and_predict",
            "description": (
                "Run the Decision Tree ML model on applicant data. "
                "Returns P(default), confidence band, and preliminary APPROVE/REJECT. "
                "MUST be called FIRST in every analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_data": {
                        "type": "object",
                        "description": (
                            "Raw applicant feature dict. Accepted keys: "
                            "income, employment_years, person_age, "
                            "person_home_ownership (RENT/OWN/MORTGAGE/OTHER), "
                            "loan_amnt($), loan_int_rate, loan_percent_income, "
                            "loan_intent, cb_person_default_on_file (Y/N), "
                            "cb_person_cred_hist_length."
                        ),
                    }
                },
                "required": ["applicant_data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_credit_rules",
            "description": (
                "Semantic search over the credit policy knowledge base. "
                "Use a query SPECIFIC to this applicant's risk profile. "
                "May be called at most twice with DIFFERENT queries per run."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type":        "string",
                        "description": "Specific risk-dimension query string.",
                    },
                    "top_k": {
                        "type":        "integer",
                        "description": "Number of policy clauses to retrieve (1-5). Default: 3.",
                        "default":     3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_risk_flags",
            "description": (
                "Run deterministic hard-coded policy checks. "
                "Call when ML probability is 0.35-0.65, OR when any of: "
                "employment_years<1, income<30000, cb_person_default_on_file=Y, "
                "loan_percent_income>0.4."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_data": {
                        "type":        "object",
                        "description": "Same applicant dict passed to preprocess_and_predict.",
                    }
                },
                "required": ["applicant_data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_applicant_segment",
            "description": (
                "Compute peer-group percentile ranks and segment classification. "
                "Required for every applicant. Call after preprocess_and_predict."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_data": {
                        "type":        "object",
                        "description": "Same applicant dict.",
                    }
                },
                "required": ["applicant_data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_decision_rationale",
            "description": (
                "Assemble the structured decision JSON. "
                "MUST be called LAST -- it terminates the executor loop. "
                "The decision field MUST match the ML model output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "decision":         {"type": "string", "enum": ["APPROVE", "REJECT"]},
                    "risk_level":       {"type": "string", "enum": ["LOW", "MODERATE", "HIGH", "VERY_HIGH"]},
                    "probability":      {"type": "number", "description": "P(default) 0.0-1.0"},
                    "segment":          {"type": "string", "enum": ["PRIME", "NEAR_PRIME", "SUBPRIME", "DEEP_SUBPRIME"]},
                    "primary_factors":  {"type": "array",  "items": {"type": "string"}, "description": "2-4 top factors."},
                    "policy_citations": {"type": "array",  "items": {"type": "string"}, "description": "Referenced policy sections."},
                    "conditions":       {"type": "array",  "items": {"type": "string"}, "description": "Approval conditions or remediation."},
                    "override_reason":  {"type": "string", "description": "Why policy overrides the model. Empty string if no override."},
                },
                "required": [
                    "decision", "risk_level", "probability", "segment",
                    "primary_factors", "policy_citations", "conditions", "override_reason",
                ],
            },
        },
    },
]

# =============================================================================
# SECTION 13 -- PHASE 1: PLANNER AGENT
# =============================================================================

# System prompt for the Planner LLM.
# CRITICAL: Asks for {"steps":[...]} object -- NOT a bare array -- so that
# response_format=json_object mode works correctly (it only guarantees objects).
_PLANNER_SYSTEM = """You are a Credit Risk Analysis Planner.
Given raw applicant data, output a structured analysis plan.

Respond with a JSON OBJECT in exactly this shape:
{
  "steps": [
    {"step": 1, "action": "preprocess_and_predict", "reason": "..."},
    {"step": 2, "action": "score_applicant_segment",  "reason": "..."},
    ...
  ]
}

Each item in "steps" must have:
  step   : integer, sequential from 1
  action : one of [preprocess_and_predict, score_applicant_segment,
                   retrieve_credit_rules, compute_risk_flags,
                   build_decision_rationale]
  reason : string describing why this step is needed for this applicant
  query  : string -- ONLY for retrieve_credit_rules steps; must be specific

Planning rules:
1. preprocess_and_predict MUST be step 1.
2. score_applicant_segment MUST be included (step 2 or 3).
3. Include compute_risk_flags if ANY of these apply:
   income < 30000 | employment_years < 1 | default_on_file = Y | loan_percent_income > 0.4
4. Include retrieve_credit_rules at least once with a SPECIFIC query.
   Add a second retrieve step only if the applicant has multiple distinct risk dimensions.
5. build_decision_rationale MUST be the last step.

STRICT OUTPUT RULES -- violations will break the pipeline:
- Use double quotes for all strings. Never single quotes.
- Use true / false / null -- never True / False / None.
- No trailing commas. No markdown fences. No preamble text.
"""


def run_planner(applicant_data, groq_client, verbose=True):
    """
    Phase 1: Ask the Planner LLM to produce a structured analysis plan.

    The Planner reads the applicant data and writes an explicit step-by-step
    plan before any tools are called. This separates the decision of WHAT
    to do from actually DOING it.

    If the LLM call fails or returns unparseable output, a hard-coded safe
    default plan is returned so the pipeline never blocks on planner errors.

    LLM settings
    ------------
    temperature=0.0             -- deterministic; plans should not vary
    response_format=json_object -- forces valid JSON at the API level,
                                   preventing Python-literal output (True/False/None)

    Parameters
    ----------
    applicant_data : dict
        Raw applicant feature dict.
    groq_client : Groq
        Authenticated Groq client instance.
    verbose : bool
        If True, print the plan to stdout after generation.

    Returns
    -------
    list of dict
        Each dict has keys: step (int), action (str), reason (str),
        and optionally query (str) for retrieve_credit_rules steps.
    """
    if verbose:
        print("\n" + "-" * 66)
        print("  PHASE 1 -- PLANNER")

    # Pass applicant data as valid JSON so the LLM sees structured input
    user_prompt = (
        "Produce an analysis plan for this loan applicant.\n"
        "Applicant data:\n"
        + json.dumps(applicant_data, indent=2, default=str)
    )

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL_FAST,
            messages=[
                {"role": "system", "content": _PLANNER_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
            # JSON mode: forces the model to emit a valid JSON object.
            # Eliminates Python literals (True/False/None) and single-quoted
            # strings at the API level before our code ever sees the response.
            response_format={"type": "json_object"},
        )
        raw    = resp.choices[0].message.content or ""
        parsed = extract_json(raw)

        # The prompt asks for {"steps": [...]}.
        # Also accept a bare list in case an older model ignores the wrapper.
        if isinstance(parsed, dict) and "steps" in parsed:
            plan = parsed["steps"]
        elif isinstance(parsed, list):
            plan = parsed
        else:
            raise ValueError(
                f"Planner returned unexpected shape: {type(parsed).__name__}"
            )

        if verbose:
            print(f"  Plan ({len(plan)} steps):")
            for step in plan:
                q = f" | query: {step.get('query', '')!r}" if "query" in step else ""
                print(f"    Step {step.get('step','?')}: {step.get('action','?')}{q}")
                print(f"      reason: {step.get('reason', '')}")

        return plan

    except Exception as exc:
        # Planner failures MUST NOT crash the pipeline.
        # Return a safe default plan covering all required tools.
        if verbose:
            print(f"  Planner error: {exc}. Falling back to default plan.")

        return [
            {"step": 1, "action": "preprocess_and_predict",  "reason": "Baseline ML risk score"},
            {"step": 2, "action": "score_applicant_segment",  "reason": "Peer-group benchmarking"},
            {"step": 3, "action": "compute_risk_flags",       "reason": "Deterministic policy checks"},
            {"step": 4, "action": "retrieve_credit_rules",    "reason": "Policy grounding",
             "query": "credit risk default probability rejection threshold"},
            {"step": 5, "action": "build_decision_rationale", "reason": "Terminal step -- assemble decision"},
        ]

# =============================================================================
# SECTION 14 -- PHASE 2: EXECUTOR AGENT
# =============================================================================

# System prompt for the Executor LLM.
_EXECUTOR_SYSTEM = """You are a Credit Risk Executor Agent.
You have been given an analysis plan and a set of tools.
Execute the plan steps in the given order.

Rules:
1. Follow the plan step order exactly.
2. You may call retrieve_credit_rules a second time with a DIFFERENT query ONLY
   if you discover a risk dimension not covered by the plan.
3. Do not call any single tool more than twice.
4. The decision field in build_decision_rationale MUST match the ML model output.
5. build_decision_rationale is the TERMINAL tool -- call it last and only once.
"""


def dispatch_tool(tool_name, args, state):
    """
    Call a named tool, write its result into state, and return (json_str, success).

    This is the single dispatch point for all tool calls. It catches every
    exception so the Executor loop is never interrupted by a tool failure.
    Errors are recorded in state["error_log"] and execution continues.

    State mutations by tool name
    ----------------------------
    preprocess_and_predict   -> state["ml_output"]
    retrieve_credit_rules    -> state["retrieved_rules"]  (accumulated, deduplicated)
    compute_risk_flags       -> state["risk_flags"]
    score_applicant_segment  -> state["segment_score"]
    build_decision_rationale -> state["decision_rationale"] and state["final_decision"]

    Parameters
    ----------
    tool_name : str
        Must be a key in TOOL_REGISTRY.
    args : dict
        Keyword arguments to pass to the tool function.
    state : dict
        Pipeline state dict. Mutated in place.

    Returns
    -------
    tuple (str, bool)
        str  -- JSON string of the tool result (or an error dict).
        bool -- True on success, False on any exception.
    """
    try:
        fn = TOOL_REGISTRY.get(tool_name)
        if fn is None:
            raise ValueError(
                f"Unknown tool '{tool_name}'. "
                f"Valid tools: {list(TOOL_REGISTRY)}"
            )

        result = fn(**args)

        # Route the result into the correct state field
        if tool_name == "preprocess_and_predict":
            state["ml_output"] = result

        elif tool_name == "retrieve_credit_rules":
            new_rules = result.get("rules", [])
            if state["retrieved_rules"] is None:
                state["retrieved_rules"] = new_rules        # first call
            else:
                # Append only rules not already in the accumulated list
                existing = {r["rule"] for r in state["retrieved_rules"]}
                state["retrieved_rules"] += [
                    r for r in new_rules if r["rule"] not in existing
                ]

        elif tool_name == "compute_risk_flags":
            state["risk_flags"] = result

        elif tool_name == "score_applicant_segment":
            state["segment_score"] = result

        elif tool_name == "build_decision_rationale":
            state["decision_rationale"] = result
            state["final_decision"]     = result.get("decision")

        log_tool_call(state, tool_name, args, result, success=True)
        return json.dumps(result, default=str), True

    except Exception as exc:
        error_msg = f"{tool_name} failed: {type(exc).__name__}: {exc}"
        state["error_log"].append({
            "tool":  tool_name,
            "error": error_msg,
            "tb":    traceback.format_exc(),
        })
        log_tool_call(state, tool_name, args, error_msg, success=False)
        return json.dumps({"error": error_msg}), False


def run_executor(state, groq_client, verbose=True):
    """
    Phase 2: Execute the analysis plan using a Groq tool-calling loop.

    Algorithm
    ---------
    1. Build an initial message list from the system prompt + plan + applicant data.
    2. Call the Groq API; the LLM responds with one or more tool calls.
    3. Execute each tool via dispatch_tool(); append results to messages.
    4. Repeat until build_decision_rationale is called (terminal condition)
       OR MAX_EXECUTOR_ITERS iterations are exhausted.

    The messages list grows with each round-trip so the LLM has full context
    of everything it has already done.

    Parameters
    ----------
    state : dict
        Pipeline state dict. Mutated in place by dispatch_tool().
    groq_client : Groq
        Authenticated Groq client.
    verbose : bool
        If True, print each tool call and its result.

    Returns
    -------
    dict
        The same state dict (mutated in place; returned for convenience).
    """
    if verbose:
        print("\n" + "-" * 66)
        print("  PHASE 2 -- EXECUTOR")

    plan_json = json.dumps(state["plan"], indent=2, default=str)
    data_json = json.dumps(state["raw_input"], indent=2, default=str)

    user_msg = (
        "Execute this analysis plan for the applicant.\n"
        f"Plan:\n{plan_json}\n\n"
        f"Applicant data:\n{data_json}"
    )

    messages = [
        {"role": "system", "content": _EXECUTOR_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]

    for iteration in range(MAX_EXECUTOR_ITERS):
        log_event(state, "EXECUTOR", f"iteration_{iteration + 1}")

        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL_STRONG,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.0,
                max_tokens=2048,
            )
        except Exception as exc:
            state["error_log"].append({
                "phase": "EXECUTOR",
                "iter":  iteration + 1,
                "error": str(exc),
            })
            if verbose:
                print(f"  Groq API error on iteration {iteration + 1}: {exc}")
            break

        msg           = resp.choices[0].message
        finish_reason = resp.choices[0].finish_reason

        # Build the assistant message entry for the conversation history
        assistant_entry = {
            "role":    "assistant",
            "content": msg.content or "",
        }
        if msg.tool_calls:
            # Groq requires tool_calls to be echoed back in the assistant turn
            assistant_entry["tool_calls"] = [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        # Stop if the LLM chose not to call any tool
        if finish_reason == "stop" or not msg.tool_calls:
            if verbose:
                print(f"  Executor finished. finish_reason={finish_reason}")
            break

        # Process every tool call returned in this iteration
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            tool_id   = tc.id

            # Parse tool arguments via extract_json -- handles any LLM quirk
            try:
                args = extract_json(tc.function.arguments or "{}")
                if not isinstance(args, dict):
                    args = {}
            except (ValueError, Exception):
                args = {}

            if verbose:
                # json.dumps keeps output valid -- prevents Colab JS JSON.parse crash
                print(f"  -> {tool_name}")
                print(f"     args: {json.dumps(args, default=str)[:200]}")

            result_str, ok = dispatch_tool(tool_name, args, state)

            if verbose:
                # Prefix "result:" so the line never starts with { or [
                # (which would trigger Colab's JS frontend JSON.parse heuristic)
                status = "OK" if ok else "ERROR"
                print(f"     {status}: {result_str[:220]}")

            messages.append({
                "role":         "tool",
                "tool_call_id": tool_id,
                "name":         tool_name,
                "content":      result_str,
            })

            # Terminal condition
            if tool_name == "build_decision_rationale" and ok:
                if verbose:
                    print("  Decision rationale built -- executor complete.")
                return state

    if state["decision_rationale"] is None:
        state["error_log"].append({
            "phase": "EXECUTOR",
            "error": (
                f"Reached MAX_EXECUTOR_ITERS ({MAX_EXECUTOR_ITERS}) "
                "without calling build_decision_rationale."
            ),
        })

    return state

# =============================================================================
# SECTION 15 -- PHASE 3: REFLECTOR AGENT
# =============================================================================

# System prompt for the Reflector LLM.
# Asks for a JSON OBJECT so response_format=json_object works correctly.
_REFLECTOR_SYSTEM = """You are a Credit Risk Analysis Auditor.
Review the execution log and decide whether the analysis is complete and consistent.

Check all six criteria:
1. ML prediction obtained    -- preprocess_and_predict called successfully.
2. Segment scoring done      -- score_applicant_segment called successfully.
3. Policy rules retrieved    -- retrieve_credit_rules called at least once.
4. Risk flags computed       -- required if income<30000, emp<1yr, default=Y, or DTI>0.4.
5. Decision rationale built  -- build_decision_rationale called successfully.
6. Decision consistency      -- decision in rationale matches the ML model prediction.

Respond with a JSON OBJECT with EXACTLY these five keys:
{
  "pass":           true,
  "gaps":           [],
  "retry_steps":    [],
  "consistency_ok": true,
  "notes":          "one-line audit summary"
}

STRICT OUTPUT RULES -- violations break the pipeline:
- Double quotes for ALL strings. Never single quotes.
- true / false -- never True / False.
- gaps and retry_steps must always be arrays (empty if nothing to report).
- No trailing commas. No markdown fences. No preamble text.
"""


def run_reflector(state, groq_client, verbose=True):
    """
    Phase 3: Audit the Executor's output for completeness and consistency.

    Sends a compact summary (tools called, their outputs, any errors) to the
    Reflector LLM. The LLM returns a pass/fail verdict with gaps and suggested
    retry tool names.

    If the Reflector LLM itself fails for any reason, defaults to pass=True
    so a reflector outage never blocks the pipeline.

    LLM settings
    ------------
    temperature=0.0             -- deterministic auditing
    response_format=json_object -- forces valid JSON; prevents Python literals

    Parameters
    ----------
    state : dict
        Pipeline state dict. Read-only in this function.
    groq_client : Groq
        Authenticated Groq client.
    verbose : bool
        If True, print the reflection verdict.

    Returns
    -------
    dict with keys:
        pass           -- bool   True if all checks passed
        gaps           -- list   descriptions of any gaps found
        retry_steps    -- list   tool names to re-run
        consistency_ok -- bool   True if decision matches ML output
        notes          -- str    one-line audit summary
    """
    if verbose:
        print("\n" + "-" * 66)
        print("  PHASE 3 -- REFLECTOR")

    # Compact JSON-safe summary of what the Executor did
    summary = {
        "tools_called":       list(get_tools_called(state)),
        "ml_output":          state["ml_output"],
        "segment_score":      state["segment_score"],
        "risk_flags":         state["risk_flags"],
        "rag_rule_count":     len(state["retrieved_rules"] or []),
        "decision_rationale": state["decision_rationale"],
        "applicant_data":     state["raw_input"],
        "errors":             state["error_log"],
    }

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL_FAST,
            messages=[
                {"role": "system", "content": _REFLECTOR_SYSTEM},
                {
                    "role":    "user",
                    "content": (
                        "Audit this execution:\n"
                        + json.dumps(summary, default=str, indent=2)
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=512,
            # JSON mode: forces valid JSON -- prevents Python booleans/single quotes
            response_format={"type": "json_object"},
        )
        raw        = resp.choices[0].message.content or ""
        reflection = extract_json(raw)

    except Exception as exc:
        reflection = {
            "pass":           True,
            "gaps":           [],
            "retry_steps":    [],
            "consistency_ok": True,
            "notes":          f"Reflector error (defaulting to pass): {exc}",
        }

    if verbose:
        passed = reflection.get("pass", True)
        mark   = "PASS" if passed else "FAIL"
        print(f"  REFLECT {mark}: pass={json.dumps(passed)}")
        for gap in reflection.get("gaps", []):
            print(f"    gap: {gap}")
        print(f"  notes: {reflection.get('notes', '')}")

    return reflection

# =============================================================================
# SECTION 16 -- PHASE 4: REPORTER AGENT
# =============================================================================

_REPORTER_SYSTEM = """You are a Credit Risk Report Writer.
Given a structured decision rationale JSON, write a professional narrative
credit risk assessment report. Maximum 400 words.

Required sections (in this order):
1. Header line: APPROVE or REJECT (make it prominent).
2. Summary paragraph: decision + P(default) + segment.
3. Key Risk Factors: bullet list of primary factors.
4. Policy Grounds: the referenced policy sections.
5. Applicant Segment: peer-group position description.
6. Conditions (if APPROVE) or Next Steps (if REJECT): bullet list.
7. Disclaimer paragraph.

Tone: professional, factual, and respectful.
"""


def run_reporter(state, groq_client, verbose=True):
    """
    Phase 4: Generate the narrative credit risk report from the decision rationale.

    Enriches the decision_rationale dict with segment context, then asks the
    Reporter LLM to write a professional narrative.

    Falls back to a template-formatted report if the LLM call fails, so
    state["final_report"] is ALWAYS populated after this function returns.

    Parameters
    ----------
    state : dict
        Pipeline state dict.
        Reads:  state["decision_rationale"], state["segment_score"], state["risk_flags"]
        Writes: state["final_report"]
    groq_client : Groq
        Authenticated Groq client.
    verbose : bool
        If True, print a completion message.

    Returns
    -------
    str
        The narrative report string (also written to state["final_report"]).
    """
    if verbose:
        print("\n" + "-" * 66)
        print("  PHASE 4 -- REPORTER")

    rationale = state["decision_rationale"] or {}
    seg_data  = state["segment_score"]      or {}

    # Enrich the rationale with segment context before sending to the LLM
    enriched = {
        **rationale,
        "segment_details":      seg_data.get("interpretation", ""),
        "composite_risk_score": seg_data.get("composite_risk_score"),
        "flag_count":           (state["risk_flags"] or {}).get("flag_count", 0),
    }

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL_STRONG,
            messages=[
                {"role": "system", "content": _REPORTER_SYSTEM},
                {
                    "role":    "user",
                    "content": (
                        "Write the report for:\n"
                        + json.dumps(enriched, indent=2, default=str)
                    ),
                },
            ],
            temperature=0.1,    # slight variation for natural-sounding prose
            max_tokens=1024,
        )
        report = (resp.choices[0].message.content or "").strip()

    except Exception as exc:
        # Template fallback ensures state["final_report"] is always set
        dec     = rationale.get("decision",        "UNKNOWN")
        prob    = rationale.get("probability_pct", "N/A")
        seg     = rationale.get("segment",         "N/A")
        ts      = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        factors = "\n".join(f"  - {f}" for f in rationale.get("primary_factors", []))
        conds   = "\n".join(f"  - {c}" for c in rationale.get("conditions",      []))
        hdr     = "CONDITIONS" if dec == "APPROVE" else "NEXT STEPS"

        report = (
            f"CREDIT RISK ASSESSMENT -- {dec}\n"
            f"Generated: {ts}\n"
            f"{'=' * 60}\n"
            f"Decision: {dec} | P(default): {prob} | Segment: {seg}\n\n"
            f"KEY RISK FACTORS\n{factors}\n\n"
            f"{hdr}\n{conds}\n\n"
            f"DISCLAIMER: AI-generated. Requires qualified credit officer review.\n"
            f"[Template fallback -- Reporter LLM error: {exc}]"
        )

    state["final_report"] = report

    if verbose:
        print("  Report generated.")

    return report

# =============================================================================
# SECTION 17 -- ORCHESTRATOR: run_per_agent()
# =============================================================================

def run_per_agent(applicant_data, verbose=True):
    """
    Run the full Plan-Execute-Reflect pipeline for one loan applicant.

    This is the single entry point for callers. It creates a fresh state dict,
    runs all four phases in order, and returns the completed state.

    Execution flow
    --------------
    Phase 1 -- run_planner()   -> state["plan"]
    Phase 2 -- run_executor()  -> state["ml_output"], ["risk_flags"], etc.
    Phase 3 -- run_reflector() -> state["reflection"]
        If reflection.pass is False and retry_steps is non-empty:
            Re-run the listed tools (max MAX_REFLECT_RETRIES times).
    Phase 4 -- run_reporter()  -> state["final_report"]

    Parameters
    ----------
    applicant_data : dict
        Raw applicant feature dict. All keys are optional; missing ones
        are filled with training-set defaults during preprocessing.
    verbose : bool
        If True, print a step-by-step trace to stdout.
        Set to False for batch processing.

    Returns
    -------
    dict
        The completed pipeline state dict.

    Key fields in the returned state
    ---------------------------------
    state["final_decision"]      -- "APPROVE" | "REJECT" | None
    state["ml_output"]           -- {prediction, probability, confidence_band, decision}
    state["segment_score"]       -- {percentiles, composite_risk_score, segment}
    state["risk_flags"]          -- {flags, severity, flag_count}
    state["decision_rationale"]  -- structured decision JSON
    state["final_report"]        -- narrative string ready for display
    state["reflection"]          -- {pass, gaps, notes}
    state["audit_trail"]         -- list of timestamped events

    Raises
    ------
    EnvironmentError
        If GROQ_API_KEY is not set in the environment.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Set it with: os.environ['GROQ_API_KEY'] = 'your-key'"
        )

    groq_client = Groq(api_key=api_key)
    state       = make_state(applicant_data)

    if verbose:
        print("\n" + "=" * 66)
        print("  PLAN-EXECUTE-REFLECT AGENT -- START")
        print("=" * 66)
        # json.dumps prevents Python dict repr reaching Colab's JS frontend
        print("  Input: " + json.dumps(applicant_data, default=str))

    # Phase 1: Plan
    log_event(state, "ORCHESTRATOR", "phase_1_planner_start")
    state["plan"] = run_planner(applicant_data, groq_client, verbose)
    log_event(state, "ORCHESTRATOR", "phase_1_planner_done",
              f"{len(state['plan'])} steps")

    # Phase 2: Execute
    log_event(state, "ORCHESTRATOR", "phase_2_executor_start")
    state = run_executor(state, groq_client, verbose)
    log_event(state, "ORCHESTRATOR", "phase_2_executor_done",
              json.dumps(list(get_tools_called(state))))

    # Phase 3: Reflect (with limited retry)
    for retry_num in range(MAX_REFLECT_RETRIES + 1):
        log_event(state, "ORCHESTRATOR", f"phase_3_reflect_attempt_{retry_num + 1}")
        reflection          = run_reflector(state, groq_client, verbose)
        state["reflection"] = reflection

        if reflection.get("pass", True):
            break

        retry_steps = reflection.get("retry_steps", [])
        if not retry_steps or state["reflect_retries"] >= MAX_REFLECT_RETRIES:
            if verbose:
                print("  Reflection failed but retry cap reached -- proceeding.")
            break

        state["reflect_retries"] += 1
        if verbose:
            # json.dumps prevents Python list repr in stdout
            print("  Reflector requested retry of: " + json.dumps(retry_steps))

        for tool_name in retry_steps:
            if tool_name not in TOOL_REGISTRY:
                continue
            result_str, ok = dispatch_tool(
                tool_name,
                {"applicant_data": applicant_data},
                state,
            )
            if verbose:
                status = "OK" if ok else "ERROR"
                print(f"    retry {tool_name}: {status}")

    log_event(state, "ORCHESTRATOR", "phase_3_reflect_done")

    # Phase 4: Report
    log_event(state, "ORCHESTRATOR", "phase_4_reporter_start")
    run_reporter(state, groq_client, verbose)
    log_event(state, "ORCHESTRATOR", "phase_4_reporter_done")

    if verbose:
        print("\n" + "=" * 66)
        print("  PER AGENT -- COMPLETE")
        print(f"  Decision  : {state['final_decision']}")
        print("  Tools run : " + json.dumps(list(get_tools_called(state))))
        print(f"  Retries   : {state['reflect_retries']}")
        print(f"  Errors    : {len(state['error_log'])}")
        print("=" * 66)

    return state

# =============================================================================
# SECTION 18 -- UTILITY: print_per_trace()
# =============================================================================

def print_per_trace(state):
    """
    Print a structured execution trace of a completed pipeline run.

    Reads from the state dict without modifying it. Useful for debugging
    and project reports.

    Parameters
    ----------
    state : dict
        Completed pipeline state dict returned by run_per_agent().
    """
    print("\n" + "=" * 66)
    print("  PER EXECUTION TRACE")
    print("=" * 66)
    print(f"  Plan steps  : {len(state['plan'] or [])}")
    print("  Tools called: " + json.dumps(list(get_tools_called(state))))
    print(f"  Retries     : {state['reflect_retries']}")
    print(f"  Errors      : {len(state['error_log'])}")

    if state["plan"]:
        print("\n  PLAN")
        for step in state["plan"]:
            q = f" (query: {step.get('query','')!r})" if "query" in step else ""
            print(f"    {step.get('step','?')}. {step.get('action','?')}{q}")
            print(f"       {step.get('reason', '')}")

    print("\n  EXECUTION LOG")
    for idx, entry in enumerate(state["execution_log"], 1):
        status = "OK   " if entry["success"] else "ERROR"
        print(f"  {idx}. {status} | {entry['tool']}")
        print(f"       {entry['result'][:150]}")

    if state["reflection"]:
        rf = state["reflection"]
        print("\n  REFLECTION")
        print(f"    pass: {json.dumps(rf.get('pass'))}")
        print(f"    notes: {rf.get('notes', '')}")
        for gap in rf.get("gaps", []):
            print(f"    gap: {gap}")

    if state["error_log"]:
        print("\n  ERRORS")
        for err in state["error_log"]:
            print(f"    ! {err.get('error', str(err))}")
