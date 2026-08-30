# -*- coding: utf-8 -*-
"""
pipeline.py
Core agentic RFP evaluation pipeline: document extraction, LLM scoring (with
retry-on-invalid-JSON via a LangGraph loop), deterministic validation,
weighted scoring, peer benchmarking, PPI + tie-break ranking, and SQLite
persistence.

Every node emits a structured trace entry (node, level, message, timestamp)
that is (a) streamed out live via callbacks for the Streamlit UI, and
(b) persisted into result_json so a run can be debugged after the fact,
not only while it's running. Exceptions inside a node never crash the
whole batch -- they're caught, logged into the trace, and that supplier
is marked 'failed' while the rest of the batch continues.

This module has no Streamlit dependency by design -- it can be tested,
demoed, or reused headlessly.
"""

import json
import operator
import traceback
from datetime import datetime
from typing import Optional, Callable, Annotated

import pypdf
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from db import get_conn

# ==========================================
# LLM CONFIGURATION
# ==========================================
_llm: Optional[ChatOpenAI] = None


def configure_llm(api_key: str, model: str = "nvidia/nemotron-3.5-lightning:free",
                   temperature: float = 0.4) -> None:
    """Initializes the module-level LLM client. Must be called before running a batch."""
    global _llm
    _llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def is_llm_configured() -> bool:
    return _llm is not None


# ==========================================
# TRACE HELPER
# ==========================================
def _trace(node: str, message: str, level: str = "info") -> list:
    """Builds a single structured trace entry, wrapped in a list so it can be
    added directly to a node's returned state update (the 'trace' state field
    uses an additive reducer, so returning [entry] appends rather than overwrites).
    level: 'info' | 'success' | 'warning' | 'error'."""
    return [{
        "node": node,
        "level": level,
        "message": message,
        "ts": datetime.now().strftime("%H:%M:%S"),
    }]


def _merge_state(state: dict, update: dict) -> dict:
    """Applies one node's partial update to the running state, mirroring the
    graph's own reducers: 'trace' entries are appended, everything else is
    overwritten. Used when driving the graph via .stream() so we can react to
    each node as it completes without running the graph a second time."""
    merged = dict(state)
    for k, v in update.items():
        if k == "trace":
            merged["trace"] = merged.get("trace", []) + v
        else:
            merged[k] = v
    return merged


# ==========================================
# GRAPH STATE
# ==========================================
class SupplierState(TypedDict):
    rfp_run_id: int
    supplier_name: str
    file_path: str
    submission_date: str
    experience_rating: float
    active_criteria: dict  # {criterion_id: {name, weight, max_score}}

    extracted_text: str
    extraction_success: bool

    raw_llm_output: dict
    llm_attempt: int

    valid_criteria: dict
    validation_warnings: list
    is_valid: bool

    absolute_score: float
    status: str  # 'evaluated' | 'failed'

    trace: Annotated[list, operator.add]  # additive: every node appends, nothing overwrites it


# ==========================================
# 1. DOCUMENT TEXT EXTRACTION
# ==========================================
def extract_pdf_text(file_path: str) -> dict:
    """Extracts raw text content from a supplier PDF using pypdf."""
    try:
        text = ""
        page_count = 0
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            page_count = len(reader.pages)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        success = len(text.strip()) > 0
        if success:
            return {"text": text.strip(), "success": True, "page_count": page_count, "error": None}
        return {"text": "", "success": False, "page_count": page_count,
                "error": "No extractable text found (PDF may be a scanned image with no OCR layer)."}
    except Exception as e:
        return {"text": "", "success": False, "page_count": 0,
                "error": f"{type(e).__name__}: {e}"}


# ==========================================
# 2. LLM EXECUTION LAYER
# ==========================================
def call_llm_scorer(extracted_text: str, active_criteria: dict, llm_attempt: int,
                     validation_warnings: list) -> dict:
    """Calls the LLM to score one supplier. Feeds prior validation warnings back into
    the prompt on retry so the model can self-correct instead of blindly re-rolling.
    Always returns a structured result -- never raises -- so the caller always knows
    exactly what happened (success, JSON-parse failure, or a network/API exception)."""
    if _llm is None:
        return {"success": False, "data": {}, "error": "LLM is not configured (missing API key).",
                "raw_text": ""}

    feedback = ""
    if llm_attempt > 1 and validation_warnings:
        feedback = (
            "\n\n[CRITICAL CORRECTION] Your previous response failed verification due to: "
            + "; ".join(validation_warnings)
            + "\nFix these validation gaps explicitly in this attempt."
        )

    criteria_str = json.dumps(active_criteria, indent=2)
    prompt = f"""
Analyze the following Supplier RFP text and evaluate it against these active criteria:
{criteria_str}

Supplier Document Text:
---
{extracted_text}
---
{feedback}

Output requirements: Return a JSON structure exactly matching this schema format:
{{
   "supplier_name": "Exact Name",
   "criteria": [
      {{"criterion_id": 1, "score": 8, "max_score": 10, "justification": "...", "evidence": "..."}}
   ],
   "risks": ["..."],
   "overall_summary": "..."
}}
Use only evidence present in the supplier document. Return one result for every
active criterion listed above. Stay within the given score range. Output JSON only --
no markdown fences, no conversational text.
"""

    raw_text = ""
    try:
        response = _llm.invoke([
            SystemMessage(content="You are an expert procurement grading engine. "
                                   "Return ONLY valid JSON. No conversational text or markdown blocks."),
            HumanMessage(content=prompt),
        ])
        raw_text = response.content.strip().removeprefix("```json").removesuffix("```").strip()
        parsed = json.loads(raw_text)
        return {"success": True, "data": parsed, "error": None, "raw_text": raw_text}
    except json.JSONDecodeError as e:
        return {"success": False, "data": {}, "error": f"JSON parse error: {e}",
                "raw_text": raw_text[:500]}
    except Exception as e:
        # Covers network errors, auth errors, rate limits, provider-side errors, etc.
        return {"success": False, "data": {}, "error": f"LLM call failed: {type(e).__name__}: {e}",
                "raw_text": ""}


# ==========================================
# 3. SCHEMA VERIFICATION & NORMALIZATION
# ==========================================
def validate_scorecard(raw_llm_output: dict, active_criteria: dict) -> dict:
    """Validates schema structure, fills missing criteria, and clamps out-of-range scores."""
    warnings = []
    valid_criteria = {}
    is_valid = True

    if not raw_llm_output or "criteria" not in raw_llm_output:
        return {
            "valid_criteria": {},
            "validation_warnings": ["Malformed or empty JSON payload from LLM response."],
            "is_valid": False,
        }

    llm_map = {c.get("criterion_id"): c for c in raw_llm_output["criteria"] if isinstance(c, dict)}

    for c_id, info in active_criteria.items():
        name = info["name"]
        weight = info.get("weight", 0.0)
        max_allowed = info.get("max_score", 100)

        if c_id not in llm_map:
            is_valid = False
            warnings.append(f"Missing evaluation metrics for criterion ID {c_id} ({name}).")
            valid_criteria[name] = {
                "criterion_id": c_id,
                "weight": weight,
                "score": 0.0,
                "max_score": max_allowed,
                "justification": "Fallback: criterion missing from LLM response",
                "evidence": "None",
            }
        else:
            item = llm_map[c_id]
            try:
                raw_score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                warnings.append(f"Non-numeric score for {name}: {item.get('score')!r}. Defaulted to 0.")
                raw_score = 0.0
                is_valid = False

            if raw_score < 0 or raw_score > max_allowed:
                warnings.append(
                    f"Score for {name} ({raw_score}) out of bounds [0-{max_allowed}]. Clamped."
                )
                raw_score = max(0.0, min(raw_score, float(max_allowed)))
                is_valid = False

            valid_criteria[name] = {
                "criterion_id": c_id,
                "weight": weight,
                "score": raw_score,
                "max_score": max_allowed,
                "justification": item.get("justification", "Missing text summary details."),
                "evidence": item.get("evidence", "No document citations recorded."),
            }

    return {"valid_criteria": valid_criteria, "validation_warnings": warnings, "is_valid": is_valid}


# ==========================================
# 4. DETERMINISTIC BUSINESS CALCULATIONS
# ==========================================
def calculate_absolute_score(valid_criteria: dict, active_criteria: dict) -> float:
    """Absolute weighted score: Sum of (Score / Max Score) * Weight, for ONE supplier."""
    total_weighted_score = 0.0
    criteria_weight_map = {info["name"]: info["weight"] for info in active_criteria.values()}

    for name, data in valid_criteria.items():
        weight = criteria_weight_map.get(name, 0.0)
        score = data["score"]
        max_score = data.get("max_score", 100.0)
        if max_score > 0:
            total_weighted_score += (score / max_score) * weight

    return round(total_weighted_score, 4)


def calculate_benchmarks(evaluated_results: list) -> dict:
    """Highest valid score observed for each criterion, across all suppliers in the batch."""
    benchmarks = {}
    for run in evaluated_results:
        for name, data in run.get("valid_criteria", {}).items():
            current_max = benchmarks.get(name, 0.0)
            if data["score"] > current_max:
                benchmarks[name] = data["score"]
    return benchmarks


def calculate_ppi_and_rank(evaluated_results: list, benchmarks: dict) -> list:
    """Calculates gaps, relative %, PPI, and applies the mandatory deterministic
    tie-break order: 1) PPI desc, 2) submission date asc, 3) experience rating desc,
    4) supplier name asc. Assigns sequential final_rank only after the stable sort."""
    ranked_outputs = []

    for run in evaluated_results:
        metrics = run.get("valid_criteria", {})
        ppi_weighted_sum = 0.0
        total_weight = 0.0
        updated_details = {}

        for name, data in metrics.items():
            score = data["score"]
            weight = data.get("weight", 1.0)
            bench_score = benchmarks.get(name, 0.0)
            gap = score - bench_score
            relative_pct = (score / bench_score * 100.0) if bench_score > 0 else 100.0

            ppi_weighted_sum += relative_pct * weight
            total_weight += weight

            updated_details[name] = {
                **data, "benchmark": bench_score, "gap": gap, "relative_pct": round(relative_pct, 2)
            }

        ppi = (ppi_weighted_sum / total_weight) if total_weight > 0 else 0.0

        ranked_outputs.append({
            "supplier_name": run["supplier_name"],
            "submission_date": run["submission_date"],
            "experience_rating": run["experience_rating"],
            "absolute_score": run["absolute_score"],
            "ppi": round(ppi, 4),
            "valid_criteria": updated_details,
            "raw_llm_output": run.get("raw_llm_output", {}),
            "status": "evaluated",
        })

    # Mandatory tie-break order (stable sort -> deterministic and reproducible)
    ranked_outputs.sort(key=lambda x: (
        -x["ppi"],
        x["submission_date"],
        -x["experience_rating"],
        x["supplier_name"].lower(),
    ))

    for index, record in enumerate(ranked_outputs):
        record["final_rank"] = index + 1

    return ranked_outputs


# ==========================================
# GRAPH NODES (each one always emits a trace entry, success or failure)
# ==========================================
def _node_extract_pdf(state: SupplierState):
    result = extract_pdf_text(state["file_path"])
    if result["success"]:
        msg = f"Extracted {len(result['text'])} characters from {result['page_count']} page(s)."
        return {"extracted_text": result["text"], "extraction_success": True,
                "trace": _trace("extract", msg, level="success")}
    msg = result["error"] or "Unknown extraction failure."
    return {"extracted_text": "", "extraction_success": False,
            "trace": _trace("extract", msg, level="error")}


def _node_evaluate_with_llm(state: SupplierState):
    attempt = state.get("llm_attempt", 0) + 1
    result = call_llm_scorer(
        state["extracted_text"], state["active_criteria"], attempt, state.get("validation_warnings", []),
    )
    if result["success"]:
        msg = f"Attempt {attempt}: LLM returned parseable JSON ({len(result['data'].get('criteria', []))} criteria scored)."
        return {"raw_llm_output": result["data"], "llm_attempt": attempt,
                "trace": _trace("evaluate", msg, level="success")}
    msg = f"Attempt {attempt}: {result['error']}"
    return {"raw_llm_output": {}, "llm_attempt": attempt,
            "trace": _trace("evaluate", msg, level="error")}


def _node_validate(state: SupplierState):
    result = validate_scorecard(state["raw_llm_output"], state["active_criteria"])
    if result["is_valid"]:
        msg = "All active criteria present and within range."
        level = "success"
    elif result["valid_criteria"]:
        msg = "Validation issues (will retry): " + "; ".join(result["validation_warnings"])
        level = "warning"
    else:
        msg = "Validation issues: " + "; ".join(result["validation_warnings"])
        level = "warning"
    return {"valid_criteria": result["valid_criteria"], "validation_warnings": result["validation_warnings"],
            "is_valid": result["is_valid"], "trace": _trace("validate", msg, level=level)}


def _node_score(state: SupplierState):
    abs_score = calculate_absolute_score(state["valid_criteria"], state["active_criteria"])
    msg = f"Absolute weighted score calculated: {abs_score}"
    return {"absolute_score": abs_score, "status": "evaluated",
            "trace": _trace("score", msg, level="success")}


def _node_mark_failed(state: SupplierState):
    if not state.get("extraction_success", True):
        reason = "PDF text extraction failed -- see 'extract' node above."
    else:
        reason = (f"Validation still failing after {state.get('llm_attempt', 0)} LLM attempt(s) "
                  "-- see 'evaluate'/'validate' entries above.")
    return {"status": "failed", "trace": _trace("fail", reason, level="error")}


def build_supplier_graph():
    """Builds the per-supplier LangGraph: extract -> evaluate -> validate -> score,
    with a retry loop (evaluate <-> validate, up to 3 attempts) on invalid LLM output.
    Benchmarking and ranking happen OUTSIDE this graph, once per batch, in plain
    deterministic Python -- the LLM never touches arithmetic, benchmarks, or rank."""
    workflow = StateGraph(SupplierState)

    workflow.add_node("extract", _node_extract_pdf)
    workflow.add_node("evaluate", _node_evaluate_with_llm)
    workflow.add_node("validate", _node_validate)
    workflow.add_node("score", _node_score)
    workflow.add_node("fail", _node_mark_failed)

    workflow.add_edge(START, "extract")
    workflow.add_conditional_edges(
        "extract",
        lambda s: "evaluate" if s["extraction_success"] else "fail",
        {"evaluate": "evaluate", "fail": "fail"},
    )
    workflow.add_edge("evaluate", "validate")
    workflow.add_conditional_edges(
        "validate",
        lambda s: "score" if s["is_valid"] else ("evaluate" if s["llm_attempt"] < 3 else "fail"),
        {"score": "score", "evaluate": "evaluate", "fail": "fail"},
    )
    workflow.add_edge("score", END)
    workflow.add_edge("fail", END)

    return workflow.compile()


# ==========================================
# PERSISTENCE
# ==========================================
def persist_supplier_result(rfp_run_id: int, result: dict) -> None:
    """Saves a supplier's initial evaluation pass (pre-ranking) to SQLite, including
    the full trace log and any unhandled-exception traceback, so the run stays
    debuggable after the fact."""
    conn = get_conn()
    cursor = conn.cursor()

    details = {
        "valid_criteria": result.get("valid_criteria", {}),
        "raw_llm_output": result.get("raw_llm_output", {}),
        "trace": result.get("trace", []),
    }
    if result.get("_exception_traceback"):
        details["exception_traceback"] = result["_exception_traceback"]

    warnings_list = list(result.get("validation_warnings", []) or [])
    if result.get("_exception_traceback"):
        warnings_list.append("An unhandled exception occurred -- see trace / traceback for details.")

    cursor.execute('''
        INSERT INTO supplier_results (
            rfp_run_id, supplier_name, submission_date, experience_rating,
            source_filename, status, validation_warnings, absolute_score, result_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        rfp_run_id,
        result.get("supplier_name", "Unknown"),
        result.get("submission_date"),
        result.get("experience_rating"),
        result.get("file_path", "").split("/")[-1].split("\\")[-1],
        result.get("status", "failed"),
        "; ".join(warnings_list) if warnings_list else None,
        result.get("absolute_score"),
        json.dumps(details),
    ))
    conn.commit()
    conn.close()


def persist_final_ranks(rfp_run_id: int, ranked_suppliers: list) -> None:
    """Updates PPI, final rank, and the ENRICHED result_json (benchmark/gap/relative_pct
    per criterion) so every score on the leaderboard stays traceable to its criterion,
    weight, evidence, and the business rule that produced it. The trace log already
    persisted by persist_supplier_result is preserved untouched."""
    conn = get_conn()
    cursor = conn.cursor()

    for row in ranked_suppliers:
        cursor.execute(
            "SELECT result_json FROM supplier_results WHERE rfp_run_id = ? AND supplier_name = ?",
            (rfp_run_id, row["supplier_name"]),
        )
        existing = cursor.fetchone()
        details = json.loads(existing[0]) if existing and existing[0] else {}
        details["valid_criteria"] = row["valid_criteria"]  # now enriched with benchmark/gap/relative_pct
        details["raw_llm_output"] = row.get("raw_llm_output", details.get("raw_llm_output", {}))
        # 'trace' key is left untouched here -- it already carries the full node history.

        cursor.execute('''
            UPDATE supplier_results
            SET ppi = ?, final_rank = ?, result_json = ?
            WHERE rfp_run_id = ? AND supplier_name = ?
        ''', (row["ppi"], row["final_rank"], json.dumps(details), rfp_run_id, row["supplier_name"]))

    conn.commit()
    conn.close()


def update_rfp_run_status(rfp_run_id: int, status: str, error_message: Optional[str] = None) -> None:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE rfp_runs
        SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ?
        WHERE rfp_run_id = ?
    ''', (status, error_message, rfp_run_id))
    conn.commit()
    conn.close()


def create_rfp_run(active_criteria: dict, supplier_count: int) -> int:
    """Creates the batch/run row and freezes a snapshot of the criteria used,
    so this run stays explainable even if criteria are edited later."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO rfp_runs (status, criteria_snapshot_json, supplier_count) VALUES (?, ?, ?)",
        ("running", json.dumps(active_criteria), supplier_count),
    )
    new_run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return new_run_id


# ==========================================
# BATCH ORCHESTRATOR
# ==========================================
def run_batch(
    rfp_run_id: int,
    uploaded_files: list,
    active_criteria: dict,
    on_supplier_start: Optional[Callable[[int, int, str], None]] = None,
    on_node_event: Optional[Callable[[str, str, dict], None]] = None,
    on_progress: Optional[Callable[[int, int, str, str], None]] = None,
) -> list:
    """Runs every supplier through the per-supplier graph, persists interim results,
    then benchmarks + ranks the whole batch together (deterministic, plain Python)
    before persisting final ranks and closing out the run.

    Callbacks (all optional, all safe to omit for headless/test use):
      on_supplier_start(index, total, supplier_name) -- fired before a supplier begins
      on_node_event(supplier_name, node_name, update_dict) -- fired after EVERY graph
          node completes, including retries; update_dict includes that node's 'trace'
          entries, giving live, granular visibility into exactly what's happening.
      on_progress(index, total, supplier_name, final_status) -- fired once a supplier
          finishes (evaluated or failed).

    A supplier that raises an unexpected exception never crashes the batch: the
    exception is caught, logged into that supplier's trace + traceback, the supplier
    is marked 'failed', and the loop continues to the next supplier.
    """
    graph = build_supplier_graph()
    all_results = []
    total = len(uploaded_files)
    any_evaluated = False

    for i, file_info in enumerate(uploaded_files):
        supplier_name = file_info.get("supplier_name") or f"Supplier {i + 1}"

        if on_supplier_start:
            on_supplier_start(i + 1, total, supplier_name)

        state = {
            **file_info,
            "rfp_run_id": rfp_run_id,
            "active_criteria": active_criteria,
            "llm_attempt": 0,
            "validation_warnings": [],
            "trace": [],
        }

        try:
            for step in graph.stream(state, stream_mode="updates"):
                for node_name, update in step.items():
                    state = _merge_state(state, update)
                    if on_node_event:
                        on_node_event(supplier_name, node_name, update)
        except Exception as e:
            tb = traceback.format_exc()
            error_trace = _trace("orchestrator", f"Unhandled exception: {type(e).__name__}: {e}", level="error")
            state["status"] = "failed"
            state["trace"] = state.get("trace", []) + error_trace
            state["_exception_traceback"] = tb
            if on_node_event:
                on_node_event(supplier_name, "orchestrator", {"trace": error_trace})

        all_results.append(state)
        persist_supplier_result(rfp_run_id, state)

        if state.get("status") == "evaluated":
            any_evaluated = True

        if on_progress:
            on_progress(i + 1, total, supplier_name, state.get("status", "failed"))

    evaluated = [r for r in all_results if r.get("status") == "evaluated"]
    benchmarks = calculate_benchmarks(evaluated)
    ranked = calculate_ppi_and_rank(evaluated, benchmarks)
    persist_final_ranks(rfp_run_id, ranked)

    if any_evaluated:
        update_rfp_run_status(rfp_run_id, "completed")
    else:
        update_rfp_run_status(rfp_run_id, "failed",
                               error_message="No suppliers were successfully evaluated -- check each supplier's trace log.")

    return all_results
