# -*- coding: utf-8 -*-
"""
app.py
Streamlit UI for the Agentic RFP Evaluation and Supplier Ranking project.

Screens: Criteria | New Evaluation (upload + run) | Leaderboard & Results
(leaderboard, detailed scorecards, run details, JSON download).
"""

import json
import os
import tempfile
from datetime import date

import pandas as pd
import streamlit as st

import db
import pipeline

st.set_page_config(page_title="Agentic RFP Evaluation", layout="wide")

# ------------------------------------------------------------------
# One-time setup
# ------------------------------------------------------------------
db.init_db()

if "supplier_meta" not in st.session_state:
    st.session_state.supplier_meta = {}  # filename -> {supplier_name, submission_date, experience_rating}
if "last_run_id" not in st.session_state:
    st.session_state.last_run_id = None

st.title("Agentic RFP Evaluation & Supplier Ranking")

# ------------------------------------------------------------------
# Sidebar: LLM configuration
# ------------------------------------------------------------------
with st.sidebar:
    st.header("LLM Configuration")
    api_key = st.text_input("OpenRouter API Key", type="password",
                             help="Used only for this session, never stored.")
    model_name = st.text_input("Model", value="nvidia/nemotron-3.5-lightning:free")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.1)

    if api_key:
        pipeline.configure_llm(api_key, model=model_name, temperature=temperature)
        st.success("LLM configured for this session.")
    else:
        st.info("Enter an API key to enable evaluation.")

tab_criteria, tab_new_eval, tab_leaderboard = st.tabs(
    ["Criteria", "New Evaluation", "Leaderboard & Results"]
)

# ==================================================================
# TAB 1 — CRITERIA
# ==================================================================
with tab_criteria:
    st.subheader("Evaluation Criteria Management")
    
    rows = db.get_all_criteria_rows()
    df = pd.DataFrame(
        rows, columns=["ID", "Name", "Description", "Weight (%)", "Max Score", "Active"]
    )
    df["Active"] = df["Active"].map({1: "Yes", 0: "No"})
    
    # Display current criteria
    st.markdown("### Current Criteria")
    st.dataframe(df, use_container_width=True, hide_index=True)

    active_weight_sum = df.loc[df["Active"] == "Yes", "Weight (%)"].sum()
    if abs(active_weight_sum - 100.0) > 0.01:
        st.warning(
            f"Active criteria weights sum to {active_weight_sum:.1f}%, not 100%. "
            "Adjust weights in evaluation_criteria before running an evaluation."
        )
    else:
        st.caption(f"Active weights sum to {active_weight_sum:.1f}% ✅")

    st.divider()

    # Add New Criterion
    st.markdown("### ➕ Add New Criterion")
    col1, col2 = st.columns(2)
    with col1:
        new_name = st.text_input("Criterion Name", key="new_crit_name")
    with col2:
        new_weight = st.number_input("Weight (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key="new_crit_weight")
    
    new_description = st.text_area("Description", height=60, key="new_crit_desc")
    new_max_score = st.number_input("Max Score", min_value=1, value=10, key="new_crit_max")
    
    if st.button("Add Criterion", type="primary", key="add_crit_btn"):
        if new_name.strip():
            db.add_criterion(new_name.strip(), new_description.strip(), new_weight, new_max_score)
            st.success(f"✅ Added criterion: {new_name}")
            st.rerun()
        else:
            st.error("Criterion name cannot be empty")

    st.divider()

    # Edit Existing Criteria
    if rows:
        st.markdown("### ✏️ Edit Existing Criteria")
        for row in rows:
            crit_id, name, desc, weight, max_score, is_active = row
            status_text = "Active" if is_active else "Inactive"
            
            with st.expander(f"{name} ({status_text})"):
                col1, col2 = st.columns(2)
                with col1:
                    edit_name = st.text_input("Name", value=name, key=f"edit_name_{crit_id}")
                    edit_weight = st.number_input("Weight (%)", min_value=0.0, max_value=100.0, value=weight, step=1.0, key=f"edit_weight_{crit_id}")
                with col2:
                    edit_desc = st.text_area("Description", value=desc or "", height=80, key=f"edit_desc_{crit_id}")
                    edit_max_score = st.number_input("Max Score", min_value=1, value=max_score, key=f"edit_max_{crit_id}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    edit_active = st.checkbox("Active", value=bool(is_active), key=f"edit_active_{crit_id}")
                
                with col2:
                    if st.button("💾 Save Changes", key=f"save_crit_{crit_id}"):
                        db.update_criterion(crit_id, edit_name.strip(), edit_desc.strip(), edit_weight, edit_max_score, int(edit_active))
                        st.success(f"✅ Updated {edit_name}")
                        st.rerun()
                
                with col3:
                    if st.button("🗑️ Delete", key=f"del_crit_{crit_id}"):
                        db.delete_criterion(crit_id)
                        st.success(f"✅ Deleted criterion")
                        st.rerun()

# ==================================================================
# TAB 2 — NEW EVALUATION (upload + run a batch)
# ==================================================================
with tab_new_eval:
    st.subheader("Upload Supplier RFP PDFs")
    uploaded = st.file_uploader(
        "Select one or more supplier proposal PDFs", type=["pdf"], accept_multiple_files=True
    )

    active_criteria = db.get_active_criteria()

    if uploaded:
        st.markdown("#### Supplier metadata")
        for f in uploaded:
            key = f.name
            if key not in st.session_state.supplier_meta:
                default_name = os.path.splitext(f.name)[0]
                st.session_state.supplier_meta[key] = {
                    "supplier_name": default_name,
                    "submission_date": date.today(),
                    "experience_rating": 4.0,
                }

            with st.expander(f"📄 {f.name}", expanded=True):
                col1, col2, col3 = st.columns(3)
                meta = st.session_state.supplier_meta[key]
                meta["supplier_name"] = col1.text_input(
                    "Supplier name", value=meta["supplier_name"], key=f"name_{key}"
                )
                meta["submission_date"] = col2.date_input(
                    "Submission date", value=meta["submission_date"], key=f"date_{key}"
                )
                meta["experience_rating"] = col3.number_input(
                    "Historical experience rating (1-5)", min_value=1.0, max_value=5.0,
                    value=meta["experience_rating"], step=0.1, key=f"exp_{key}",
                )

        st.divider()
        run_disabled = not pipeline.is_llm_configured() or not active_criteria
        if not pipeline.is_llm_configured():
            st.caption("⚠️ Enter an API key in the sidebar to enable evaluation.")
        if not active_criteria:
            st.caption("⚠️ No active criteria found — seed evaluation_criteria first.")

        if st.button("Evaluate Batch", type="primary", disabled=run_disabled):
            with tempfile.TemporaryDirectory() as tmpdir:
                uploaded_files_payload = []
                for f in uploaded:
                    meta = st.session_state.supplier_meta[f.name]
                    file_path = os.path.join(tmpdir, f.name)
                    with open(file_path, "wb") as out:
                        out.write(f.getvalue())
                    uploaded_files_payload.append({
                        "supplier_name": meta["supplier_name"],
                        "file_path": file_path,
                        "submission_date": meta["submission_date"].isoformat(),
                        "experience_rating": meta["experience_rating"],
                    })

                new_run_id = pipeline.create_rfp_run(active_criteria, len(uploaded_files_payload))
                st.info(f"Created batch RFP_RUN_ID = {new_run_id}")

                progress_bar = st.progress(0.0)
                LEVEL_ICON = {"error": "🔴", "warning": "🟡", "success": "🟢", "info": "🔵"}
                status_containers = {}

                def on_supplier_start(index, total, supplier_name):
                    container = st.status(f"[{index}/{total}] {supplier_name}", expanded=True)
                    log_placeholder = container.empty()
                    status_containers[supplier_name] = {
                        "status": container, "log": log_placeholder, "lines": [],
                    }

                def on_node_event(supplier_name, node_name, update):
                    entry = status_containers.get(supplier_name)
                    if not entry:
                        return
                    for t in update.get("trace", []):
                        icon = LEVEL_ICON.get(t["level"], "🔵")
                        entry["lines"].append(f"{icon} `{t['ts']}` **{t['node']}** — {t['message']}")
                    entry["log"].markdown("\n\n".join(entry["lines"]))

                def on_supplier_done(index, total, supplier_name, status):
                    entry = status_containers.get(supplier_name)
                    if entry:
                        final_state = "complete" if status == "evaluated" else "error"
                        entry["status"].update(
                            label=f"[{index}/{total}] {supplier_name} — {status}",
                            state=final_state,
                            expanded=(status != "evaluated"),
                        )
                    progress_bar.progress(index / total)

                try:
                    pipeline.run_batch(
                        rfp_run_id=new_run_id,
                        uploaded_files=uploaded_files_payload,
                        active_criteria=active_criteria,
                        on_supplier_start=on_supplier_start,
                        on_node_event=on_node_event,
                        on_progress=on_supplier_done,
                    )
                    st.session_state.last_run_id = new_run_id
                    st.success(
                        f"Batch {new_run_id} completed. Open the **Leaderboard & Results** "
                        "tab to view the ranking. Expand any supplier above (or the Detailed "
                        "Scorecard in that tab) to see exactly what each step did."
                    )
                except Exception as e:
                    pipeline.update_rfp_run_status(new_run_id, "failed", error_message=str(e))
                    st.error(f"Batch-level failure (before any supplier could be isolated): {e}")
    else:
        st.caption("Upload at least one supplier PDF to begin.")

# ==================================================================
# TAB 3 — LEADERBOARD & RESULTS
# ==================================================================
with tab_leaderboard:
    runs = db.list_runs()
    if not runs:
        st.caption("No evaluation runs yet. Create one in the **New Evaluation** tab.")
    else:
        run_labels = {
            r[0]: f"Run #{r[0]} — {r[1]} — {r[3]} — {r[4]} supplier(s)" for r in runs
        }
        default_index = 0
        if st.session_state.last_run_id in run_labels:
            default_index = list(run_labels.keys()).index(st.session_state.last_run_id)

        selected_run_id = st.selectbox(
            "Select a batch (RFP_RUN_ID)",
            options=list(run_labels.keys()),
            format_func=lambda rid: run_labels[rid],
            index=default_index,
        )

        run_row = db.get_run(selected_run_id)
        _, created_at, completed_at, status, criteria_snapshot_json, supplier_count, error_message = run_row

        c1, c2, c3 = st.columns(3)
        c1.metric("Status", status)
        c2.metric("Suppliers", supplier_count)
        c3.metric("Completed at", completed_at or "—")
        if error_message:
            st.error(f"Run error: {error_message}")

        results = db.get_supplier_results(selected_run_id)
        ranked = [r for r in results if r[9] is not None]  # final_rank not null
        unranked = [r for r in results if r[9] is None]

        st.markdown("### 🏆 Leaderboard")
        if ranked:
            lb_df = pd.DataFrame(
                [
                    {
                        "Rank": r[9],
                        "Supplier": r[1],
                        "Absolute Score": round(r[7], 2) if r[7] is not None else None,
                        "PPI": round(r[8], 2) if r[8] is not None else None,
                        "Submission Date": r[2],
                        "Experience Rating": r[3],
                    }
                    for r in ranked
                ]
            )
            st.dataframe(lb_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No suppliers were successfully ranked in this run.")

        if unranked:
            st.markdown("### ⚠️ Suppliers that did not complete evaluation")
            fail_df = pd.DataFrame(
                [{"Supplier": r[1], "Status": r[5], "Warnings": r[6]} for r in unranked]
            )
            st.dataframe(fail_df, use_container_width=True, hide_index=True)

        st.markdown("### 🔍 Detailed Scorecard")
        if results:
            supplier_names = [r[1] for r in results]
            selected_supplier = st.selectbox("Select a supplier", supplier_names)
            row = next(r for r in results if r[1] == selected_supplier)
            (_, s_name, s_date, s_exp, s_file, s_status, s_warnings,
             s_abs, s_ppi, s_rank, s_result_json, s_evaluated_at) = row

            colA, colB, colC, colD = st.columns(4)
            colA.metric("Status", s_status)
            colB.metric("Absolute Score", round(s_abs, 2) if s_abs is not None else "—")
            colC.metric("PPI", round(s_ppi, 2) if s_ppi is not None else "—")
            colD.metric("Final Rank", s_rank if s_rank is not None else "—")

            if s_warnings:
                st.warning(f"Validation warnings: {s_warnings}")

            details = json.loads(s_result_json) if s_result_json else {}
            valid_criteria = details.get("valid_criteria", {})
            if valid_criteria:
                crit_df = pd.DataFrame(
                    [
                        {
                            "Criterion": name,
                            "Score": d.get("score"),
                            "Max": d.get("max_score"),
                            "Weight (%)": d.get("weight"),
                            "Benchmark": d.get("benchmark", "—"),
                            "Gap": d.get("gap", "—"),
                            "Relative %": d.get("relative_pct", "—"),
                            "Justification": d.get("justification"),
                            "Evidence": d.get("evidence"),
                        }
                        for name, d in valid_criteria.items()
                    ]
                )
                st.dataframe(crit_df, use_container_width=True, hide_index=True)

            raw = details.get("raw_llm_output", {})
            if raw.get("overall_summary"):
                st.markdown(f"**Overall summary:** {raw['overall_summary']}")
            if raw.get("risks"):
                st.markdown("**Risks flagged:** " + "; ".join(raw["risks"]))

            trace = details.get("trace", [])
            if trace:
                LEVEL_ICON = {"error": "🔴", "warning": "🟡", "success": "🟢", "info": "🔵"}
                with st.expander(f"🛠️ Execution trace ({len(trace)} steps) — debug log", expanded=(s_status == "failed")):
                    for t in trace:
                        icon = LEVEL_ICON.get(t.get("level"), "🔵")
                        st.markdown(f"{icon} `{t.get('ts')}` **{t.get('node')}** — {t.get('message')}")

            if details.get("exception_traceback"):
                with st.expander("⚠️ Unhandled exception traceback", expanded=True):
                    st.code(details["exception_traceback"], language="text")

        st.markdown("### 📎 Run Details & Tie-break Rule")
        st.caption(
            "Tie-break order applied when ranking: 1) Higher PPI first → 2) Earlier submission "
            "date → 3) Higher historical experience rating → 4) Supplier name (ascending)."
        )
        if criteria_snapshot_json:
            with st.expander("Criteria snapshot used for this run"):
                st.json(json.loads(criteria_snapshot_json))

        # Full JSON export for this run
        export = {
            "rfp_run_id": selected_run_id,
            "created_at": created_at,
            "completed_at": completed_at,
            "status": status,
            "criteria_snapshot": json.loads(criteria_snapshot_json) if criteria_snapshot_json else {},
            "results": [
                {
                    "supplier_name": r[1],
                    "submission_date": r[2],
                    "experience_rating": r[3],
                    "source_filename": r[4],
                    "status": r[5],
                    "validation_warnings": r[6],
                    "absolute_score": r[7],
                    "ppi": r[8],
                    "final_rank": r[9],
                    "detail": json.loads(r[10]) if r[10] else {},
                }
                for r in results
            ],
        }
        st.download_button(
            "⬇️ Download complete run as JSON",
            data=json.dumps(export, indent=2),
            file_name=f"rfp_run_{selected_run_id}.json",
            mime="application/json",
        )
