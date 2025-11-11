import streamlit as st
from model_a_response_qa import get_llm_a_response   # 내부에서 고정 모델 사용
from model_b_response_qa import get_llm_b_response   # 내부에서 고정 모델 사용
import pandas as pd
import json

def run_tab3():
    st.subheader("LLM 결과 생성")

    # === 실행 옵션 ===
    with st.expander("실행 옵션", expanded=True):
        num_print = st.number_input("미리보기 개수(num_print)", min_value=1, value=50, step=1)

    # === 평가 프롬프트 가져오기 ===
    eval_prompts = st.session_state.get("eval_prompts", None)
    if not eval_prompts:
        st.warning("평가 데이터셋 이 없습니다. 먼저 '샘플링' 탭에서 평가 프롬프트를 생성하세요.")
        return

    # 실제 실행할 프롬프트 (limit 적용)
    eval_datasets = eval_prompts[:limit]

    # === 실행 버튼 ===
    run = st.button("🔎 실행")
    if not run:
        return

    # === 모델 A/B 실행 (각 함수는 [{prompt, response_a}], [{prompt, response_b}] 형태 반환 가정) ===
    with st.spinner("모델 A 실행 중..."):
        res_a = get_llm_a_response(eval_datasets)
    with st.spinner("모델 B 실행 중..."):
        res_b = get_llm_b_response(eval_datasets)

    # === DataFrame 변환 및 병합 ===
    df_a = pd.DataFrame(res_a)
    df_b = pd.DataFrame(res_b)
    # prompt 기준 outer merge (어느 한쪽에만 있더라도 보존)
    df = pd.merge(df_a, df_b, on="prompt", how="outer")

    st.success(f"생성 완료! (총 {len(df)}개)")

    # === 미리보기: num_print 개수만 표시 ===
    preview = df.head(num_print)

    tab_a, tab_b, tab_merge = st.tabs(["A 결과", "B 결과", "병합 보기"])
    with tab_a:
        st.dataframe(df_a.head(num_print), use_container_width=True, height=480)
    with tab_b:
        st.dataframe(df_b.head(num_print), use_container_width=True, height=480)
    with tab_merge:
        st.dataframe(preview, use_container_width=True, height=600)

    # === 다운로드용 변환 ===
    # 1) JSON (배열) : [{prompt, response_a, response_b}, ...]
    merged_records = []
    for _, row in df.iterrows():
        merged_records.append({
            "prompt": row.get("prompt", ""),
            "response_a": row.get("response_a", ""),
            "response_b": row.get("response_b", "")
        })
    json_array_bytes = json.dumps(merged_records, ensure_ascii=False, indent=2).encode("utf-8")

    # 2) JSONL (레코드별 1줄)
    jsonl_lines = "\n".join([json.dumps(r, ensure_ascii=False) for r in merged_records]).encode("utf-8")

    # 3) TXT (네가 쓰던 포맷)
    txt_lines = "\n".join([
        f"{{prompt:{r['prompt']}, response_a:{r.get('response_a','')}, response_b:{r.get('response_b','')}}}"
        for r in merged_records
    ]).encode("utf-8")

    # === 다운로드 버튼 ===
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.download_button("⬇️ 병합 JSON (배열)", data=json_array_bytes,
                           file_name="llm_AB_merge.json", mime="application/json")
    with col_d2:
        st.download_button("⬇️ 병합 JSONL", data=jsonl_lines,
                           file_name="llm_AB_merge.jsonl", mime="application/json")
    with col_d3:
        st.download_button("⬇️ 병합 TXT (기존 포맷)", data=txt_lines,
                           file_name="llm_AB_merge.txt", mime="text/plain")
