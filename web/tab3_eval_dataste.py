import streamlit as st
from model_a_response_qa import get_llm_a_response   # 여기서는 "배치 리스트 입력 → 리스트 출력" 함수라고 가정
from model_b_response_qa import get_llm_b_response
import pandas as pd
import json

def run_tab3(switch_tab):
    st.subheader("LLM 결과 생성")

    # === 평가 프롬프트 가져오기 ===
    eval_prompts = st.session_state.get("eval_prompts", None)
    if not eval_prompts:
        st.warning("평가 데이터셋이 없습니다. 먼저 '샘플링' 탭에서 평가 프롬프트를 생성하세요.")
        return

    total = len(eval_prompts)
    st.info(f"현재 샘플링된 평가 프롬프트 개수: {total}개")

    # 예시 출력 개수 (슬라이더는 매번 새로 그려져도 상관 없음)
    num_print = st.slider(
        "예시 출력 개수",
        min_value=1,
        max_value=total,
        value=min(10, total),
        step=1,
        key="tab3_num_print"
    )

    # === 세션 상태 초기화 ===
    if "tab3_df" not in st.session_state:
        st.session_state["tab3_df"] = None
        st.session_state["tab3_df_a"] = None
        st.session_state["tab3_df_b"] = None

    run = st.button("🔎 LLM A/B 실행")

    if run:
        results_a = []
        results_b = []
        with st.spinner("모델 A/B 실행 중..."):
            print(eval_prompts)
            for prompt in eval_prompts:
                print(f"Processing prompt: {prompt}")
                answers_a = get_llm_a_response(prompt)
                answers_b = get_llm_b_response(prompt)
                print(f"Model A answers: {answers_a}")
                print(f"Model B answers: {answers_b}")
                results_a.append({"prompt": prompt, "response_a": answers_a})
                results_b.append({"prompt": prompt, "response_b": answers_b})

        # === DataFrame 변환 ===
        df_a = pd.DataFrame(results_a)  # columns: prompt, response_a
        df_b = pd.DataFrame(results_b)  # columns: prompt, response_b
        df = pd.merge(df_a, df_b, on="prompt", how="outer")

        st.session_state["tab3_df"] = df
        st.session_state["tab3_df_a"] = df_a
        st.session_state["tab3_df_b"] = df_b

        st.success(f"생성 완료! (총 {len(df)}개)")

    # 여기서 세션에 결과가 없으면(아직 실행 전이면) 그냥 종료
    if st.session_state["tab3_df"] is None:
        return

    # === 세션에서 결과 불러오기 ===
    df = st.session_state["tab3_df"]
    df_a = st.session_state["tab3_df_a"]
    df_b = st.session_state["tab3_df_b"]

    preview = df.head(int(num_print))

    tab_a, tab_b, tab_merge = st.tabs(["A 결과", "B 결과", "병합 보기"])
    with tab_a:
        st.dataframe(df_a.head(int(num_print)))
    with tab_b:
        st.dataframe(df_b.head(int(num_print)))
    with tab_merge:
        st.dataframe(preview)

    # === 다운로드용 변환 ===
    merged_records = []
    for _, row in df.iterrows():
        merged_records.append({
            "prompt": row.get("prompt", ""),
            "response_a": row.get("response_a", ""),
            "response_b": row.get("response_b", "")
        })

    json_array_bytes = json.dumps(
        merged_records, ensure_ascii=False, indent=2
    ).encode("utf-8")

    jsonl_lines = "\n".join([
        json.dumps(r, ensure_ascii=False) for r in merged_records
    ]).encode("utf-8")

    txt_lines = "\n".join([
        f"{{prompt:{r['prompt']}, response_a:{r.get('response_a','')}, response_b:{r.get('response_b','')}}}"
        for r in merged_records
    ]).encode("utf-8")

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.download_button(
            "⬇️ 병합 JSON (배열)",
            data=json_array_bytes,
            file_name="llm_AB_merge.json",
            mime="application/json"
        )
    with col_d2:
        st.download_button(
            "⬇️ 병합 JSONL",
            data=jsonl_lines,
            file_name="llm_AB_merge.jsonl",
            mime="application/json"
        )
    with col_d3:
        st.download_button(
            "⬇️ 병합 TXT (기존 포맷)",
            data=txt_lines,
            file_name="llm_AB_merge.txt",
            mime="text/plain"
        )

    if st.button("LLM Comparator로 이동"):
        switch_tab("LLM Comparator 실행")
