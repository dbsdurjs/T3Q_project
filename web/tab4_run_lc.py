import streamlit as st
from LLM_comparator import run_llm_comparator_official_complete as lc

def run_tab4():
    st.header("LLM Comparator 실행 (Tab3 결과 기반)")

    merge_data = st.session_state.get("comparator_payload")
    if merge_data is None:
        st.warning("Tab3에서 먼저 LLM A/B 실행을 완료해야 합니다.")
        return

    model_a_name = merge_data["metadata"]["A_model_name"]
    model_b_name = merge_data["metadata"]["B_model_name"]
    inputs = lc.transform_data(merge_data)

    # 🚀 실행 버튼
    run_clicked = st.button("🚀 LLM Comparator 실행하기")

    if run_clicked:
        with st.spinner("Vertex AI 및 LLM Comparator 실행 중..."):
            credentials = lc.authenticate_vertex_ai(lc.KEY_PATH)
            lc.init_vertex_ai(lc.PROJECT_ID, lc.LOCATION, credentials)

            generator, embedder = lc.init_model_helpers(lc.JUDGE_MODEL, lc.EMBEDDING_MODEL)
            judge, bulletizer, clusterer = lc.init_comparator_components(generator, embedder)

            comparison_result, elapsed_time = lc.run_llm_comparator(
                inputs, judge, bulletizer, clusterer, model_a_name, model_b_name
            )

            output_file = lc.save_comparison_result(comparison_result, lc.OUTPUT_DIR)

            # 👉 세션에 저장
            st.session_state["lc_output_file"] = output_file
            st.session_state["lc_elapsed_time"] = elapsed_time

        st.success("LLM Comparator 실행 완료!")
        st.write(f"소요 시간: {elapsed_time/60:.1f}분")
        st.write(f"결과 JSON: `{output_file}`")

    # ✅ 항상 렌더되지만, 파일이 있을 때만 동작
    open_clicked = st.button("VSCode LLM Comparator Web UI 열기")

    if open_clicked:
        output_file = st.session_state.get("lc_output_file")
        if not output_file:
            st.error("먼저 LLM Comparator를 실행해 결과 파일을 생성하세요.")
        else:
            try:
                lc.open_vscode_viewer(output_file)
                st.info("터미널/로컬 환경에서 VSCode Web UI가 실행되었습니다. (Streamlit 창에서는 직접 보이지 않습니다.)")
            except Exception as e:
                st.error(f"VSCode Web UI 실행 중 오류 발생: {e}")
