import streamlit as st
from LLM_comparator import run_llm_comparator_official_complete as lc

def run_tab4():
    st.header("LLM Comparator 실행 (Tab3 결과 기반)")

    # 1) Tab3에서 만든 payload 가져오기
    merge_data = st.session_state.get("comparator_payload")
    if merge_data is None:
        st.warning("Tab3에서 먼저 LLM A/B 실행을 완료해야 합니다.")
        return

    # 2) model 이름, inputs 준비 (✅ 여기서 JSON 대신 session 사용)
    model_a_name = merge_data["metadata"]["A_model_name"]
    model_b_name = merge_data["metadata"]["B_model_name"]
    inputs = lc.transform_data(merge_data)

    # if st.button("🚀 LLM Comparator 실행하기"):
    #     with st.spinner("Vertex AI 및 LLM Comparator 실행 중..."):
    #         # 3) 인증 & 초기화
    #         credentials = lc.authenticate_vertex_ai(lc.KEY_PATH)
    #         lc.init_vertex_ai(lc.PROJECT_ID, lc.LOCATION, credentials)

    #         # 4) 모델 헬퍼 및 컴포넌트 초기화
    #         generator, embedder = lc.init_model_helpers(lc.JUDGE_MODEL, lc.EMBEDDING_MODEL)
    #         judge, bulletizer, clusterer = lc.init_comparator_components(generator, embedder)

    #         # 5) LLM Comparator 실행
    #         comparison_result, elapsed_time = lc.run_llm_comparator(
    #             inputs, judge, bulletizer, clusterer, model_a_name, model_b_name
    #         )

    #         # 6) 결과 저장
    #         output_file = lc.save_comparison_result(comparison_result, lc.OUTPUT_DIR)

        # st.success("LLM Comparator 실행 완료!")
        # st.write(f"소요 시간: {elapsed_time/60:.1f}분")
        # st.write(f"결과 JSON: `{output_file}`")
    
    output_file = "../llm_comparison_results/llm_comparator_auto_viewer_20251205_141952.json"  # 예시 파일명, 실제로는 저장된 파일 경로 사용
    if st.button("VSCode LLM Comparator Web UI 열기"):
        try:
            lc.open_vscode_viewer(output_file)
            st.info("터미널/로컬 환경에서 VSCode Web UI가 실행되었습니다. (Streamlit 창에서는 직접 보이지 않습니다.)")
        except Exception as e:
            st.error(f"VSCode Web UI 실행 중 오류 발생: {e}")