import streamlit as st
import qa_cluster
import choice_cluster
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_tab1(switch_tab):
    # 1) 키워드 입력 폼
    with st.form(key="qa_form"):
        keyword_input = st.text_area(
            "키워드를 입력하세요",
            placeholder="예: 국방, 일상, 의료 등",
            height=150,
        )
        submit_button = st.form_submit_button(label="키워드 제출")

    # 2) 제출 시: 유사도 계산 → best_list 세션 저장
    if submit_button:
        keywords = [k.strip() for k in keyword_input.split(",") if k.strip()]
        if not keywords:
            st.warning("최소 하나의 키워드를 입력하세요.")
        else:
            with st.spinner("클러스터링 데이터 로드 중..."):
                sentence_embeddings, result_cluster = qa_cluster.main()

            with st.spinner("유사도 계산 중..."):
                sims = choice_cluster.compute_defense_similarity(
                    keywords, sentence_embeddings, result_cluster
                )  # (num_clusters, num_keywords)

                sel_ret = choice_cluster.select_best_clusters_per_keyword(sims, keywords)
                best_list = sel_ret[0] if isinstance(sel_ret, tuple) else sel_ret

            st.session_state["keyword_best"] = best_list  # [{keyword, cluster, score}, ...]

            st.success("분석 완료!")
            num_clusters = sims.shape[0]
            cluster_names = [f"Cluster {i}" for i in range(num_clusters)]
            st.subheader("클러스터 - 키워드 유사도 히트맵")
            df_heat = pd.DataFrame(sims, index=cluster_names, columns=keywords)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df_heat, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, cbar_kws={'label': '유사도'})
            ax.set_title("키워드와 클러스터 간 코사인 유사도")
            st.pyplot(fig)

    # 3) 키워드별 베스트 클러스터 단일 선택(라디오) + 확정 시 (키워드→클러스터)만 저장
    if st.session_state.get("keyword_best"):
        st.subheader("키워드별 베스트 클러스터 선택")

        bl = st.session_state["keyword_best"]  # [{keyword, cluster, score}]
        options = list(range(len(bl)))
        default_idx = st.session_state.get("kw_best_radio_idx", 0)

        with st.form("radio_form", clear_on_submit=False):
            sel_idx = st.radio(
                "하나만 선택하세요:",
                options=options,
                index=min(default_idx, len(options)-1),
                key="kw_best_radio",
                format_func=lambda i: f"{bl[i]['keyword']} → Cluster {bl[i]['cluster']} (score={bl[i]['score']:.4f})",
            )
            st.session_state["kw_best_radio_idx"] = sel_idx

            submitted = st.form_submit_button("선택 확정 (2번째 탭에서 사용)")
            if submitted:
                item = bl[sel_idx]
                kw, c, sc = item["keyword"], int(item["cluster"]), float(item["score"])

                # 필요한 최소한만 세션에 저장 (탭2에서 바로 사용)
                st.session_state["kw_best_selected"] = {kw: c}   # {'keyword': number cluster}
                st.session_state["kw_best_item"] = {"keyword": kw, "cluster": c, "score": sc}

                st.success(f"선택 확정: {kw} → Cluster {c} (score={sc:.4f})")

                # 탭 전환
                switch_tab("샘플링")
    else:
        st.info("먼저 키워드를 제출해 주세요.")
