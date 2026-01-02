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
                sentence_embeddings, result_cluster, all_questions = qa_cluster.main()
                st.session_state["sentence_embeddings"] = sentence_embeddings
                st.session_state["labels"] = result_cluster
                st.session_state["all_questions"] = all_questions
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

                # st.session_state["kw_best_selected"] = {kw: c}   # {'keyword': number cluster}
                st.session_state["kw_best_item"] = {"keyword": kw, "cluster": c, "score": sc}

                st.success(f"선택 확정: {kw} → Cluster {c} (score={sc:.4f})")

                # 탭 전환
                switch_tab("샘플링")

    else:
        st.info("먼저 키워드를 제출해 주세요.")

    # =============================
    # 📌 Baseline: Random Sampling
    # =============================
    st.markdown("---")
    st.subheader("📌 Baseline: 무작위 샘플링 실행")

    # 샘플 개수 설정 UI
    baseline_n = st.number_input(
        "무작위로 선택할 샘플 개수",
        min_value=5,
        max_value=200,
        value=20,
        step=1,
        key="baseline_sample_count"
    )

    # 버튼 생성
    if st.button("🔀 전체 데이터에서 랜덤 샘플링 실행"):
        all_questions = st.session_state.get("all_questions", None)

        if all_questions is None:
            st.warning("⚠ 먼저 키워드 분석을 실행하여 데이터를 불러오세요.")
        else:
            # np 선택
            rng = np.random.default_rng()   # 원하는 경우 seed 가능

            rand_idx = rng.choice(
                len(all_questions),
                size=baseline_n,
                replace=False
            )
            rand_samples = [all_questions[i] for i in rand_idx]

            # session 저장
            st.session_state["baseline_prompts"] = rand_samples
            st.session_state["baseline_sample_indices"] = rand_idx.tolist()
            st.session_state["baseline_ready"] = True

            st.success(f"랜덤 샘플링 완료! {baseline_n}개 질문 선택됨 → Tab3에서 사용 가능")

            # 미리보기 표시
            st.write("📌 무작위로 선택된 질문 Preview:")
            preview_df = pd.DataFrame({"Random Sample": rand_samples})
            st.dataframe(preview_df.head(10))

