import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from utils import encode_texts


def run_tab2_1(switch_tab):
    st.header("📊 MMR 샘플링 vs Random 샘플링 비교 분석")

    # =============================
    # 1) 필수 데이터 확인
    # =============================
    sentence_embeddings = st.session_state.get("sentence_embeddings")
    all_questions = st.session_state.get("all_questions")
    mmr_idx = st.session_state.get("mmr_sample_indices")
    rand_idx = st.session_state.get("baseline_sample_indices")
    mmr_params = st.session_state.get("mmr_params", {})

    if sentence_embeddings is None or all_questions is None:
        st.warning("⚠ Tab1을 먼저 실행하여 데이터 로드 & 임베딩을 저장하세요.")
        return
    if mmr_idx is None:
        st.warning("⚠ Tab2 (MMR 샘플링)를 먼저 실행하세요.")
        return
    if rand_idx is None:
        st.warning("⚠ Tab1에서 Random Baseline 샘플링을 먼저 실행하세요.")
        return

    mmr_idx = np.array(mmr_idx)
    rand_idx = np.array(rand_idx)

    # =============================
    # 2) 유사도 계산 키워드 세트 만들기
    # =============================
    main_keyword = mmr_params.get("sel_keyword", None)
    sub_keywords = mmr_params.get("sub_keywords", [])
    if main_keyword is None:
        st.warning("⚠ MMR 메인 키워드가 없습니다. Tab2를 먼저 실행하세요.")
        return

    keywords_for_similarity = [main_keyword] + sub_keywords

    st.write("🔎 유사도 계산에 사용된 키워드:")
    st.write(", ".join(keywords_for_similarity))

    # =============================
    # 3) 키워드별 유사도 계산 (각 키워드 따로)
    # =============================
    kw_embs = encode_texts(keywords_for_similarity)  # (K, D)
    sims = cosine_similarity(sentence_embeddings, kw_embs)  # (N, K)

    rows = []
    for k_idx, kw in enumerate(keywords_for_similarity):
        mmr_sims_k = sims[mmr_idx, k_idx]   # (len(mmr_idx),)
        rand_sims_k = sims[rand_idx, k_idx] # (len(rand_idx),)

        for s in mmr_sims_k:
            rows.append({
                "keyword": kw,
                "method": "MMR",
                "similarity": float(s),
            })
        for s in rand_sims_k:
            rows.append({
                "keyword": kw,
                "method": "Random",
                "similarity": float(s),
            })

    df_vis = pd.DataFrame(rows)

    # =============================
    # 4) 키워드별 요약 표
    # =============================
    st.markdown("### 🔸 키워드별 통계 요약")

    summary = (
        df_vis
        .groupby(["keyword", "method"])["similarity"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_sim", "std": "std_sim"})
    )
    st.dataframe(summary)

    # =============================
    # 5) 상세 분석용 키워드 선택
    # =============================
    st.markdown("### 🔸 상세 분석 키워드 선택")
    sel_kw = st.selectbox(
        "어떤 키워드에 대해 분포를 자세히 볼까요?",
        keywords_for_similarity,
    )

    df_sel = df_vis[df_vis["keyword"] == sel_kw].copy()

    st.write(f"선택된 키워드: **{sel_kw}**")

    # =============================
    # 6) 분포 비교 (KDE)
    # =============================
    st.markdown("### 🔸 분포 비교 (KDE)")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(
        data=df_sel,
        x="similarity",
        hue="method",
        fill=True,
        common_norm=False,
        alpha=0.4,
        ax=ax,
    )
    ax.set_xlabel(f"Keyword relevance score (키워드: {sel_kw})")
    ax.set_title(f"MMR vs Random - similarity distribution for '{sel_kw}'")
    st.pyplot(fig)

    st.caption("""
KDE(확률 밀도 곡선)는 선택한 키워드에 대해
각 샘플링 방식이 얼마나 '유사도가 높은 질문'을 많이 선택했는지 보여줍니다.
곡선이 오른쪽으로 치우칠수록 해당 키워드와 더 관련 있는 질문이 많다는 뜻입니다.
""")

    # =============================
    # 7) Boxplot
    # =============================
    st.markdown("### 🔸 박스플롯 비교")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df_sel, x="method", y="similarity", ax=ax2)
    ax2.set_ylabel("Keyword relevance score")
    ax2.set_title(f"MMR vs Random for '{sel_kw}'")
    st.pyplot(fig2)

    # =============================
    # 8) 선택된 키워드 기준 질문 예시
    # =============================
    st.markdown("### 🔸 선택된 키워드 기준 질문 예시")

    # 선택된 키워드에 대한 raw similarity 벡터 다시 계산
    k_sel_idx = keywords_for_similarity.index(sel_kw)
    mmr_sims_sel = sims[mmr_idx, k_sel_idx]
    rand_sims_sel = sims[rand_idx, k_sel_idx]

    # ---------- MMR ----------
    st.subheader("📌 MMR 샘플 선정 결과")

    st.write("**상위 5개 (해당 키워드와 가장 유사한 질문)**")
    top_mmr = np.argsort(-mmr_sims_sel)[:5]
    for i in top_mmr:
        q = all_questions[mmr_idx[i]]
        st.write(f"- ({mmr_sims_sel[i]:.3f}) {q}")

    st.write("**하위 5개 (해당 키워드와 가장 덜 관련된 질문)**")
    bottom_mmr = np.argsort(mmr_sims_sel)[:5]
    for i in bottom_mmr:
        q = all_questions[mmr_idx[i]]
        st.write(f"- ({mmr_sims_sel[i]:.3f}) {q}")

    st.markdown("---")

    # ---------- Random ----------
    st.subheader("📌 Random 샘플 선정 결과")

    st.write("**상위 5개 (해당 키워드와 가장 유사한 질문)**")
    top_rand = np.argsort(-rand_sims_sel)[:5]
    for i in top_rand:
        q = all_questions[rand_idx[i]]
        st.write(f"- ({rand_sims_sel[i]:.3f}) {q}")

    st.write("**하위 5개 (해당 키워드와 가장 덜 관련된 질문)**")
    bottom_rand = np.argsort(rand_sims_sel)[:5]
    for i in bottom_rand:
        q = all_questions[rand_idx[i]]
        st.write(f"- ({rand_sims_sel[i]:.3f}) {q}")
