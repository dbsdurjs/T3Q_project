import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import qa_cluster
from utils import encode_texts
import mmr_sampling

def run_tab2(switch_tab):
    st.header("샘플링 MMR 방식")

    # 1) 탭1에서 확정한 (키워드→클러스터) 확인
    kw_best = st.session_state.get("kw_best_selected")
    if not kw_best:
        st.warning("탭 1에서 키워드-클러스터를 먼저 선택 확정하세요.")
        return

    # (한 개만 이미 확정되어 있다고 가정)
    sel_keyword, sel_cluster = next(iter(kw_best.items()))
    st.write(f"확정: **{sel_keyword} → Cluster {sel_cluster}**")

    # 2) 후보 집합 로드: 클러스터 내부 임베딩
    with st.spinner("후보 임베딩 로드 중..."):
        # qa_cluster.main()에서 (sentence_embeddings, labels) 재구성
        sentence_embeddings, labels = qa_cluster.main()
        labels = np.array(labels)
        cand_idx = np.where(labels == int(sel_cluster))[0]
        if len(cand_idx) == 0:
            st.error("선택된 클러스터에 후보가 없습니다.")
            return
        X = sentence_embeddings[cand_idx]  # (Ncand, D)
        st.info(f"후보 개수: {len(cand_idx)}")

    # 3) 파라미터/하위 키워드 입력 폼
    with st.form("mmr_form"):
        sub_kw_input = st.text_area(
            "하위 키워드(쉼표로 구분)",
            placeholder="예: 전력증강, 동원체계, UGV, 지휘통제, 예비군",
            height=120,
        )
        n_samples = st.number_input("샘플 개수", min_value=1, max_value=int(len(cand_idx)), value=min(50, int(len(cand_idx))))
        alpha     = st.slider("관련성 비중 α", 0.0, 1.0, 0.8, 0.05)
        beta      = st.slider("중복 억제 비중 β", 0.0, 1.0, 0.6, 0.05)
        agg       = st.selectbox("하위 키워드 관련성 집계", ["max", "mean", "min"])
        dedup_thr = st.slider("near-duplicate 임계치", 0.80, 0.99, 0.92, 0.01)

        submitted = st.form_submit_button("MMR 샘플링 실행")

    # 4) 실행
    if submitted:
        sub_keywords = [k.strip() for k in sub_kw_input.split(",") if k.strip()]
        if len(sub_keywords) == 0:
            st.warning("하위 키워드를 1개 이상 입력하세요.")
            return

        with st.spinner("하위 키워드 임베딩 계산..."):
            kw_embs = encode_texts(sub_keywords)
            if kw_embs.ndim != 2 or kw_embs.shape[0] == 0:
                return

        with st.spinner("MMR 샘플링 중..."):
            sel_local_idx, rel = mmr_select(
                X, kw_embs,
                n_samples=int(n_samples),
                alpha=float(alpha),
                beta=float(beta),
                agg=agg,
                dedup_thr=float(dedup_thr)
            )

        # 후보 전체 인덱스(cand_idx)에서 선택된 로컬 인덱스(sel_local_idx)로 전역 인덱스 환산
        sampled_global_idx = cand_idx[sel_local_idx]

        # 결과 표/다운로드
        st.success(f"샘플링 완료! 선택 {len(sampled_global_idx)}개")
        df_out = pd.DataFrame({
            "global_index": sampled_global_idx,
            "mmr_rel": rel[sel_local_idx]
        })
        st.dataframe(df_out, use_container_width=True)
        st.download_button(
            "샘플 인덱스 CSV 다운로드",
            df_out.to_csv(index=False).encode("utf-8"),
            file_name="mmr_sample_indices.csv",
            mime="text/csv"
        )

        # 세션 저장 → 탭3에서 사용
        st.session_state["mmr_sample_indices"] = sampled_global_idx.tolist()
        st.session_state["mmr_params"] = {
            "sel_keyword": sel_keyword,
            "sel_cluster": int(sel_cluster),
            "sub_keywords": sub_keywords,
            "alpha": float(alpha),
            "beta": float(beta),
            "agg": agg,
            "dedup_thr": float(dedup_thr),
            "n_samples": int(n_samples)
        }

        # 탭3로 이동
        st.info("평가 데이터셋 만들기 탭으로 이동합니다.")
        switch_tab("평가 데이터셋 만들기")
