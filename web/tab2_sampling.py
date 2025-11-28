import streamlit as st
from utils import encode_texts
from mmr_select import mmr_
import numpy as np
import pandas as pd

def run_tab2(switch_tab):
    st.header("샘플링 MMR 방식")

    kw_best = st.session_state.get("kw_best_item")
    if not kw_best:
        st.warning("탭 1에서 키워드-클러스터를 먼저 선택 확정하세요.")
        return

    sel_keyword = kw_best["keyword"]
    sel_cluster = kw_best["cluster"]
    st.write(f"확정: **{sel_keyword} → Cluster {sel_cluster}**")

    # 2) 후보 집합 로드: 클러스터 내부 임베딩
    with st.spinner("후보 임베딩 로드 중..."):
        sentence_embeddings = st.session_state.get("sentence_embeddings")
        labels = st.session_state.get("labels")
        all_questions = st.session_state.get("all_questions")

        labels = np.array(labels)
        cand_idx = np.where(labels == int(sel_cluster))[0]
        if len(cand_idx) == 0:
            st.error("선택된 클러스터에 후보가 없습니다.")
            return
        X = sentence_embeddings[cand_idx]  # (Ncand, D)

    # 3) 파라미터/하위 키워드 입력 폼
    with st.form("mmr_form"):
        sub_kw_input = st.text_area(
            "하위 키워드(쉼표로 구분)",
            placeholder="예: 전력증강, 동원체계, UGV, 지휘통제, 예비군",
            height=120,
        )
        st.info(f"후보 개수: {int(len(cand_idx))}")
        st.caption("※ 아래에서 입력하는 샘플 개수는 '후보 개수' 이내의 값만 선택할 수 있습니다.")
        n_samples = st.number_input(
            "샘플 개수 (후보 개수 이내로 입력)",
            min_value=1,
            max_value=int(len(cand_idx)),
            value=min(3, int(len(cand_idx))),
            help=f"현재 선택된 클러스터의 후보 개수는 {int(len(cand_idx))}개입니다.\n1 이상 {int(len(cand_idx))} 이하의 정수만 입력할 수 있습니다."
        )

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
            sel_local_idx, rel = mmr_(
                X, kw_embs,
                n_samples=int(n_samples),
                alpha=float(alpha),
                beta=float(beta),
                agg=agg,
                dedup_thr=float(dedup_thr)
            )

        sampled_global_idx = cand_idx[sel_local_idx]
        selected_sentences = [all_questions[i] for i in sampled_global_idx]
        st.session_state["eval_prompts"] = selected_sentences

        # 결과 표/다운로드
        st.success(f"샘플링 완료! 선택 {len(sampled_global_idx)}개")
        df_out = pd.DataFrame({
            "query": selected_sentences,
            "mmr_rel": rel[sel_local_idx]
        })
        st.dataframe(df_out, width='stretch')
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
        st.session_state["mmr_done"] = True

    if st.session_state.get("mmr_done"):
        st.info("MMR 샘플링이 완료되었습니다. 평가 데이터셋 만들기 탭으로 이동할 수 있습니다.")
        if st.button("평가 데이터셋 만들기로 이동"):
            switch_tab("평가 데이터셋 만들기")
