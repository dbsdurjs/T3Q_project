def mmr_select(
    X,                    # (N, D) 후보(선택된 클러스터 내부) 임베딩
    kw_embs,              # (K, D) 하위 키워드 임베딩
    n_samples=50,
    alpha=0.8,            # 관련성 비중 (0~1)
    beta=0.6,             # 중복 억제 비중 (0~1)  (실제로는 (1-alpha)로 써도 되나 독립 가중치로 둠)
    agg="max",            # 'max' | 'mean' | 'min'  (하위 키워드 관련성 집계 방식)
    dedup_thr=0.92        # 뽑힌 샘플 간 코사인 유사도 상한
):
    # 1) 하위 키워드 관련성 계산
    sims = cosine_similarity(X, kw_embs)  # (N, K)
    if agg == "max":
        rel = sims.max(axis=1)
    elif agg == "mean":
        rel = sims.mean(axis=1)
    elif agg == "min":
        rel = sims.min(axis=1)
    else:
        raise ValueError("agg must be 'max' | 'mean' | 'min'")

    # 2) 후보-후보 유사도(중복 측정용)
    XX = cosine_similarity(X)  # (N, N)

    # 3) Greedy MMR
    selected = []
    candidate = set(range(X.shape[0]))
    while len(selected) < min(n_samples, X.shape[0]) and candidate:
        best_i, best_score = None, -1e9
        for i in candidate:
            red = 0.0 if not selected else XX[i, selected].max()
            score = alpha * rel[i] - beta * red
            if score > best_score:
                best_score, best_i = score, i

        # near-duplicate 필터
        if selected and XX[best_i, selected].max() >= dedup_thr:
            candidate.remove(best_i)
            continue

        selected.append(best_i)
        candidate.remove(best_i)

    return np.array(selected), rel
