import qa_cluster
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from utils import mean_pooling, encode_texts

def l2norm(x, axis=1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)

def compute_defense_similarity(keywords, sentence_embeddings, labels):
    # 클러스터별 평균 벡터 계산
    num_clusters = len(set(labels))
    centroids = []
    for cid in range(num_clusters):
        cluster_vecs = sentence_embeddings[labels == cid]
        centroid = cluster_vecs.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)  # L2 정규화
        centroids.append(centroid)
    centroids = np.stack(centroids)  # (K, D)

    keywords_emb = encode_texts(keywords)

    # 키워드별로 클러스터 유사도 나열
    sims = cosine_similarity(centroids, keywords_emb)
    print("\n=== 키워드별 클러스터 유사도 ===")
    for j, kw in enumerate(keywords):
        print(f"[{kw}]")
        for i in range(num_clusters):
            print(f"  Cluster {i}: {sims[i, j]:.4f}")
        best_c = int(np.argmax(sims[:, j]))
        print(f"  → 최고 유사도 클러스터: Cluster {best_c} (score={sims[best_c, j]:.4f})\n")

    return sims
    
def select_best_clusters_per_keyword(sims, keywords):
    best_idx = np.argmax(sims, axis=0)    # (# keywords,)
    best_scores = np.max(sims, axis=0)    # (# keywords,)
    best_list = []
    for kw, ci, sc in zip(keywords, best_idx, best_scores):
        best_list.append({
            "keyword": kw,
            "cluster": int(ci),
            "score": float(sc),
        })
    return best_list

if __name__ == "__main__":
    sentence_embeddings, result_cluster = qa_cluster.main()

    keywords = ['국방', '군사법규', '군사역사', '무기', '군사 조직', '의학', '경제', '법률', '과학', 'IT', '날씨', '일상', '요리', '계절', '예절', '음식']

    sims = compute_defense_similarity(keywords, sentence_embeddings, result_cluster)
    select_best_clusters_per_keyword(sims, keywords)
