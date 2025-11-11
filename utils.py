import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import csv
from transformers import AutoTokenizer, AutoModel

def run_model():
    tokenizer = AutoTokenizer.from_pretrained('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    model = AutoModel.from_pretrained('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

    return tokenizer, model

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def visualize_clusters(sentence_embeddings, labels, num_clusters, cluster_to_domain):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(sentence_embeddings.detach().cpu().numpy())

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab20', alpha=0.7)
    plt.title('KMeans Clustering of Questions (PCA Visualization)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    handles, _ = scatter.legend_elements()
    plt.legend(handles, [cluster_to_domain[i] for i in range(num_clusters)], 
            title="Domains", 
            loc='center left', 
            bbox_to_anchor=(1.05, 0.5),  # 플롯 바깥 오른쪽에 배치
            borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('kmeans_clusters.png')
    plt.show()

def visualize_embeddings_pca(sentence_embeddings, title='Embeddings PCA'):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(sentence_embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=10)
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.tight_layout()
    plt.savefig('embeddings_pca.png', dpi=200)
    plt.show()

def read_prompts_from_csv(file_path):
    prompts = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            prompts.append(row[0])
    return prompts

def read_prompts_from_txt(file_path):
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            prompts.append(line.strip())
                
    return prompts

def encode_texts(texts):
    tokenizer, model = run_model()
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        out = model(**enc)
    keywords_emb = mean_pooling(out, enc['attention_mask']).cpu().numpy()
    keywords_emb = keywords_emb / np.linalg.norm(keywords_emb, axis=1, keepdims=True)

    return keywords_emb