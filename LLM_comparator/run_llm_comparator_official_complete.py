#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Comparator - 공식 클러스터링 포함 함수형 정리 버전
"""

import os
import json
import sys
from datetime import datetime

import vertexai
from google.oauth2 import service_account

# 맨 위 import 부분을 이렇게 수정
from LLM_comparator.llm_comparator import comparison
from LLM_comparator.llm_comparator import llm_judge_runner
from LLM_comparator.llm_comparator import rationale_bullet_generator
from LLM_comparator.llm_comparator import rationale_cluster_generator
from LLM_comparator.llm_comparator import custom_model_helper

# ==========================
# 상수 설정 (필요한 부분만 수정)
# ==========================
KEY_PATH = "../army22-12412f909096.json"  # 서비스 계정 키 경로
PROJECT_ID = "army22"
LOCATION = "us-central1"

JUDGE_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "text-embedding-004"
MAX_OUTPUT_TOKENS = 2048  # 필요하면 custom_model_helper에서 사용

OUTPUT_DIR = "../llm_comparison_results"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(KEY_PATH)
print("GOOGLE_APPLICATION_CREDENTIALS =", os.environ["GOOGLE_APPLICATION_CREDENTIALS"])

# ==========================
# 1. 인증 & Vertex AI 초기화
# ==========================
def authenticate_vertex_ai(key_path: str) -> service_account.Credentials:
    """서비스 계정 키로 Vertex AI 인증."""
    print("\n[1단계] Vertex AI 인증")
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"키 파일을 찾을 수 없음: {key_path}")
    print(f"키 파일 확인됨: {key_path}")
    credentials = service_account.Credentials.from_service_account_file(
        key_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    print("  ✓ 인증 완료")
    return credentials


def init_vertex_ai(project: str, location: str, credentials) -> None:

    print("\n[2단계] Vertex AI 초기화")
    vertexai.init(
        project=project,
        location=location,
        credentials=credentials,
    )
    print("  ✓ 초기화 완료")


# ==========================
# 2. 데이터 로드 및 변환
# ==========================
def load_llm_outputs(llm1_file: str, llm2_file: str):
    """LLM1/LLM2의 출력 JSON 파일을 로드."""
    print("\n[3단계] 데이터 로드")

    with open(llm1_file, "r", encoding="utf-8") as f:
        llm1_data = json.load(f)

    with open(llm2_file, "r", encoding="utf-8") as f:
        llm2_data = json.load(f)

    print(f"  ✓ LLM1 응답: {len(llm1_data['examples'])}개")
    print(f"  ✓ LLM2 응답: {len(llm2_data['examples'])}개")

    model_a_name = llm1_data["metadata"]["model_name"]
    model_b_name = llm2_data["metadata"]["model_name"]

    print(f"  ✓ Model A: {model_a_name}")
    print(f"  ✓ Model B: {model_b_name}")

    return llm1_data, llm2_data, model_a_name, model_b_name


def build_comparator_inputs(llm1_data, llm2_data):
    """LLM Comparator에서 사용하는 inputs 포맷으로 변환."""
    print("\n[4단계] 데이터 변환")
    inputs = []
    for item1, item2 in zip(llm1_data["examples"], llm2_data["examples"]):
        inputs.append(
            {
                "prompt": item1["prompt"],
                "response_a": item1["response"],
                "response_b": item2["response"],
            }
        )

    print(f"  ✓ {len(inputs)}개 질문 준비 완료")
    return inputs

def transform_data(merge_data):
    inputs = []
    for item in merge_data['examples']:
        inputs.append(
            {
                "prompt": item["input_text"],
                "response_a": item["output_text_a"],
                "response_b": item["output_text_b"],
            }
        )

    print(f"  ✓ {len(inputs)}개 질문 준비 완료")
    return inputs

# ==========================
# 3. 모델 헬퍼 & 컴포넌트 초기화
# ==========================
def init_model_helpers(judge_model: str, embedding_model: str):
    """Judge/Embedding용 Vertex 모델 헬퍼 초기화."""
    print("\n[5단계] 모델 헬퍼 초기화")
    print(f"  - Judge Model: {judge_model}")
    print(f"  - Embedding Model: {embedding_model}")
    print(f"  - Max Output Tokens: {MAX_OUTPUT_TOKENS}")

    generator = custom_model_helper.VertexGenerationModelHelper(judge_model)
    embedder = custom_model_helper.VertexEmbeddingModelHelper(embedding_model)
    print("  ✓ 모델 헬퍼 준비 완료")

    return generator, embedder

def init_comparator_components(generator, embedder):
    """LLM Comparator의 Judge/Bulletizer/Clusterer 초기화."""
    print("\n[6단계] LLM Comparator 컴포넌트 초기화")
    judge = llm_judge_runner.LLMJudgeRunner(generator)
    bulletizer = rationale_bullet_generator.RationaleBulletGenerator(generator)
    clusterer = rationale_cluster_generator.RationaleClusterGenerator(
        generator, embedder
    )
    print("  ✓ Judge, Bulletizer, Clusterer 준비 완료")
    return judge, bulletizer, clusterer

# ==========================
# 4. LLM Comparator 실행
# ==========================
def run_llm_comparator(inputs, judge, bulletizer, clusterer, model_a_name, model_b_name):
    """LLM Comparator를 실행하고 결과와 실행 시간을 반환."""
    print("\n[7단계] LLM Comparator 실행 (공식 클러스터링 포함)")
    print(f"  - 총 질문 수: {len(inputs)}개")
    print(f"\n  ⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    start_time = datetime.now()

    try:
        comparison_result = comparison.run(
            inputs,
            judge,
            bulletizer,
            clusterer,
            model_names=(model_a_name, model_b_name),
        )
    except Exception as e:
        print(f"\n  ❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        raise

    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()

    print(f"\n  ✓ LLM Comparator 실행 완료")
    print(f"  ⏰ 종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  ⏱️  소요 시간: {elapsed_time / 60:.1f}분")

    return comparison_result, elapsed_time

# ==========================
# 5. 결과 저장
# ==========================
def save_comparison_result(comparison_result, output_dir: str) -> str:
    """비교 결과를 JSON 파일로 저장하고 경로를 반환."""
    print("\n[8단계] 결과 저장")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/llm_comparator_auto_viewer_{timestamp}.json"

    comparison.write(comparison_result, output_file)
    print(f"  ✓ 결과 저장: {output_file}")
    return output_file

# ==========================
# 6. 통계 출력
# ==========================
def print_overall_stats(comparison_result, judge_model, model_a_name, model_b_name):
    """승/패/동점 및 기본 통계 출력."""
    print("\n" + "=" * 80)
    print("평가 완료!")
    print("=" * 80)

    examples = comparison_result["examples"]
    scores = [ex["score"] for ex in examples]

    a_wins = sum(1 for s in scores if s > 0)
    b_wins = sum(1 for s in scores if s < 0)
    ties = sum(1 for s in scores if s == 0)

    print(f"\n📊 결과 요약:")
    print(f"  • 총 평가 쌍: {len(examples)}개")
    print(f"  • Judge 모델: {judge_model}")
    print(f"  • Model A ({model_a_name}): {a_wins}승 ({a_wins / len(examples) * 100:.1f}%)")
    print(f"  • Model B ({model_b_name}): {b_wins}승 ({b_wins / len(examples) * 100:.1f}%)")
    print(f"  • 동점: {ties}개 ({ties / len(examples) * 100:.1f}%)")
    print(f"  • 평균 점수 차이: {sum(scores) / len(scores):.3f}")


def print_rationale_stats(comparison_result):
    """Rationale 관련 통계 출력."""
    examples = comparison_result["examples"]

    rationale_count = 0
    total_ratings = 0
    for ex in examples:
        individual_scores = ex.get("individual_rater_scores", [])
        total_ratings += len(individual_scores)
        for score_item in individual_scores:
            if isinstance(score_item, dict) and score_item.get("rationale"):
                rationale_count += 1

    print(f"\n📝 Rationale 통계:")
    print(f"  • 총 평가 횟수: {total_ratings}회")
    print(f"  • Rationale 포함: {rationale_count}회")
    if total_ratings > 0:
        print(f"  • Rationale 비율: {rationale_count / total_ratings * 100:.1f}%")


def print_cluster_stats(comparison_result):
    """클러스터링 통계 출력."""
    clusters = comparison_result.get("rationale_clusters", [])
    if not clusters:
        print(f"\n⚠️  클러스터링 정보 없음")
        return

    print(f"\n🔍 클러스터링 통계:")
    print(f"  • 클러스터 수: {len(clusters)}개")
    print(f"\n  클러스터 목록:")
    for i, cluster in enumerate(clusters, 1):
        title = cluster.get("title", f"Cluster {i}")
        print(f"    {i}. {title}")


def print_top_examples(comparison_result, top_k: int = 5):
    """상위 top_k개 예시 출력."""
    examples = comparison_result["examples"]
    print(f"\n📋 상위 {top_k}개 질문 결과:")
    for i, ex in enumerate(examples[:top_k], 1):
        text_preview = ex.get("input_text", "")[:60]
        print(f"\n  [{i}] {text_preview}...")
        score = ex["score"]
        print(f"      점수: {score:.2f}", end="")
        if score > 0.5:
            print(" → Model A 승리")
        elif score < -0.5:
            print(" → Model B 승리")
        else:
            print(" → 비슷함")

# ==========================
# 7. VSCode 웹 UI 자동 실행
# ==========================
def open_vscode_viewer(output_file: str):
    """VSCode 환경에서 LLM Comparator 웹 UI를 자동 실행."""
    print("\n" + "=" * 80)
    print("[10단계] VSCode에서 웹 UI 자동 실행")
    print("=" * 80)

    try:
        # 패치된 comparison.py의 show_in_vscode() 사용
        comparison.show_in_vscode(output_file)
    except KeyboardInterrupt:
        print("\n\n✅ 사용자가 서버를 종료했습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print(f"\n수동으로 확인하려면:")
        print("  1. https://pair-code.github.io/llm-comparator/ 접속")
        print("  2. 'Load data' 버튼 클릭")
        print(f"  3. {output_file} 업로드")

# ==========================
# main
# ==========================
def main():
    print("=" * 80)
    print("LLM Comparator - 공식 클러스터링 포함 완전 버전")
    print("=" * 80)

    # 1) 인증 및 초기화
    credentials = authenticate_vertex_ai(KEY_PATH)
    init_vertex_ai(PROJECT_ID, LOCATION, credentials)

    with open('../llm_comparator_input.json', "r", encoding="utf-8") as f:
        merge_data = json.load(f)

    model_a_name = merge_data["metadata"]["A_model_name"]
    model_b_name = merge_data["metadata"]["B_model_name"] 
    inputs = transform_data(merge_data)

    # 3) 모델 헬퍼 및 컴포넌트 초기화
    generator, embedder = init_model_helpers(JUDGE_MODEL, EMBEDDING_MODEL)
    judge, bulletizer, clusterer = init_comparator_components(generator, embedder)

    # 4) LLM Comparator 실행
    comparison_result, _ = run_llm_comparator(
        inputs, judge, bulletizer, clusterer, model_a_name, model_b_name
    )

    # 5) 결과 저장
    output_file = save_comparison_result(comparison_result, OUTPUT_DIR)

    # 6) 통계 출력
    print_overall_stats(comparison_result, JUDGE_MODEL, model_a_name, model_b_name)
    print_rationale_stats(comparison_result)
    print_cluster_stats(comparison_result)
    print_top_examples(comparison_result, top_k=5)

    print(f"\n📁 출력 파일:")
    print(f"  {output_file}")

    # 7) VSCode 웹 UI 실행
    open_vscode_viewer(output_file)

    print("\n" + "=" * 80)
    print("✅ 모든 작업 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()
