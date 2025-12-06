# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Entrypoint for running comparative evaluations with LLM Comparator."""

from collections.abc import Sequence
import json
import os
import pathlib
import shutil
import socket
import threading
import webbrowser
from typing import Optional
from http.server import HTTPServer, SimpleHTTPRequestHandler

from LLM_comparator.llm_comparator import llm_judge_runner
from LLM_comparator.llm_comparator import rationale_bullet_generator
from LLM_comparator.llm_comparator import rationale_cluster_generator
from LLM_comparator.llm_comparator import types


# TODO(llm-comparator): Provide convenience utilities for converting from, e.g.,
# CSV/TSV to the dictionary format required by this function.
def run(
    inputs: Sequence[types.LLMJudgeInput],
    judge: llm_judge_runner.LLMJudgeRunner,
    bulletizer: rationale_bullet_generator.RationaleBulletGenerator,
    clusterer: rationale_cluster_generator.RationaleClusterGenerator,
    model_names: Sequence[str] = ('A', 'B'),
    judge_opts: Optional[types.JsonDict] = None,
    bulletizer_opts: Optional[types.JsonDict] = None,
    clusterer_opts: Optional[types.JsonDict] = None,
) -> types.JsonDict:
  """Runs a comparison with LLM Comparator.

  LLM Comparator comparisons are run in three steps:

  1. An LLM Judge is run on the inputs to produce a set of judgements.
  2. A Rationale Bullet Generator is run on the judgements to produce a set of
     rationale bullets.
  3. The Rationale Cluster Generator is run on the rationale bullets to produce
     a set of rationale clusters with similarity scores.

  Args:
    inputs: The inputs to the evaluation.
    judge: The LLM Judge to use.
    bulletizer: The Rationale Bullet Generator to use.
    clusterer: The Rationale Cluster Generator to use.
    model_names: The names of the models as you would like them to appear in the
      LLM Comparator web application.
    judge_opts: keyword arguments passed to judge.run(). See the
      llm_comparator.llm_judge_runner.LLMJudgeRunner.run() documentation for
      details.
    bulletizer_opts: keyword arguments passed to bulletizer.run(). See the
      llm_comparator.rationale_bullet_generator.RationaleBulletGenerator.run()
      documentation for details.
    clusterer_opts: keyword arguments passed to clusterer.run(). See the
      llm_comparator.rationale_cluster_generator.RationaleClusterGenerator.run()
      documentation for details.

  Returns:
    The evaluation results as a JSON object, or the value of output_path if
    provided and writing to that file was successful.
  """

  judgements = judge.run(inputs, **(judge_opts or {}))
  bullets = bulletizer.run(judgements, **(bulletizer_opts or {}))
  clusters, cluster_similarities = clusterer.run(
      bullets, **(clusterer_opts or {})
  )

  per_example_generator = zip(inputs, judgements, cluster_similarities)

  return {
      'metadata': {'custom_fields_schema': []},
      'models': [{'name': name} for name in model_names],
      'examples': [
          {
              'input_text': input['prompt'],
              'tags': [],
              'output_text_a': input['response_a'],
              'output_text_b': input['response_b'],
              'score': judgement['score'],
              'individual_rater_scores': judgement['individual_rater_scores'],
              'rationale_list': similarity,
              'custom_fields': {},
          }
          for input, judgement, similarity in per_example_generator
      ],
      'rationale_clusters': clusters,
  }


def write(comparison_result: types.JsonDict, file_path: str) -> str:
  with open(file_path, 'w') as f:
    json.dump(comparison_result, f)
  return file_path


def show_in_colab(file_path: str, height: int = 800, port: int = 8888) -> None:
  """Serves the LLM Comparator app from the Colab content directory."""
  import IPython  # pylint: disable=g-import-not-at-top #pytype: disable=import-error

  if (ishell := IPython.get_ipython()) is None:
    raise RuntimeError('Not running in an IPython context.')

  # Copy the website files from the data directory to the Colab content
  # directory if they don't already exist.
  if not os.path.isdir('/content/llm_comparator'):
    website_root = pathlib.Path(__file__).parent / 'data'
    ishell.system_raw(f'cp -R {website_root} /content/llm_comparator')

  # Serve the website from the Colab content directory.
  # TODO(llm-comparator): Check if a server is already running before trying to
  # start a new one.
  ishell.system_raw(f'python3 -m http.server {port} &')

  # Display the served website in an iframe.
  IPython.display.display(IPython.display.Javascript("""
  (async () => {
    const serverAddress = await google.colab.kernel.proxyPort(%s);
    const results_path = serverAddress + '%s';

    const fm = document.createElement('iframe');
    fm.frameBorder = 0
    fm.height = '%d'
    fm.width = '100%%'
    fm.src = serverAddress + 'llm_comparator/?results_path=' + results_path;
    document.body.append(fm)
  })();
  """ % (port, file_path, height)))


def show_in_vscode(
    file_path: str,
    web_dir: Optional[str] = None,
    port: Optional[int] = None,
    auto_open: bool = True
) -> None:
  """VSCode 환경에서 LLM Comparator를 자동으로 실행합니다.
  
  Args:
    file_path: LLM Comparator JSON 결과 파일 경로
    web_dir: LLM Comparator 웹 파일 디렉토리 (기본: 자동 감지)
    port: HTTP 서버 포트 (기본: 8000-8099 범위에서 자동 선택)
    auto_open: 브라우저 자동 열기 여부 (기본: True)
  """
  
  # 1. 웹 파일 디렉토리 찾기
  if web_dir is None:
    # 기본 경로들 시도
    possible_paths = [
        pathlib.Path(__file__).parent / 'data',
    ]
    
    for path in possible_paths:
      if os.path.isdir(path) and os.path.isfile(os.path.join(path, 'index.html')):
        web_dir = path
        break
    
    if web_dir is None:
      raise RuntimeError(
          'LLM Comparator 웹 파일을 찾을 수 없습니다. '
          'web_dir 파라미터로 경로를 지정해주세요.'
      )
  
  # 2. 사용 가능한 포트 찾기
  if port is None:
    for p in range(8000, 8100):
      try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
          s.bind(('', p))
          port = p
          break
      except OSError:
        continue
    
    if port is None:
      raise RuntimeError('사용 가능한 포트를 찾을 수 없습니다 (8000-8099).')
  
  # 3. 임시 웹 디렉토리 생성 및 파일 복사
  import tempfile
  temp_web_dir = tempfile.mkdtemp(prefix='llm_comparator_')
  
  # 웹 파일 복사
  for item in os.listdir(web_dir):
    src = os.path.join(web_dir, item)
    dst = os.path.join(temp_web_dir, item)
    if os.path.isdir(src):
      shutil.copytree(src, dst)
    else:
      shutil.copy2(src, dst)
  
  # JSON 파일 복사
  json_filename = os.path.basename(file_path)
  shutil.copy2(file_path, os.path.join(temp_web_dir, json_filename))
  
  # 4. CORS 헤더를 추가한 HTTP 서버 시작
  class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
      self.send_header('Access-Control-Allow-Origin', '*')
      self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
      self.send_header('Access-Control-Allow-Headers', '*')
      super().end_headers()
    
    def log_message(self, format, *args):
      # 로그 출력 최소화
      pass
  
  os.chdir(temp_web_dir)
  server = HTTPServer(('', port), CORSRequestHandler)
  
  # 백그라운드 스레드로 서버 시작
  def run_server():
    server.serve_forever()
  
  server_thread = threading.Thread(target=run_server, daemon=True)
  server_thread.start()
  
  # 5. URL 생성 및 출력
  comparator_url = f'http://localhost:{port}/?results_path=http://localhost:{port}/{json_filename}'
  
  print("\n" + "=" * 80)
  print("🌐 LLM Comparator 웹 UI 실행")
  print("=" * 80)
  print(f"\n📁 임시 웹 디렉토리: {temp_web_dir}")
  print(f"🌐 HTTP 서버: http://localhost:{port}")
  print(f"📊 결과 파일: {json_filename}")
  print(f"\n🔗 LLM Comparator URL:")
  print(f"   {comparator_url}")
  print("\n" + "=" * 80)
  print("VSCode에서 확인하는 방법:")
  print("=" * 80)
  print("\n1️⃣  VSCode Simple Browser (권장)")
  print("   - Ctrl+Shift+P (또는 Cmd+Shift+P)")
  print("   - 'Simple Browser: Show' 입력")
  print("   - 위 URL 붙여넣기")
  print("\n2️⃣  VSCode Ports 패널")
  print("   - VSCode 하단 'PORTS' 탭 클릭")
  print(f"   - 포트 {port} 찾기")
  print("   - 'Open in Browser' 클릭")
  print("\n3️⃣  외부 브라우저")
  print("   - 위 URL을 브라우저에 붙여넣기")
  print("\n" + "=" * 80)
  print("⚠️  주의사항:")
  print("   • 서버를 종료하려면 Ctrl+C를 누르세요")
  print("   • 서버가 실행 중일 때만 웹 UI를 사용할 수 있습니다")
  print("=" * 80 + "\n")
  
  # 6. 브라우저 자동 열기
  if auto_open:
    try:
      webbrowser.open(comparator_url)
      print("✅ 브라우저가 자동으로 열렸습니다.\n")
    except Exception as e:
      print(f"⚠️  브라우저 자동 열기 실패: {e}")
      print("   위 URL을 수동으로 복사하여 브라우저에 붙여넣으세요.\n")
  
  # 7. 서버 계속 실행
  print("🔄 서버 실행 중... (Ctrl+C로 종료)\n")
  
  try:
    # 메인 스레드를 유지하여 서버가 계속 실행되도록 함
    server_thread.join()
  except KeyboardInterrupt:
    print("\n\n✅ 서버를 종료합니다...")
    server.shutdown()
    # 임시 디렉토리 정리
    try:
      shutil.rmtree(temp_web_dir)
      print(f"✅ 임시 디렉토리 삭제: {temp_web_dir}")
    except Exception as e:
      print(f"⚠️  임시 디렉토리 삭제 실패: {e}")