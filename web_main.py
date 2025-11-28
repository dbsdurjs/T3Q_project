import streamlit as st
import matplotlib.pyplot as plt
from web.tab1_keyword_analysis import run_tab1
from web.tab2_sampling import run_tab2
from web.tab3_eval_dataste import run_tab3
from streamlit.components.v1 import html

def switch_tab(tab_label: str):
    # 탭 헤더의 role="tab" 버튼들 중, 레이블이 일치하는 것을 찾아 클릭
    html(f"""
    <script>
    const tabs = parent.document.querySelectorAll('button[role="tab"]');
    for (const t of tabs) {{
        if ((t.innerText || t.textContent).trim() === "{tab_label}") {{
            t.click();
            break;
        }}
    }}
    </script>
    """, height=0, width=0)

plt.rcParams["font.family"] = "NanumGothic"
st.title("평가 데이터셋 생성 및 LLM 평가")

tab1, tab2, tab3, tab4 = st.tabs([
    "클러스터링 및 필터링", "샘플링", "평가 데이터셋 만들기", "LLM Comparator 실행"
])
with tab1:
    run_tab1(switch_tab)
with tab2:
    run_tab2(switch_tab)
with tab3:
    run_tab3(switch_tab)
# with tab4:
#     run_tab4()

