# ==========================================
# 0. 라이브러리 및 API 키 연결
# ==========================================

import uuid
import base64
import altair as alt
import json
import re
import numpy as np
import pandas as pd
import streamlit as st

from typing import Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    before_model,
)

from langchain.messages import AIMessage
from langgraph.types import Command
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver

import math
import joblib
from pathlib import Path

load_dotenv()

st.set_page_config(
    page_title="SIL | 구매 업무 어시스턴트",
    page_icon="📦",
    layout="wide"
)


# 메인 모델
model = init_chat_model("gpt-5.4-mini")

# 질문 분류용 경량 모델
filter_model = init_chat_model("gpt-5.4-nano")

# ==========================================
# 2. 툴 생성
# ==========================================

# ==========================================
# 2-1. 기능 1 - 재고 예측
# ==========================================

# ------------------------------------------
# Tool 1: 국내 / 국외 판매량 예측
# ------------------------------------------

@tool
def predict_sales(
    sku: str = "A-SRM-100",
    currency: str = "KRW",
    ad_spend: float = 892418,
    promo_days: int = 0,
    promo_discount: float = 0,
    promo_price: float = 38000,
    quarter: int = 3,
    year: int = 2024,
    month: int = 8,
    dayofweek: int = 0,
    channel_amazon_us: int = 1,
    channel_coupang: int = 0,
    channel_naver: int = 0,
    channel_own_mall: int = 0
) -> dict:
    """
    제품, 판매 채널, 광고비, 프로모션 및 날짜 정보를 기반으로
    학습된 XGBRegressor 모델을 사용하여 판매량을 예측합니다.

    Amazon US 채널은 국외 판매량으로 분류하며,
    자사몰, 쿠팡, 네이버 채널은 국내 판매량으로 분류합니다.
    """

    # 입력 데이터 생성
    input_data = pd.DataFrame([{
        "sku": sku,
        "currency": currency,
        "ad_spend": ad_spend,
        "promo_days": promo_days,
        "promo_discount": promo_discount,
        "promo_price": promo_price,
        "quarter": quarter,
        "year": year,
        "month": month,
        "dayofweek": dayofweek,
        "channel_amazon_us": channel_amazon_us,
        "channel_coupang": channel_coupang,
        "channel_naver": channel_naver,
        "channel_own_mall": channel_own_mall
    }])

    # 함수 2: 판매량 예측
    result = f"{sku}의 1주의, 2주, 3주의 예상 판매량은 10개입니다."

    return result


# ------------------------------------------
# 함수 3: 재고 데이터 불러오기
# ------------------------------------------

@st.cache_data
def load_inventory_data(
    inventory_path: str,
    product_path: str
) -> pd.DataFrame:
    """재고 현황과 제품 정보를 불러옵니다."""

    inventory = pd.read_csv(inventory_path)
    products = pd.read_csv(product_path)

    df = inventory.merge(
        products[["sku", "product_name"]],
        on="sku",
        how="left",
        suffixes=("", "_master"),
        validate="many_to_one"
    )

    return df


# ------------------------------------------
# 함수 4: 재고량 파악
# ------------------------------------------

def find_inventory_data(
    sku: str,
    df: pd.DataFrame,
) -> int:
    """제품명을 바탕으로 재고 데이터에서 특정 제품의 재고량을 파악합니다."""
    
    item_inventory = df[(df["sku"] == sku) & (df["location"] == "KR_WH")]

    current_stock = item_inventory["on_hand"].iloc[0]
    
    return current_stock

# ------------------------------------------
# 함수 5: 추가 주문량 계산
# ------------------------------------------

def calculate_order_quantity(
    current_stock: int,
    sales_week1: int,
    sales_week2: int,
    sales_week3: int
) -> dict:
    """
    현재 재고와 1~3주 예상 판매량을 비교해
    500개 단위로 추가 주문량을 계산합니다.
    """

    # 1~3주 예상 판매량
    predicted_sales = (
        sales_week1
        + sales_week2
        + sales_week3
    )

    # 예상 재고 부족량
    required_qty = max(
        0,
        predicted_sales - current_stock
    )

    # 500개 단위 주문
    order_qty = (
        math.ceil(required_qty / 500) * 500
    )

    return {
        "predicted_sales": predicted_sales,
        "required_qty": math.ceil(required_qty),
        "order_qty": order_qty
    }


# ------------------------------------------
# Tool 2: 창고별 추가 주문량 계산
# ------------------------------------------

@tool
def check_inventory(
    sku: str,
    sales_week1: int,
    sales_week2: int,
    sales_week3: int
) -> dict:
    """
    SKU별 재고를 조회하고
    1~3주 예상 판매량을 기반으로 추가 주문량을 계산합니다.
    """

    # 재고 데이터 불러오기
    df = load_inventory_data("inventory_now.csv", "product_master.csv")

    # 특정 제품의 재고량 확인
    current_stock = find_inventory_data(sku, df)

    # 추가 주문량 계산
    result = calculate_order_quantity(current_stock, sales_week1, sales_week2, sales_week3)

    return result


# ------------------------------------------
# 2-2. 기능 2 - CSV 생성
# ------------------------------------------

def create_order_csv(order_quantity: int) -> bytes:
    """
    예측된 주문량을 CSV 파일로 변환합니다.
    """

    df = pd.DataFrame({
        "주문량": [order_quantity]
    })

    return df.to_csv(
        index=False
    ).encode("utf-8-sig")


# ==========================================
# 3. 미들웨어 생성
# ==========================================

# ------------------------------------------
# 3-1. 질문 분류 결과 구조 정의
# ------------------------------------------

class QuestionFilter(BaseModel):

    is_relevant: bool = Field(
        description="재고 및 주문 관리 관련 질문이면 True, 아니면 False"
    )


filter_llm = filter_model.with_structured_output(
    QuestionFilter
)


# ------------------------------------------
# 3-2. 질문 분류 미들웨어
# ------------------------------------------

@before_model
def check_order_question(
    state: AgentState,
    runtime: Runtime
) -> dict[str, Any] | Command:

    """
    경량 LLM으로 재고 관련 질문을 판단합니다.
    관련 없는 질문은 메인 모델에 전달하지 않습니다.
    """

    # 가장 최근 사용자 질문 추출
    user_input = next(
        (
            msg.content
            for msg in reversed(state["messages"])
            if msg.type == "human"
        ),
        ""
    )

    # 질문 관련성 판단
    result = filter_llm.invoke([
        (
            "system",
            """
            당신은 재고 관리 AI 서비스의 질문 분류기입니다.

            사용자 질문이 다음 업무와 관련되는지 판단하세요.

            [관련 있는 질문]
            - 재고 관리 및 재고 부족 예측
            - 제품 발주 및 주문량 결정
            - 판매량 및 수요 예측
            - 광고비 변화에 따른 판매량 분석
            - 제품별, 채널별 판매 실적
            - 입고 일정 및 재고 운영
            - 위 업무와 관련된 후속 질문

            [관련 없는 질문]
            - 날씨, 뉴스 등 일반적인 정보
            - 재고 관리와 무관한 일상 대화
            - 일반적인 프로그래밍 질문
            - 기타 재고 및 판매 운영과 관련 없는 질문

            질문의 의미와 의도를 기준으로 판단하세요.
            단순히 특정 키워드가 포함되어 있다는 이유만으로
            관련 질문이라고 판단하지 마세요.
            """
        ),
        ("human", str(user_input))
    ])

    # 관련 없는 질문이면 END
    if not result.is_relevant:

        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=(
                            "재고 관리 및 주문량 예측과 "
                            "관련된 질문만 답변할 수 있습니다."
                        )
                    )
                ]
            },
            goto="__end__"
        )

    # 관련 질문이면 메인 모델 실행
    return {}


# ==========================================
# 4. Agent 생성
# ==========================================

# 세션별 Agent 및 체크포인터 생성
def get_agent():

    if "agent" not in st.session_state:

        checkpointer = InMemorySaver()

        st.session_state.agent = create_agent(
            model=model,
            tools=[predict_sales, check_inventory],
            middleware=[check_order_question],
            checkpointer=checkpointer,
            system_prompt = """
            당신은 제품별 판매량과 재고 주문량을 예측하는 AI 에이전트입니다.

            사용자가 판매량을 물어보면 predict_sales Tool을 호출하여 판매량에 대한 답변을 제공합니다.
            사용자가 재고 주문량에 대해 물어보면 predict_sales Tool을 호출한 후check_inventory Tool을 호출하여 재고 주문량 대한 답변을 제공합니다.

            사용자가 제품 관련 정보를 입력하지 않는다면 predict_sales Tool을 호출할 때 디폴트 값을 활용합니다.

            """
        )

    return st.session_state.agent


agent = get_agent()


# ==========================================
# 5. Streamlit 세션 초기화
# ==========================================

# 대화 기록
if "messages" not in st.session_state:
    st.session_state.messages = []


# 최종 AI 답변
if "final_answer" not in st.session_state:
    st.session_state.final_answer = None


# 주문량
if "order_quantity" not in st.session_state:
    st.session_state.order_quantity = None


# 사용자별 대화 ID
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# ==========================================
# 6. 출력 전용 데이터 변환 및 보고서
# ==========================================

CHANNELS = {
    "channel_amazon_us": "Amazon US",
    "channel_coupang": "쿠팡",
    "channel_naver": "네이버",
    "channel_own_mall": "자사몰",
}


def extract_display_results(messages):
    """이번 응답의 도구 반환값과 호출 인자만 화면용 행으로 변환합니다."""
    # 체크포인터가 반환하는 과거 대화의 결과를 이번 결과에 섞지 않습니다.
    start = next((i for i in range(len(messages) - 1, -1, -1)
                  if getattr(messages[i], "type", None) == "human"), -1)
    calls, sales, orders = {}, [], []
    for message in messages[start + 1:]:
        for call in getattr(message, "tool_calls", []) or []:
            calls[call["id"]] = call
        if getattr(message, "type", None) != "tool":
            continue
        if getattr(message, "status", None) == "error":
            continue
        call = calls.get(getattr(message, "tool_call_id", None), {})
        args = call.get("args", {})
        name = getattr(message, "name", None)
        content = message.content
        if name == "predict_sales" and isinstance(content, str):
            # 현재 도구의 문장 형식을 명시적으로 지원합니다. 임의의 숫자는 추정하지 않습니다.
            match = re.fullmatch(r"(.+?)의 1주의, 2주, 3주의 예상 판매량은 (\d+)개입니다\.", content.strip())
            if match:
                active = [label for flag, label in CHANNELS.items()
                          if args.get(flag, 1 if flag == "channel_amazon_us" else 0) == 1]
                channel = active[0] if len(active) == 1 else "채널 구분 불가"
                for week in range(1, 4):
                    sales.append({"SKU": match[1], "판매 사이트": channel,
                                  "기간": f"{week}주차", "예상 판매량(개)": int(match[2])})
        elif name == "check_inventory":
            try:
                result = json.loads(content) if isinstance(content, str) else content
                values = [result[key] for key in ("predicted_sales", "required_qty", "order_qty")]
                if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                           and math.isfinite(v) and v >= 0 for v in values):
                    continue
                # 기존 check_inventory는 KR_WH를 조회하므로 국내 결과로만 표시합니다.
                orders.append({"SKU": args.get("sku", "미확인"), "지역": "국내",
                               "창고": "KR_WH", "예상 판매량(개)": values[0],
                               "부족 수량(개)": values[1], "주문량(개)": values[2]})
            except (ValueError, TypeError, KeyError):
                continue
    return {"sales": sales, "orders": orders}


def render_results(results, key):
    sales, orders = results.get("sales", []), results.get("orders", [])
    if sales:
        st.markdown("#### 판매량 예측")
        site = st.segmented_control("판매 사이트", ["전체", *CHANNELS.values()],
                                    default="전체", key=f"site_{key}") or "전체"
        frame = pd.DataFrame(sales)
        if site != "전체":
            frame = frame[frame["판매 사이트"] == site]
        if frame.empty:
            st.info("선택한 판매 사이트의 조회 결과가 없습니다.")
        else:
            st.metric("조회된 예상 판매량", f"{frame['예상 판매량(개)'].sum():,.0f}개")
            plot = frame.copy()
            plot["제품 · 사이트"] = plot["SKU"] + " · " + plot["판매 사이트"]
            chart = alt.Chart(plot).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("기간:N", title=None, sort=["1주차", "2주차", "3주차"]),
                y=alt.Y("예상 판매량(개):Q", title="판매량(개)"),
                xOffset="제품 · 사이트:N",
                color=alt.Color("제품 · 사이트:N", scale=alt.Scale(range=["#FFBCA6", "#DFA18C", "#BE8170", "#8D655B"]),
                                legend=alt.Legend(title=None, orient="bottom", labelFontSize=16)),
                tooltip=["SKU", "판매 사이트", "기간", "예상 판매량(개)"]
            ).properties(height=220).configure_view(strokeWidth=0).configure_axis(
                gridColor="#F4E6E0", domain=False, labelColor="#796B65", titleColor="#796B65",
                labelFontSize=16, titleFontSize=16)
            st.altair_chart(chart, width="stretch")
            with st.expander("판매량 상세 표", expanded=False):
                st.dataframe(frame, hide_index=True, width="stretch")
        st.caption("현재 판매량 도구는 1~3주 고정 예시값을 반환합니다. 실제 예측 모델 결과가 아닙니다.")
    if orders:
        if sales:
            st.divider()
        st.markdown("#### 재고 주문 제안")
        region = st.segmented_control("주문 지역", ["전체", "국내", "국외"],
                                      default="전체", key=f"region_{key}") or "전체"
        frame = pd.DataFrame(orders)
        if region != "전체":
            frame = frame[frame["지역"] == region]
        if frame.empty:
            st.info("국외 주문량 결과가 없습니다. 현재 도구는 국내 창고(KR_WH)만 조회합니다.")
        else:
            c1, c2 = st.columns(2)
            c1.metric("부족 수량", f"{frame['부족 수량(개)'].sum():,.0f}개")
            c2.metric("추천 주문량", f"{frame['주문량(개)'].sum():,.0f}개")
            st.dataframe(frame, hide_index=True, width="stretch")
            st.download_button("CSV 저장", icon=":material/download:",
                               data=frame.to_csv(index=False).encode("utf-8-sig"),
                               file_name="sil_inventory_order.csv", mime="text/csv",
                               key=f"download_{key}", on_click="ignore")
            st.caption("표시된 주문 내역을 저장합니다. 실제 발주는 실행되지 않습니다.")


@st.dialog("재고 주문 보고서", width="large")
def show_order_report():
    results = st.session_state.get("display_results", {})
    if not results.get("orders"):
        st.info("재고 주문량을 질문하면 주문 내역과 CSV 저장 버튼이 표시됩니다.")
        return
    render_results({"orders": results["orders"]}, "report")


# ==========================================
# 7. 프롬프트 전송 함수
# ==========================================

def send_message():

    prompt = st.session_state.prompt_input.strip()

    if not prompt:
        return

    # --------------------------------------
    # 7-1. 사용자 메시지 저장
    # --------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # 이전 예측 결과 초기화
    st.session_state.final_answer = None
    st.session_state.order_quantity = None
    st.session_state.display_results = {}

    # --------------------------------------
    # 7-2. Agent 실행
    # --------------------------------------

    try:

        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            {
                "configurable": {
                    "thread_id": st.session_state.thread_id
                }
            }
        )

        # ----------------------------------
        # 7-3. 최종 AI 답변 저장
        # ----------------------------------

        answer = response["messages"][-1].content

        st.session_state.final_answer = answer

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        # 도구 실행은 그대로 유지하고, 반환 메시지만 화면용으로 변환합니다.
        display_results = extract_display_results(response["messages"])
        st.session_state.display_results = display_results
        st.session_state.messages[-1]["results"] = display_results
        if display_results["orders"]:
            st.session_state.order_quantity = display_results["orders"][-1]["주문량(개)"]

    except Exception as e:

        st.session_state.messages.append({
            "role": "assistant",
            "content": f"오류가 발생했습니다: {e}"
        })

    # 입력창 초기화
    st.session_state.prompt_input = ""


# ==========================================
# 8. SIL 구매 워크스페이스 — 출력 및 탐색 UI
# ==========================================

def start_new_chat():
    # 기존 대화는 화면 보관함에 남기고 새 thread_id로 기존 에이전트를 이용합니다.
    if st.session_state.messages:
        archive = st.session_state.setdefault("chat_archive", [])
        archive.append({"messages": list(st.session_state.messages),
                        "thread_id": st.session_state.thread_id,
                        "results": st.session_state.get("display_results", {})})
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.final_answer = None
    st.session_state.order_quantity = None
    st.session_state.display_results = {}
    st.session_state.pop("analysis_choice", None)


def restore_chat(index):
    archive = st.session_state.chat_archive
    selected = archive.pop(index)
    if st.session_state.messages:
        archive.append({"messages": list(st.session_state.messages),
                        "thread_id": st.session_state.thread_id,
                        "results": st.session_state.get("display_results", {})})
    st.session_state.messages = selected["messages"]
    st.session_state.thread_id = selected["thread_id"]
    st.session_state.display_results = selected["results"]
    st.session_state.final_answer = next((m["content"] for m in reversed(selected["messages"])
                                         if m["role"] == "assistant"), None)
    rows = selected["results"].get("orders", [])
    st.session_state.order_quantity = rows[-1]["주문량(개)"] if rows else None
    st.session_state.pop("analysis_choice", None)


@st.cache_data
def image_data_uri(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


st.markdown("""
<style>
:root { --sil-ink:#392822; --sil-peach:#FFC5B3;
    --sil-title:32px; --sil-subtitle:22px; --sil-body:16px; }
.stApp { background:#FFF8F5; color:var(--sil-ink); }
[data-testid="stHeader"] { background:rgba(255,255,255,.95); }
.block-container { max-width:1600px; padding:4.5rem 2rem 2rem; }
[data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid #F3E4DD; }
[data-testid="stSidebar"] hr { border-color:#F5E9E3; margin:1.3rem 0; }
h1,h2,h3,h4 { color:#392822; letter-spacing:-.04em; }
.stButton > button, .stDownloadButton > button {
    border:1px solid #F2D5C8; border-radius:12px; color:#392822; background:#FFFFFF; }
.stButton > button:hover, .stDownloadButton > button:hover { background:#FFF1EB; border-color:#EBAE97; color:#392822; }
.stButton > button[kind="primary"] { background:#FFC5B3; border-color:#FFC5B3; color:#392822; }
[data-testid="stBaseButton-segmented_controlActive"] { background:#FFC5B3; color:#392822; border-color:#FFC5B3; }
[data-testid="stBaseButton-segmented_control"] { background:#FFFAF8; border-color:#F4DCD1; color:#594039; }
[data-testid="stMetricValue"] { color:#392822; font-size:var(--sil-title); font-weight:650; }
[data-testid="stMetricLabel"] { color:#8A7B75; }
[data-testid="stCaptionContainer"] { color:#897B76; }
.st-key-chat_panel, .st-key-analysis_panel {
    background:#FFFFFF; border:1px solid #F5E6DF; border-radius:18px; padding:22px;
    box-shadow:0 6px 24px rgba(90,50,30,.025); }
[data-testid="stChatMessage"] { background:transparent; border:none; padding:10px 0; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background:#FFF2ED; border-radius:16px; padding:14px; }
[data-testid="stChatMessageAvatarUser"] { background:#FFE5DA; color:#604438; }
[data-testid="stChatMessage"] img { object-fit:contain; border-radius:0; }
[data-testid="stChatInput"] { border:1px solid #F1CFC1; background:#FFFFFF; border-radius:14px; }
[data-testid="stChatInput"] textarea { color:#392822; }
[data-testid="stChatInputSubmitButton"] { background:#FFC5B3; color:#392822; border-radius:10px; }
.sil-topline { display:flex; justify-content:space-between; align-items:center; gap:12px;
    font-size:var(--sil-body); color:#7E706A; margin-bottom:18px; }
.sil-badge { background:#FFEFE8; padding:6px 12px; border-radius:30px; font-size:var(--sil-body); }
.sil-hero { position:relative; overflow:hidden; min-height:210px; border-radius:18px;
    background:white; margin-bottom:22px; border:1px solid #F6E7E0; }
.sil-hero-photo { position:absolute; right:0; top:0; width:58%; height:100%;
    background-size:cover; background-position:center 30%; }
.sil-hero-photo::after { content:''; position:absolute; inset:0;
    background:linear-gradient(90deg,#FFFFFF 0%,rgba(255,255,255,.4) 25%,transparent 55%); }
.sil-hero-copy { position:relative; z-index:1; padding:32px; width:59%; }
.sil-hero h1 { font-size:var(--sil-title); line-height:1.28; font-weight:750; margin:0 0 14px; padding:0; }
.sil-hero p { color:#85716A; font-size:var(--sil-body); margin:0; }
.sil-eyebrow { color:#8E776C; font-size:var(--sil-body); letter-spacing:.12em; margin:0 0 16px; }
.sil-empty { padding:50px 15px; text-align:center; color:#97837A; }
.sil-empty strong { display:block; color:#60493F; font-size:var(--sil-subtitle); margin-bottom:14px; }
/* 제목 / 소제목 / 본문 세 단계의 공통 타이포그래피 */
.stApp h1, .stApp [data-testid="stMetricValue"] { font-size:var(--sil-title) !important; }
.stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.sil-empty strong { font-size:var(--sil-subtitle) !important; line-height:1.4; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp button, .stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p,
.stApp [data-testid="stMetricLabel"] p,
.stApp [data-baseweb="select"] { font-size:var(--sil-body) !important; line-height:1.6; }
.sil-brand-logo { width:100%; max-width:250px; height:112px; overflow:hidden; }
.sil-brand-logo img { width:100%; height:100%; object-fit:contain;
    transform:scale(1.5); transform-origin:52% 52%; }
/* 양쪽 패널은 동일한 외곽 높이, 긴 결과는 패널 안에서 스크롤 */
.st-key-chat_panel, .st-key-analysis_panel { box-sizing:border-box; }
@media(max-width:900px) {
    .block-container { padding:4.5rem 1rem 2rem; }
    .sil-hero-copy { width:75%; padding:24px; }
    .sil-hero-photo { opacity:.45; width:70%; }
    .sil-hero { min-height:180px; }
    [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-chat_panel) { flex-direction:column; }
    [data-testid="stColumn"]:has(.st-key-chat_panel), [data-testid="stColumn"]:has(.st-key-analysis_panel) { width:100%; flex:1 1 100%; }
    .st-key-chat_panel, .st-key-analysis_panel { padding:16px; }
}
</style>
""", unsafe_allow_html=True)

assets = Path(__file__).parent / "assets"
logo_path = assets / "sil-logo.png"
assistant_logo_path = assets / "taekwang-wordmark.png"
assistant_avatar = str(assistant_logo_path) if assistant_logo_path.exists() else None

with st.sidebar:
    if logo_path.exists():
        st.markdown(
            f'<div class="sil-brand-logo"><img src="{image_data_uri(str(logo_path))}" alt="SIL 로고"></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="sil-eyebrow">PURCHASING AI</div>', unsafe_allow_html=True)
    st.button("새 대화", icon=":material/add:", type="primary", width="stretch", on_click=start_new_chat)
    # 단일 페이지 앱의 현재 메뉴 표시: 페이지 탐색 API를 사용하지 않습니다.
    st.markdown(
        '<div aria-current="page" style="background:#FFF1EB; color:#392822; '
        'border-radius:12px; padding:14px 16px; font-weight:600; text-align:center; margin-bottom:12px;">구매 어시스턴트</div>',
        unsafe_allow_html=True,
    )
    if st.button("주문 보고서", icon=":material/description:", width="stretch",
                 disabled=not st.session_state.get("display_results", {}).get("orders")):
        show_order_report()
    st.divider()
    st.caption("분석 내역 · 현재 대화")
    analysis_indices = [i for i,m in enumerate(st.session_state.messages)
                        if m.get("results", {}).get("sales") or m.get("results", {}).get("orders")]
    if analysis_indices:
        def analysis_label(i):
            return next((m["content"][:40] for m in reversed(st.session_state.messages[:i])
                         if m["role"] == "user"), "분석 결과")
        choices = ["latest", *reversed(analysis_indices)]
        if st.session_state.get("analysis_choice") not in choices:
            st.session_state.pop("analysis_choice", None)
        selected = st.selectbox("표시할 분석", choices, key="analysis_choice",
                                format_func=lambda i: "가장 최근 질문" if i == "latest" else analysis_label(i),
                                label_visibility="collapsed")
    else:
        selected = "latest"
        st.caption("질문 후 분석 내역이 표시됩니다.")
    st.divider()
    st.caption("이전 대화 · 현재 세션")
    for index, chat in reversed(list(enumerate(st.session_state.get("chat_archive", [])))):
        title = next((m["content"] for m in chat["messages"] if m["role"] == "user"), "이전 대화")
        st.button(title[:32], key=f"restore_{chat['thread_id']}", icon=":material/chat:",
                  width="stretch", on_click=restore_chat, args=(index,))
    st.caption("세션이 종료되면 대화 보관함도 초기화됩니다.")
    st.divider()
    st.markdown(":material/account_circle:　구매 담당자")

st.markdown('<div class="sil-topline"><span>워크스페이스 / <b>구매 어시스턴트</b></span><span class="sil-badge">SIL · 구매 의사결정 지원</span></div>', unsafe_allow_html=True)
banner_path = assets / "safin-banner.png"
banner_style = f"background-image:url('{image_data_uri(str(banner_path))}');" if banner_path.exists() else ""
st.markdown(f"""<section class="sil-hero">
<div class="sil-hero-photo" style="{banner_style}"></div>
<div class="sil-hero-copy"><h1>AI 구매 어시스턴트</h1>
<p>판매 예측부터 발주 계획까지, AI와 함께하세요.</p></div></section>""", unsafe_allow_html=True)

chat_col, analysis_col = st.columns([0.95, 1.25], gap="medium")
PANEL_HEIGHT = 740
with chat_col, st.container(key="chat_panel", height=PANEL_HEIGHT, border=False):
    title_icon, title_text = st.columns([1, 6], vertical_alignment="center")
    with title_icon:
        if assistant_avatar:
            st.image(assistant_avatar, width=55)
    title_text.markdown("### AI 어시스턴트")
    with st.container(height=430, border=False):
        if not st.session_state.messages:
            st.markdown('<div class="sil-empty"><strong>어떤 구매 계획을 세울까요?</strong>제품 SKU와 궁금한 판매량 또는 주문량을 입력해 주세요.</div>', unsafe_allow_html=True)
            st.caption("질문 예시 · A-SRM-100의 재고를 확인하고 추가 주문량을 계산해줘.")
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=assistant_avatar if message["role"] == "assistant" else None):
                st.write(message["content"])
    st.chat_input("제품과 궁금한 내용을 입력하세요", key="prompt_input", on_submit=send_message)
    st.caption("판매량과 주문 결과는 오른쪽 분석 영역에서 확인하세요.")

with analysis_col, st.container(key="analysis_panel", height=PANEL_HEIGHT, border=False):
    st.markdown("### 판매 및 재고 분석")
    results = (st.session_state.get("display_results", {}) if selected == "latest"
               else st.session_state.messages[selected]["results"])
    if results.get("sales") or results.get("orders"):
        st.caption("최근 질문의 결과" if selected == "latest" else f"선택한 분석 · {analysis_label(selected)}")
        render_results(results, f"workspace_{st.session_state.thread_id}_{selected}")
    else:
        st.markdown('<div class="sil-empty"><strong>분석 결과를 기다리고 있어요</strong>판매량이나 재고 주문량을 질문하면<br>차트와 주문 제안이 이곳에 표시됩니다.</div>', unsafe_allow_html=True)
        st.divider()
        st.markdown("**판매량 예측**　주차별 수량과 판매 사이트 비교")
        st.markdown("**재고 주문 제안**　국내·국외 구분과 CSV 저장")
        st.caption("현재 판매량은 고정 예시값이며 주문 계산은 국내 창고 기준입니다.")
