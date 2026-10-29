# ==========================================
# 0. 라이브러리 및 API 키 연결
# ==========================================

import uuid
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
    page_title="재고 주문량 관리 챗봇",
    page_icon="📦",
    layout="centered"
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
# 6. 재고 주문 보고서 팝업
# ==========================================

@st.dialog(
    "📋 재고 주문 보고서",
    width="large"
)
def show_order_report():

    final_answer = st.session_state.final_answer
    order_quantity = st.session_state.order_quantity

    # --------------------------------------
    # 6-1. 최종 AI 답변
    # --------------------------------------

    st.subheader("AI 분석 결과")

    if final_answer is None:
        st.warning("생성된 AI 답변이 없습니다.")
        return

    st.write(final_answer)

    st.divider()

    # --------------------------------------
    # 6-2. 재고 주문 내역
    # --------------------------------------

    st.subheader("재고 주문 내역")

    if order_quantity is None:

        st.warning(
            "주문량 예측 결과가 없어 CSV를 생성할 수 없습니다."
        )

        return

    df = pd.DataFrame({
        "주문량": [order_quantity]
    })

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------
    # 6-3. CSV 다운로드
    # --------------------------------------

    csv_data = create_order_csv(order_quantity)

    st.download_button(
        label="💾 저장하기",
        data=csv_data,
        file_name="inventory_order.csv",
        mime="text/csv",
        use_container_width=True,
        on_click="ignore"
    )


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

        # ----------------------------------
        # 7-4. Tool 결과 추출
        # ----------------------------------

        tool_results = [
            msg
            for msg in response["messages"]
            if getattr(msg, "type", None) == "tool"
            and getattr(msg, "name", None) == "predict_order"
        ]

        if tool_results:

            try:

                order_quantity = int(
                    tool_results[-1].content
                )

                st.session_state.order_quantity = order_quantity

            except (ValueError, TypeError):
                st.session_state.order_quantity = None

    except Exception as e:

        st.session_state.messages.append({
            "role": "assistant",
            "content": f"오류가 발생했습니다: {e}"
        })

    # 입력창 초기화
    st.session_state.prompt_input = ""


# ==========================================
# 8. Streamlit 화면 구성
# ==========================================

st.title("재고 주문량 관리 챗봇")


# ------------------------------------------
# 8-1. 하단 입력창 고정 CSS
# ------------------------------------------

st.markdown(
    """
    <style>

    /* 대화 내용이 하단 입력창에 가려지지 않도록 여백 확보 */
    .block-container {
        padding-bottom: 120px;
    }

    /* 입력창 전체를 화면 하단에 고정 */
    .st-key-chat_input_bar {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: min(100%, 800px);
        padding: 12px 20px 20px 20px;
        background-color: var(--background-color, white);
        z-index: 999;
    }

    /* 전송 및 클립 버튼 높이 */
    .st-key-chat_input_bar button {
        height: 38px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------
# 8-2. 대화 내역 출력
# ------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# ------------------------------------------
# 8-3. 하단 고정 입력창
# ------------------------------------------

with st.container(key="chat_input_bar"):

    col_input, col_send, col_csv = st.columns(
        [8, 1, 1],
        vertical_alignment="bottom",
        gap="small"
    )

    # 프롬프트 입력창
    with col_input:

        st.text_input(
            label="질문 입력",
            placeholder="무엇이든 물어보세요.",
            key="prompt_input",
            label_visibility="collapsed",
            on_change=send_message
        )

    # 전송 버튼
    with col_send:

        st.button(
            label="➤",
            key="send_button",
            help="메시지 전송",
            on_click=send_message,
            use_container_width=True
        )

    # 클립 버튼
    with col_csv:

        if st.button(
            label="",
            icon=":material/attach_file:",
            key="csv_button",
            help="재고 주문 보고서 보기",
            use_container_width=True,
            disabled=(
                st.session_state.final_answer is None
            )
        ):

            show_order_report()