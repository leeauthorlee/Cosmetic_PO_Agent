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
# 함수 1: 학습된 XGBRegressor 모델 불러오기
# ------------------------------------------

@st.cache_resource
def sales_model(pile_path: str):
    """입력받은 파일 경로에서 학습된 판매량 예측 모델을 불러오는 함수"""

    model = joblib.load(pile_path)

    return model


# ------------------------------------------
# 함수 2: 국내 / 국외 판매량 예측
# ------------------------------------------

def sales_pre(input_data: pd.DataFrame) -> dict:
    """
    입력 데이터를 기반으로 판매량을 예측하고,
    국내와 국외 판매량을 구분합니다.

    Returns:
        dict: 국내 및 국외 예측 판매량
    """

    # 함수 1: 모델 로드
    model = sales_model("sales_model.json")

    # 판매량 예측# ------------------------------------------
# 함수 2: 국내 / 국외 판매량 예측
# ------------------------------------------

def sales_pre(input_data: pd.DataFrame) -> dict:
    """
    입력 데이터를 기반으로 판매량을 예측하고,
    국내와 국외 판매량을 구분합니다.

    Returns:
        dict: 국내 및 국외 예측 판매량
    """

    # 함수 1: 모델 로드
    model = joblib.load('xgbr_model.joblib')
    mms = joblib.load('minmax_scaler.joblib')

    # 판매량 예측
    input_data = mms.transform(input_data)
    predictions = model.predict(input_data)

    # 음수 예측값 0으로 보정
    predictions = np.maximum(predictions, 0)

    # 채널 구분
    # Amazon US = 국외
    # Coupang, Naver, Own Mall = 국내
    is_overseas = input_data["channel_amazon_us"].eq(1).to_numpy()

    # 국내 판매량
    domestic_sales = float(
        predictions[~is_overseas].sum()
    )

    # 국외 판매량
    overseas_sales = float(
        predictions[is_overseas].sum()
    )

    return {
        "domestic_sales": domestic_sales,
        "overseas_sales": overseas_sales
    }


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

    [입력 파라미터]

    sku (str):
        판매량을 예측할 제품의 고유 코드입니다.
        product_master.csv의 sku 컬럼을 기준으로 합니다.
        기본값: "A-SRM-100"

    currency (str):
        광고비(ad_spend)와 프로모션 가격(promo_price)의 통화입니다.
        국내 채널은 KRW(원), Amazon US는 USD(달러)를 사용합니다.
        기본값: "KRW"

    ad_spend (float):
        해당 판매 채널의 해당 주 광고비입니다.
        currency에 따라 원화 또는 달러 단위로 입력합니다.
        같은 주, 같은 채널에서는 모든 제품에 동일한
        광고비가 적용됩니다.
        기본값: 892418

    promo_days (int):
        해당 주에 제품과 판매 채널에서
        프로모션이 진행되는 일수입니다.
        0~7 사이의 정수를 입력합니다.
        0은 프로모션이 없는 경우입니다.
        기본값: 0

    promo_discount (float):
        해당 주에 적용되는 프로모션의 최대 할인율입니다.
        프로모션이 없으면 0을 입력합니다.
        할인율의 입력 단위(예: 0.1 또는 10)는
        학습 데이터의 전처리 기준과 일치해야 합니다.
        기본값: 0

    promo_price (float):
        해당 제품의 프로모션 판매 가격입니다.
        프로모션 가격은 제공된 원본 CSV 컬럼에
        직접 포함되어 있지 않습니다.
        모델 학습 시 사용한 가격 산정 및 전처리 기준에
        맞춰 입력해야 합니다.
        기본값: 38000

    quarter (int):
        판매량을 예측할 날짜가 속한 분기입니다.
        1~4 사이의 정수를 입력합니다.
        기본값: 3

    year (int):
        판매량을 예측할 날짜의 연도입니다.
        기본값: 2024

    month (int):
        판매량을 예측할 날짜의 월입니다.
        1~12 사이의 정수를 입력합니다.
        기본값: 8

    dayofweek (int):
        판매량을 예측할 날짜의 요일입니다.
        월요일은 0, 화요일은 1, ... 일요일은 6입니다.
        주간 판매량 예측에서 week_start(월요일)를
        기준으로 입력한다면 0을 사용합니다.
        기본값: 0

    channel_amazon_us (int):
        Amazon US 판매 채널 여부입니다.
        Amazon US이면 1, 아니면 0입니다.
        해당 채널은 국외 판매량으로 분류됩니다.
        기본값: 1

    channel_coupang (int):
        쿠팡 판매 채널 여부입니다.
        쿠팡이면 1, 아니면 0입니다.
        해당 채널은 국내 판매량으로 분류됩니다.
        기본값: 0

    channel_naver (int):
        네이버 판매 채널 여부입니다.
        네이버이면 1, 아니면 0입니다.
        해당 채널은 국내 판매량으로 분류됩니다.
        기본값: 0

    channel_own_mall (int):
        자사몰 판매 채널 여부입니다.
        자사몰이면 1, 아니면 0입니다.
        해당 채널은 국내 판매량으로 분류됩니다.
        기본값: 0

    [채널 입력 규칙]

    4개의 채널 변수 중 하나만 1로 설정해야 합니다.

    예시:
        Amazon US: (1, 0, 0, 0)
        쿠팡:      (0, 1, 0, 0)
        네이버:    (0, 0, 1, 0)
        자사몰:    (0, 0, 0, 1)

    [반환값]

    domestic_sales (float):
        국내 채널의 예측 판매량입니다.
        자사몰, 쿠팡, 네이버 채널의 예측값을 합산합니다.
        단, 실제 입력된 채널의 예측값만 포함합니다.

    overseas_sales (float):
        Amazon US 채널의 예측 판매량입니다.

    [주의사항]

    - 판매량 예측 모델의 예측 대상은 units입니다.
    - units는 해당 주의 제품별·채널별 판매 수량입니다.
    - 입력하지 않은 채널의 판매량은 예측되지 않습니다.
    - 광고비와 프로모션 정보는 예측 대상 주간의 값을 사용합니다.
    - 입력값은 모델 학습 당시의 컬럼 및 전처리 기준과
      동일해야 합니다.
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
    result = sales_pre(input_data)

    return result


# # ------------------------------------------
# # 함수 3: 재고 데이터 불러오기
# # ------------------------------------------

# @st.cache_data
# def load_inventory_data(
#     inventory_path: str,
#     product_path: str
# ) -> pd.DataFrame:
#     """재고 현황과 제품 정보를 불러옵니다."""

#     inventory = pd.read_csv(inventory_path)
#     products = pd.read_csv(product_path)

#     df = inventory.merge(
#         products[["sku", "product_name"]],
#         on="sku",
#         how="left",
#         suffixes=("", "_master"),
#         validate="many_to_one"
#     )

#     return df


# # ------------------------------------------
# # 함수 4: 추가 주문량 계산
# # ------------------------------------------

# def calculate_order_quantity(
#     current_stock: int,
#     sales_week1: float,
#     sales_week2: float,
#     sales_week3: float
# ) -> dict:
#     """
#     현재 재고와 1~3주 예상 판매량을 비교해
#     500개 단위로 추가 주문량을 계산합니다.
#     """

#     # 1~3주 예상 판매량
#     predicted_sales = (
#         sales_week1
#         + sales_week2
#         + sales_week3
#     )

#     # 예상 재고 부족량
#     required_qty = max(
#         0,
#         predicted_sales - current_stock
#     )

#     # 500개 단위 주문
#     order_qty = (
#         math.ceil(required_qty / 500) * 500
#     )

#     return {
#         "predicted_sales": predicted_sales,
#         "required_qty": math.ceil(required_qty),
#         "order_qty": order_qty
#     }


# # ------------------------------------------
# # Tool 2: 창고별 추가 주문량 계산
# # ------------------------------------------

# @tool
# def check_inventory(
#     sku: str,
#     domestic_sales: list[float],
#     overseas_sales: list[float]
# ) -> dict:
#     """
#     SKU별 국내 및 미국 창고 재고를 조회하고
#     1~3주 예상 판매량을 기반으로 추가 주문량을 계산합니다.

#     Args:
#         sku: 제품 SKU
#         domestic_sales: 국내 1~3주 예상 판매량
#         overseas_sales: 미국 1~3주 예상 판매량

#     Returns:
#         창고별 현재 재고, 예상 판매량,
#         재고 부족량, 추가 주문량
#     """

#     # 입력값 검증
#     if len(domestic_sales) != 3:
#         raise ValueError(
#             "domestic_sales에는 1~3주 예상 판매량 3개가 필요합니다."
#         )

#     if len(overseas_sales) != 3:
#         raise ValueError(
#             "overseas_sales에는 1~3주 예상 판매량 3개가 필요합니다."
#         )

#     # 데이터 불러오기
#     base_dir = Path(__file__).resolve().parent

#     df = load_inventory_data(
#         str(base_dir / "inventory_now.csv"),
#         str(base_dir / "product_master.csv")
#     )

#     # SKU 조회
#     target = df[df["sku"] == sku]

#     if target.empty:
#         raise ValueError(
#             f"재고 정보를 찾을 수 없습니다: {sku}"
#         )

#     results = {}

#     # 창고별 판매량 매핑
#     sales_by_location = {
#         "KR_WH": domestic_sales,
#         "AMZ_FBA": overseas_sales
#     }

#     # 창고별 계산
#     for location, weekly_sales in sales_by_location.items():

#         warehouse = target[
#             target["location"] == location
#         ]

#         if warehouse.empty:
#             raise ValueError(
#                 f"{sku}의 {location} 재고가 없습니다."
#             )

#         row = warehouse.iloc[0]

#         current_stock = int(row["on_hand"])

#         # 추가 주문량 계산
#         order_result = calculate_order_quantity(
#             current_stock=current_stock,
#             sales_week1=weekly_sales[0],
#             sales_week2=weekly_sales[1],
#             sales_week3=weekly_sales[2]
#         )

#         results[location] = {
#             "current_stock": current_stock,
#             "open_po_qty": int(row["open_po_qty"]),
#             "weekly_sales": weekly_sales,
#             "predicted_sales": order_result["predicted_sales"],
#             "required_qty": order_result["required_qty"],
#             "order_qty": order_result["order_qty"]
#         }

#     return {
#         "sku": sku,
#         "as_of": str(target.iloc[0]["as_of"]),
#         "inventory": results
#     }



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
            tools=[predict_sales],
            middleware=[check_order_question],
            checkpointer=checkpointer,
            system_prompt = """
            당신은 제품별 판매량을 예측하는 AI 에이전트입니다.

            현재는 판매량 예측 기능을 검증하는 테스트 단계입니다.
            사용자의 자연어 질문을 분석하여 predict_sales Tool을
            자동으로 호출하고, 예측 결과를 안내하는 것이 목표입니다.


            [1. 역할]

            1. 사용자의 질문에서 판매량 예측에 필요한 정보를 추출합니다.
            2. 추출한 정보를 predict_sales의 입력 파라미터에 맞게 변환합니다.
            3. 사용자가 입력하지 않은 정보는 Tool에 정의된 기본값을 사용합니다.
            4. predict_sales Tool을 호출하여 판매량을 예측합니다.
            5. Tool의 실제 반환값을 바탕으로 사용자에게 답변합니다.


            [2. 입력 파라미터 설정]

            predict_sales는 다음 입력값을 사용합니다.

            - sku: 제품 코드
            - currency: 통화
            - ad_spend: 광고비
            - promo_days: 프로모션 일수
            - promo_discount: 최대 할인율
            - promo_price: 프로모션 가격
            - quarter: 분기
            - year: 연도
            - month: 월
            - dayofweek: 요일
            - channel_amazon_us: Amazon US 여부
            - channel_coupang: 쿠팡 여부
            - channel_naver: 네이버 여부
            - channel_own_mall: 자사몰 여부

            입력값 설정 우선순위:

            1순위: 사용자가 명시한 값
            2순위: 이전 대화에서 확인된 값
            3순위: predict_sales에 정의된 기본값

            사용자가 명시하지 않은 값은 임의로 생성하지 말고
            Tool에 설정된 기본값을 활용합니다.


            [3. 기본값]

            사용자가 입력값을 지정하지 않았다면
            다음 기본값을 사용합니다.

            sku = "A-SRM-100"
            currency = "KRW"
            ad_spend = 892418
            promo_days = 0
            promo_discount = 0
            promo_price = 38000
            quarter = 3
            year = 2024
            month = 8
            dayofweek = 0
            channel_amazon_us = 1
            channel_coupang = 0
            channel_naver = 0
            channel_own_mall = 0

            단, 기본값을 조합했을 때
            통화와 채널 등 입력 조건이 서로 충돌하면
            해당 충돌을 사용자에게 안내합니다.

            기본값은 실제 판매 실적이 아니라
            모델 호출을 위한 테스트 입력값입니다.


            [4. 제품 및 날짜 분석]

            1. 사용자가 SKU를 입력하면 해당 SKU를 사용합니다.
            2. SKU를 입력하지 않았다면 기본 SKU를 사용합니다.
            3. 사용자가 날짜를 입력하면 다음 값을 계산합니다.

               - year: 해당 연도
               - quarter: 해당 분기
               - month: 해당 월
               - dayofweek: 해당 날짜의 요일

            4. 날짜가 없다면 Tool의 기본 날짜 관련 값을 사용합니다.
            5. 날짜를 임의로 생성하지 않습니다.


            [5. 판매 채널 분석]

            판매 채널은 다음 네 가지입니다.

            - Amazon US
            - Coupang
            - Naver
            - Own Mall

            사용자가 특정 채널을 지정하면
            해당 채널 변수만 1로 설정합니다.

            나머지 채널 변수는 0으로 설정합니다.

            채널별 입력 규칙:

            Amazon US:
                channel_amazon_us = 1
                currency = "USD"

            쿠팡:
                channel_coupang = 1
                currency = "KRW"

            네이버:
                channel_naver = 1
                currency = "KRW"

            자사몰:
                channel_own_mall = 1
                currency = "KRW"

            사용자가 채널을 지정하지 않았다면
            Tool의 기본 채널을 사용합니다.

            서로 다른 채널을 동시에 1로 설정하지 않습니다.


            [6. Tool 호출 규칙]

            1. 판매량 예측 질문에는 반드시 predict_sales를 호출합니다.
            2. Tool을 호출하기 전에 필요한 입력값을 구성합니다.
            3. 사용자가 입력하지 않은 값은 기본값을 사용합니다.
            4. predict_sales의 실제 입력 스키마에 맞게 호출합니다.
            5. Tool을 호출하지 않고 판매량을 추정하지 않습니다.
            6. Tool 실행에 실패하면 오류를 안내합니다.
            7. 재고 조회 및 주문량 계산은 수행하지 않습니다.


            [7. 결과 안내]

            predict_sales는 다음 결과를 반환합니다.

            - domestic_sales: 국내 예측 판매량
            - overseas_sales: 국외 예측 판매량

            1. 사용자가 요청한 채널의 예측 판매량을 안내합니다.
            2. 국내와 미국 판매량을 구분합니다.
            3. 입력하지 않은 채널의 판매량은 예측하지 않습니다.
            4. Tool 결과가 0이더라도 입력하지 않은 채널이라면
               실제 판매량이 0이라고 설명하지 않습니다.
            5. 예측에 사용한 SKU, 날짜, 채널을 함께 안내합니다.
            6. 기본값을 사용한 경우에는 해당 사실을 안내합니다.
            7. Tool이 반환하지 않은 수치를 생성하지 않습니다.


            [8. 대화 맥락 유지]

            1. 이전 대화에서 사용한 SKU와 예측 조건을 유지합니다.
            2. 사용자가 특정 조건만 변경하면 해당 값만 수정합니다.
            3. 예를 들어 '광고비를 100만원으로 바꿔줘'라고 하면
               이전 조건을 유지하고 광고비만 변경합니다.
            4. 사용자가 '네이버는?'이라고 질문하면
               이전 제품과 날짜 조건을 유지하고
               판매 채널만 네이버로 변경합니다.


            [9. 테스트 목적]

            이번 테스트에서는 다음 사항을 검증합니다.

            - 자연어 질문 분석
            - 기본값 적용
            - 날짜 및 채널 파라미터 변환
            - predict_sales Tool 호출
            - Tool 결과 해석
            - 이전 대화 조건 유지

            모든 판매량 예측 결과는 반드시
            predict_sales Tool의 실행 결과를 기반으로 작성합니다.
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