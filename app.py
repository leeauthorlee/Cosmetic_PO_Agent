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

from typing import Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    before_model,
)

from langchain.messages import AIMessage, ToolMessage
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

# ------------------------------------------
# 2-1. 기능 1 - 재고 예측
# ------------------------------------------

# ------------------------------------------
# 함수 1-1: 판매 데이터 불러오기
# ------------------------------------------

# 기존 데이터 호출
df_history = pd.read_csv('data/sales_weekly.csv')
df_product_master = pd.read_csv('data/product_master.csv')
price_map = df_product_master.set_index('sku')['list_price_krw']

df_history['promo_price'] = (
    df_history['sku'].map(price_map)
    * (1 - df_history['promo_discount'])
)

# 한국 모델
xgbr_k = joblib.load('models/xgbr_model_k.joblib')
mms_k = joblib.load('models/minmax_scaler_k.joblib')

# 미국 모델
xgbr_u = joblib.load('models/xgbr_model_u.joblib')
mms_u = joblib.load('models/minmax_scaler_u.joblib')


# ------------------------------------------
# 함수 1-2: 날짜 생성 함수
# ------------------------------------------
def create_future_dates(
    df_history: pd.DataFrame,
    start_after_weeks: int = 1,
    weeks: int = 3,
    start_date: Optional[str] = None
) -> pd.DatetimeIndex:

    if weeks < 1:
        raise ValueError("예측 기간은 1주 이상이어야 합니다.")

    # 사용자가 특정 날짜를 입력한 경우
    if start_date is not None:

        target_date = pd.Timestamp(start_date)

        if pd.isna(target_date):
            raise ValueError("유효한 시작 날짜를 입력해야 합니다.")

        # 해당 주 월요일로 보정
        start_monday = target_date - pd.Timedelta(
            days=target_date.weekday()
        )

    # 특정 날짜가 없다면 기존 방식 유지
    else:
        if start_after_weeks < 1:
            raise ValueError("시작 시점은 1주 이상이어야 합니다.")

        last_date = pd.to_datetime(
            df_history['week_start']
        ).max()

        base_monday = last_date - pd.Timedelta(
            days=last_date.weekday()
        )

        start_monday = base_monday + pd.Timedelta(
            weeks=start_after_weeks
        )

    future_dates = pd.date_range(
        start=start_monday,
        periods=weeks,
        freq='W-MON'
    )

    return future_dates



# ------------------------------------------
# 함수 1-3: 입력데이터셋 생성 함수
# ------------------------------------------

FEATURE_COLUMNS = [
    'sku',
    'ad_spend',
    'promo_days',
    'promo_discount',
    'quarter',
    'year',
    'month',
    'day',
    'promo_price',
    'channel_coupang',
    'channel_naver',
    'channel_own_mall'
]

US_FEATURE_COLUMNS = [
    'sku',
    'ad_spend',
    'promo_days',
    'promo_discount',
    'quarter',
    'year',
    'month',
    'day',
    'promo_price'
]

KR_CHANNELS = [
    "coupang",
    "naver",
    "own_mall"
]

CHANNEL_MAP = {
    "쿠팡": "coupang",
    "coupang": "coupang",
    "네이버": "naver",
    "naver": "naver",
    "자사몰": "own_mall",
    "자사 몰": "own_mall",
    "own_mall": "own_mall",
    "own mall": "own_mall"
}

SKU_MAP = {
    'A-SRM-100': 0,
    'A-CRM-050': 1,
    'A-CAP-HY': 2,
    'A-CAP-VC': 3,
    'A-CAP-PN': 4
}

def normalize_channel(channel: Optional[str]) -> Optional[str]:

    if channel is None:
        return None

    channel = channel.strip().lower()

    if channel not in CHANNEL_MAP:
        raise ValueError(f"지원하지 않는 채널: {channel}")

    return CHANNEL_MAP[channel]



def create_future_dataset(
    df_history: pd.DataFrame,
    channels: list[str],
    skus: list[str],
    weeks: int,
    start_after_weeks: int = 1,
    start_date: Optional[str] = None,
    ad_spend_change_pct: float = 0.0,
    promo_discount: Optional[float] = None,
    promo_days: Optional[int] = None
) -> pd.DataFrame:

    df = df_history.copy()
    df['week_start'] = pd.to_datetime(df['week_start'])

    if weeks < 1 or start_after_weeks < 1:
        raise ValueError("예측 주차와 시작 시점은 1 이상이어야 합니다.")

    if not channels or not skus:
        raise ValueError("채널과 제품 목록이 비어 있습니다.")

    if not set(channels).issubset(KR_CHANNELS):
        raise ValueError("지원하지 않는 국내 채널입니다.")

    if promo_discount is not None and not (0 <= promo_discount <= 1):
        raise ValueError("할인율은 0~1 사이여야 합니다.")

    if promo_days is not None and not (0 <= promo_days <= 7):
        raise ValueError("주간 프로모션 기간은 0~7일이어야 합니다.")

    future_dates = create_future_dates(
            df_history=df,
            start_after_weeks=start_after_weeks,
            weeks=weeks,
            start_date=start_date
        )

    rows = []

    for channel in channels:
        for sku in skus:

            history = df[
                (df['channel'] == channel) &
                (df['sku'] == sku)
            ].sort_values('week_start')

            if history.empty:
                raise ValueError(
                    f"{channel} / {sku} 판매 이력이 없습니다."
                )

            base = history.iloc[-1]

            # 광고비 증감률
            new_ad_spend = float(base['ad_spend']) * (
                1 + ad_spend_change_pct / 100
            )

            # 사용자 지정 할인율이 있으면 사용
            new_discount = (
                promo_discount
                if promo_discount is not None
                else float(base['promo_discount'])
            )

            # 사용자 지정 프로모션 기간
            new_promo_days = (
                promo_days
                if promo_days is not None
                else int(base['promo_days'])
            )

            # 할인 전 판매가 계산
            # 기존 promo_price가 할인 적용 가격이라고 가정
            base_discount = float(base['promo_discount'])
            base_promo_price = float(base['promo_price'])

            if base_discount >= 1:
                raise ValueError(
                    "기존 할인율이 100%여서 정가를 역산할 수 없습니다."
                )

            list_price = base_promo_price / (1 - base_discount)

            # 변경된 할인율로 프로모션 가격 재계산
            new_promo_price = list_price * (1 - new_discount)

            for date in future_dates:

                row = {
                    'sku': sku,  # 직접 구현할 SKU 인코딩 적용
                    'ad_spend': new_ad_spend,
                    'promo_days': new_promo_days,
                    'promo_discount': new_discount,
                    'quarter': date.quarter,
                    'year': date.year,
                    'month': date.month,
                    'day': date.day,
                    'promo_price': new_promo_price,
                    'channel_coupang': int(channel == 'coupang'),
                    'channel_naver': int(channel == 'naver'),
                    'channel_own_mall': int(channel == 'own_mall')
                }

                rows.append(row)

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)

# ==============
# 아마존 데이터셋 구성
# ==============

def create_future_dataset_us(
    df_history: pd.DataFrame,
    df_product_master: pd.DataFrame,
    skus: list[str],
    weeks: int = 1,
    start_date: Optional[str] = None,
    start_after_weeks: int = 1,
    ad_spend_change_pct: float = 0.0,
    promo_discount: Optional[float] = None,
    promo_days: Optional[int] = None
) -> pd.DataFrame:

    if ad_spend_change_pct < -100:
        raise ValueError("광고비 감소율은 -100% 이상이어야 합니다.")

    if promo_discount is not None and not 0 <= promo_discount <= 1:
        raise ValueError("할인율은 0~1 사이여야 합니다.")

    if promo_days is not None and not 0 <= promo_days <= 7:
        raise ValueError("프로모션 일수는 0~7일이어야 합니다.")

    # 아마존 데이터만 추출
    df_us = df_history[
        df_history['channel'] == 'amazon_us'
    ].copy()

    df_us['week_start'] = pd.to_datetime(
        df_us['week_start']
    )

    if df_us.empty:
        raise ValueError("아마존 판매 이력이 없습니다.")

    # 아마존 마지막 학습 주를 기준으로 날짜 생성
    future_dates = create_future_dates(
        df_history=df_us,
        start_after_weeks=start_after_weeks,
        weeks=weeks,
        start_date=start_date
    )

    # 제품별 USD 정가
    usd_price_map = (
        df_product_master
        .set_index('sku')['list_price_usd']
    )

    rows = []

    for sku in skus:

        history = df_us[
            df_us['sku'] == sku
        ].sort_values('week_start')

        if history.empty:
            raise ValueError(
                f"아마존 판매 이력이 없는 제품: {sku}"
            )

        if sku not in usd_price_map.index:
            raise ValueError(
                f"USD 정가가 없는 제품: {sku}"
            )

        base = history.iloc[-1]

        # 마지막 광고비 기준 증감률 적용
        new_ad_spend = float(base['ad_spend']) * (
            1 + ad_spend_change_pct / 100
        )

        # 할인율
        new_discount = (
            float(promo_discount)
            if promo_discount is not None
            else float(base['promo_discount'])
        )

        # 프로모션 일수
        new_promo_days = (
            int(promo_days)
            if promo_days is not None
            else int(base['promo_days'])
        )

        # USD 기준 프로모션 가격
        list_price_usd = float(usd_price_map[sku])

        new_promo_price = (
            list_price_usd * (1 - new_discount)
        )

        for date in future_dates:

            rows.append({
                'week_start': date,
                'original_sku': sku,
                'sku': SKU_MAP[sku],
                'ad_spend': new_ad_spend,
                'promo_days': new_promo_days,
                'promo_discount': new_discount,
                'quarter': date.quarter,
                'year': date.year,
                'month': date.month,
                'day': date.day,
                'promo_price': new_promo_price
            })

    return pd.DataFrame(rows)

# ------------------------------------------
# Tool 1: 국내 / 국외 판매량 예측
# ------------------------------------------


@tool
def predict_sales_kr(
    channel: Optional[str] = None,
    sku: Optional[str | list[str]] = None,
    start_date: Optional[str] = None,
    start_after_weeks: int = 1,
    weeks: int = 1,
    ad_spend_change_pct: float = 0.0,
    promo_discount: Optional[float] = None,
    promo_days: Optional[int] = None
) -> dict:
    """
    국내 화장품(국내 채널 쿠팡, 네이버, 자사몰) 판매량을 XGBoost 모델로 예측합니다.

    sku: 단일 SKU 또는 여러 SKU의 목록. 생략하면 전체 제품을 예측합니다.

    start_date:
        사용자가 특정 날짜를 지정한 경우 YYYY-MM-DD 형식으로 입력.
        해당 날짜가 속한 주의 월요일부터 예측합니다.
        예: 2026-10-20 -> 2026-10-19

    start_after_weeks:
        start_date가 없는 경우에만 사용합니다.
        학습 데이터 마지막 주 기준 N주 뒤부터 예측합니다.

    weeks:
        예측할 주차 수.

    ad_spend_change_pct:
        광고비 증감률(%).
        예: 10이면 10% 증가.

    promo_discount:
        할인율. 예: 0.3이면 30%.

    promo_days:
        주간 프로모션 진행 일수.
    """

    # 1. 입력값 검증
    if weeks < 1 or start_after_weeks < 1:
        return {"error": "예측 기간과 시작 시점은 1 이상이어야 합니다."}

    if ad_spend_change_pct < -100:
        return {"error": "광고비 감소율은 -100%보다 작을 수 없습니다."}

    if promo_discount is not None:
        if not 0 <= promo_discount <= 1:
            return {"error": "할인율은 0~1 사이여야 합니다."}

    if promo_days is not None:
        if not 0 <= promo_days <= 7:
            return {"error": "프로모션 일수는 0~7일이어야 합니다."}

    # 2. 채널 선택
    try:
        normalized_channel = normalize_channel(channel)

    except ValueError as e:
        return {"error": str(e)}

    if normalized_channel is None:
        channels = KR_CHANNELS
    else:
        channels = [normalized_channel]

    # 3. 제품 선택
    if sku is None:
        skus = sorted(df_history[
            df_history['channel'].isin(channels)
        ]['sku'].dropna().unique().tolist())
    else:
        skus = list(dict.fromkeys(sku if isinstance(sku, list) else [sku]))

    # 4. 미래 입력 데이터 생성
    try:
        future_df = create_future_dataset(
            df_history=df_history,
            channels=channels,
            skus=skus,
            weeks=weeks,
            start_date=start_date,
            start_after_weeks=start_after_weeks,
            ad_spend_change_pct=ad_spend_change_pct,
            promo_discount=promo_discount,
            promo_days=promo_days
        )
    except (ValueError, KeyError) as e:
        return {"error": str(e)}
    
    # sku 인코딩
    original_skus = future_df["sku"].copy()
    future_df['sku'] = future_df['sku'].replace({'A-SRM-100':0, 'A-CRM-050':1, 'A-CAP-HY':2, 'A-CAP-VC':3, 'A-CAP-PN':4})

    # 5. 모델 입력 데이터 구성
    X_future = future_df[FEATURE_COLUMNS].copy()

    # SKU 인코딩은 create_future_dataset에서
    # 학습 당시 방식으로 처리했다고 가정

    # 6. 학습 당시 전처리 적용
    # 아래는 전체 12개 컬럼을 scaler로 학습한 경우의 예시
    # 실제 학습 당시 일부 컬럼만 스케일링했다면
    # 해당 컬럼에만 transform 적용해야 함


    X_scaled = pd.DataFrame(
        mms_k.transform(X_future),
        columns=FEATURE_COLUMNS,
        index=X_future.index
    )


    # 7. 판매량 예측
    # XGBoost 모델이 학습 시 기대하는 입력을 사용
    predictions = xgbr_k.predict(X_scaled)

    # 판매량은 음수가 될 수 없으므로 0 이상으로 보정
    predictions = np.maximum(predictions, 0)

    # 8. 결과 DataFrame
    result_df = future_df.copy()
    result_df["sku"] = original_skus
    result_df["predicted_units"] = predictions
    # 집계 전 제품·채널·날짜 정보를 보존하여 여러 제품을 화면에서 비교합니다.
    result_df["channel"] = result_df.apply(
        lambda row: next(c for c in KR_CHANNELS if row[f"channel_{c}"] == 1), axis=1
    )
    result_df["week_start"] = pd.to_datetime(
        result_df[["year", "month", "day"]]
    ).dt.strftime("%Y-%m-%d")

    # 9. 결과 집계
    total_units = float(result_df['predicted_units'].sum())

    weekly_result = (
        result_df
        .groupby(['year', 'month', 'day'], as_index=False)
        ['predicted_units']
        .sum()
    )

    product_result = (
        result_df
        .groupby('sku', as_index=False)
        ['predicted_units']
        .sum()
    )

    return {
        "status": "success",
        "conditions": {
            "channels": channels,
            "skus": skus,
            "start_date": str(
                create_future_dates(
                    df_history=df_history,
                    start_date=start_date,
                    start_after_weeks=start_after_weeks,
                    weeks=weeks
                )[0].date()
            ),
            "start_after_weeks": start_after_weeks,
            "weeks": weeks,
            "ad_spend_change_pct": ad_spend_change_pct,
            "promo_discount": promo_discount,
            "promo_days": promo_days
        },
        "total_predicted_units": round(total_units, 2),
        "detailed_predictions": result_df[
            ["sku", "channel", "week_start", "predicted_units"]
        ].to_dict(orient="records"),
        "weekly_predictions": weekly_result.to_dict(
            orient="records"
        ),
        "product_predictions": product_result.to_dict(
            orient="records"
        )
    }


# ==========
# 아마존 예측 툴
# ==========
@tool
def predict_sales_us(
    sku: Optional[str] = None,
    start_date: Optional[str] = None,
    start_after_weeks: int = 1,
    weeks: int = 1,
    ad_spend_change_pct: float = 0.0,
    promo_discount: Optional[float] = None,
    promo_days: Optional[int] = None
) -> dict:
    """
    아마존 미국(amazon_us) 채널의 화장품 판매량을
    미국 전용 XGBoost 모델로 예측합니다.

    사용자가 아마존, Amazon, Amazon US, 미국 판매량,
    아마존 광고비 또는 할인율 변경을 요청하면 사용합니다.

    sku:
        제품 SKU. None이면 아마존 전체 제품.

    start_date:
        YYYY-MM-DD. 입력 날짜가 속한 주의 월요일부터 예측.

    start_after_weeks:
        start_date가 없으면 아마존 학습 데이터
        마지막 주 기준 N주 뒤부터 예측.

    weeks:
        예측 주차 수.

    ad_spend_change_pct:
        광고비 증감률(%). 20이면 20% 증가.

    promo_discount:
        할인율. 0.3이면 30% 할인.

    promo_days:
        주간 프로모션 진행 일수.
    """

    try:
        # 1. 제품 선택
        df_us = df_history[
            df_history['channel'] == 'amazon_us'
        ]

        if sku is None:
            skus = sorted(
                df_us['sku'].dropna().unique().tolist()
            )
        else:
            skus = [sku]

        if not skus:
            raise ValueError("예측 가능한 아마존 제품이 없습니다.")

        # 2. 기준 시나리오 생성
        baseline_df = create_future_dataset_us(
            df_history=df_history,
            df_product_master=df_product_master,
            skus=skus,
            weeks=weeks,
            start_date=start_date,
            start_after_weeks=start_after_weeks,
            ad_spend_change_pct=0,
            promo_discount=promo_discount,
            promo_days=promo_days
        )

        # 3. 광고비 변경 시나리오 생성
        scenario_df = create_future_dataset_us(
            df_history=df_history,
            df_product_master=df_product_master,
            skus=skus,
            weeks=weeks,
            start_date=start_date,
            start_after_weeks=start_after_weeks,
            ad_spend_change_pct=ad_spend_change_pct,
            promo_discount=promo_discount,
            promo_days=promo_days
        )

        # 4. 예측 함수
        def run_prediction(df):

            X = df[US_FEATURE_COLUMNS].copy()

            # 미국 모델 전용 스케일러
            X_scaled = pd.DataFrame(
                mms_u.transform(X),
                columns=US_FEATURE_COLUMNS,
                index=X.index
            )

            predictions = xgbr_u.predict(X_scaled)

            return np.maximum(predictions, 0)

        # 5. 기준 / 변경 예측
        baseline_predictions = run_prediction(baseline_df)
        scenario_predictions = run_prediction(scenario_df)

        # 6. 결과 DataFrame
        result_df = scenario_df[
            ['week_start', 'original_sku']
        ].copy()

        result_df['baseline_units'] = baseline_predictions
        result_df['scenario_units'] = scenario_predictions

        result_df['difference_units'] = (
            result_df['scenario_units']
            - result_df['baseline_units']
        )

        # 7. 전체 판매량 집계
        baseline_total = float(
            result_df['baseline_units'].sum()
        )

        scenario_total = float(
            result_df['scenario_units'].sum()
        )

        difference = scenario_total - baseline_total

        change_pct = (
            difference / baseline_total * 100
            if baseline_total > 0
            else None
        )

        # 8. 주차별 결과
        weekly = (
            result_df
            .groupby('week_start', as_index=False)[
                [
                    'baseline_units',
                    'scenario_units',
                    'difference_units'
                ]
            ]
            .sum()
        )

        weekly['week_start'] = (
            weekly['week_start']
            .dt.strftime('%Y-%m-%d')
        )

        # 9. 제품별 결과
        products = (
            result_df
            .groupby('original_sku', as_index=False)[
                [
                    'baseline_units',
                    'scenario_units',
                    'difference_units'
                ]
            ]
            .sum()
        )

        # 화면에 제품·주차별 실제 예측값을 전달합니다.
        details = result_df.rename(columns={"original_sku": "sku"}).copy()
        details["week_start"] = details["week_start"].dt.strftime("%Y-%m-%d")
        details["channel"] = "amazon_us"
        details["predicted_units"] = details["scenario_units"]

        # 10. Agent에 결과 반환
        return {
            "status": "success",
            "channel": "amazon_us",
            "currency": "USD",
            "conditions": {
                "skus": skus,
                "start_date": str(
                    scenario_df['week_start'].min().date()
                ),
                "weeks": weeks,
                "ad_spend_change_pct": ad_spend_change_pct,
                "promo_discount": promo_discount,
                "promo_days": promo_days
            },
            "baseline_total_units": round(baseline_total, 2),
            "scenario_total_units": round(scenario_total, 2),
            "detailed_predictions": details.to_dict(orient="records"),
            "difference_units": round(difference, 2),
            "change_pct": (
                round(change_pct, 2)
                if change_pct is not None
                else None
            ),
            "weekly_predictions": weekly.to_dict(
                orient="records"
            ),
            "product_predictions": products.to_dict(
                orient="records"
            )
        }

    except (ValueError, KeyError, TypeError) as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ------------------------------------------
# 함수 2-1: 재고 데이터 불러오기
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
# 함수 2-2: 재고량 파악
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
# 함수 2-3: 추가 주문량 계산
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
    df = load_inventory_data("data/inventory_now.csv", "data/product_master.csv")

    # 특정 제품의 재고량 확인
    current_stock = find_inventory_data(sku, df)

    # 추가 주문량 계산
    result = calculate_order_quantity(current_stock, sales_week1, sales_week2, sales_week3)

    return result


# ------------------------------------------
# 2-2. 기능 2 - CSV 생성
# ------------------------------------------

class ProductForecast(BaseModel):
    sku: str
    predicted_units: float = Field(ge=0, allow_inf_nan=False)


@tool
def check_inventory_batch(predictions: list[ProductForecast]) -> dict:
    """여러 제품의 국내 주문량을 계산합니다.

    predict_sales_kr의 product_predictions를 predictions에 그대로 전달하세요.
    각 제품의 전체 예측 기간 판매량을 국내 재고와 비교합니다.
    """
    df = load_inventory_data("data/inventory_now.csv", "data/product_master.csv")
    orders, errors = [], []
    seen = set()
    for prediction in predictions:
        if prediction.sku in seen:
            continue
        seen.add(prediction.sku)
        stock = df[(df["sku"] == prediction.sku) & (df["location"] == "KR_WH")]
        if stock.empty:
            errors.append(f"{prediction.sku}: 국내 창고 재고가 없습니다.")
            continue
        result = calculate_order_quantity(
            int(stock["on_hand"].iloc[0]), prediction.predicted_units, 0, 0
        )
        orders.append({"sku": prediction.sku, **result})
    return {"orders": orders, "errors": errors}


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

    messages = state["messages"]

    if not messages:
        return {}

    # Tool 실행 결과가 들어온 경우 재분류하지 않음
    if isinstance(messages[-1], ToolMessage):
        return {}

    # 마지막 메시지가 사용자 메시지가 아니면 재분류하지 않음
    if messages[-1].type != "human":
        return {}

    user_input = messages[-1].content

    result = filter_llm.invoke([
        (
            "system",
            """
            당신은 재고 관리 AI 서비스의 질문 분류기입니다.

            재고, 주문, 판매량 예측, 광고비 변경,
            프로모션, 제품 및 판매 채널과 관련된
            질문이면 is_relevant=True로 판단하세요.

            무관한 질문이면 False로 판단하세요.
            """
        ),
        ("human", str(user_input))
    ])

    if not result.is_relevant:
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=(
                            "재고 관리 및 판매량 예측과 "
                            "관련된 질문만 답변할 수 있습니다."
                        )
                    )
                ]
            },
            goto="__end__"
        )

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
            tools=[predict_sales_kr, predict_sales_us, check_inventory, check_inventory_batch],
            middleware=[check_order_question],
            checkpointer=checkpointer,
            system_prompt = """
            당신은 화장품 판매량 예측 AI Agent입니다.
            무조건 Tool을 호출여여 도출된 결과로 판단하세요.

            [국내/해외 판매량 예측 Tool 선택 규칙]

            1. 사용자가 쿠팡, 네이버, 자사몰의 판매량을 요청하면
                predict_sales_kr 도구를 사용하세요.

            2. 사용자가 아마존, Amazon, Amazon US, 미국 채널의
                판매량을 요청하면 predict_sales_us 도구를 사용하세요.

            3. 아마존은 미국 전용 예측 모델을 사용합니다.
                아마존 요청을 predict_sales_kr에 전달하지 마세요.

            4. 국내와 미국 모델은 입력 Feature와 Scaler가 다르므로
                반드시 각각의 Tool을 사용하세요.

            5. 광고비 10% 증가는 ad_spend_change_pct=10입니다.

            6. 할인율 30%는 promo_discount=0.3입니다.

            7. 사용자가 판매량 증가분을 물어보면
                기준 시나리오와 변경 시나리오의 차이를 설명하세요.

            8. 예측 결과는 실제 판매량이 아닌 모델 추정치입니다.


            [날짜 처리 규칙]

            1. 사용자가 특정 날짜를 지정하거나
                start_date에 YYYY-MM-DD 형식으로 전달하세요.
            1-1. 사용자가 '다음달', '다다음달'처럼 입력하면
                start_date에 해당월 1일 기준으로 YYYY-MM-1 형식으로 전달하세요
            1-1-1. 사용자가 '다음달', '다다음달'처럼 입력하고 판매량 예측 규칙에 '몇주간'입력이 없다면
                weeks에 해당 월 (일수/7)한 값의 정수값을 전달합니다.
                (예시) 해당 월이 31일까지 있다면 weeks=(31/7)int 입니다.
            1-2. 사용자가 특정 월을 입력하면 
                start_dat가에 해당월 1일 기준으로 YYYY-MM-1 형식으로 전달하세요
            1-2-1. 사용자가 '다음달', '다다음달'처럼 입력하고 판매량 예측 규칙에 '몇주간'입력이 없다면
                weeks에 해당 월 (일수/7)한 값의 정수값을 전달합니다.
                (예시) 해당 월이 31일까지 있다면 weeks=(31/7)int 입니다.

            2. 특정 날짜가 지정되었다면
                start_after_weeks는 사용하지 않습니다.

            3. 특정 날짜가 없고 '3주 뒤부터'라고 하면
                start_after_weeks=3으로 전달하세요.

            4. 특정 날짜가 없으면 각 Tool이 사용하는
                학습 데이터 마지막 주를 기준으로 예측합니다.

            5. 예측 시작일은 항상 해당 주의 월요일로 보정됩니다.
                날짜 보정은 Tool 내부에서 수행합니다.

            [판매량 예측 규칙]

            6. '3주간'은 weeks=3입니다.

            7. 광고비 10% 증가는
                ad_spend_change_pct=10입니다.

            8. 할인율 30%는 promo_discount=0.3입니다.

            9. 제품을 지정하지 않으면 전체 제품을 예측합니다.

            10. 채널을 지정하지 않으면 전체 국내 채널과 미국(아마존, AMAZON)채널을 예측합니다.
            10-1. 영문명과 한글명 모두, 비슷한 의미의 채널은 모두 같은 채널입니다.
            [판매 채널 변환 규칙]

                사용자가 한국어 채널명을 입력하면
                반드시 내부 채널명으로 변환하세요.

                - 쿠팡 → coupang
                - 네이버 → naver
                - 자사몰 → own_mall

                predict_sales_kr 도구의 channel 인자에는
                가능하면 내부 채널명을 전달하세요.

            11. 예측 결과는 실제 판매량이 아닌 모델 추정치입니다.

            12. 도구가 오류를 반환하면 결과를 임의로 생성하지 마세요.

            [상대 날짜 해석 규칙]

                - 이번 달: 현재 날짜가 속한 달
                - 다음 달: 현재 날짜 기준 1개월 뒤
                - 다다음달: 현재 날짜 기준 2개월 뒤

                특정 월을 요청하면 해당 월에 속하는 주차를
                예측 대상으로 설정하세요.
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
        elif name in ("predict_sales_kr", "predict_sales_us"):
            try:
                result = json.loads(content) if isinstance(content, str) else content
                if not isinstance(result, dict) or result.get("status") != "success":
                    continue
                for row in result.get("detailed_predictions", []):
                    units = row["predicted_units"]
                    if isinstance(units, bool) or not isinstance(units, (int, float)):
                        continue
                    if not math.isfinite(units) or units < 0:
                        continue
                    display_row = {
                        "SKU": row["sku"],
                        "판매 사이트": CHANNELS.get(f"channel_{row['channel']}", row["channel"]),
                        "기간": row["week_start"], "예상 판매량(개)": units
                    }
                    if name == "predict_sales_us":
                        baseline = row.get("baseline_units")
                        difference = row.get("difference_units")
                        if all(isinstance(v, (int, float)) and not isinstance(v, bool)
                               and math.isfinite(v) for v in (baseline, difference)):
                            display_row["기준 판매량(개)"] = baseline
                            display_row["판매량 변화(개)"] = difference
                    sales.append(display_row)
            except (ValueError, TypeError, KeyError):
                continue
        elif name in ("check_inventory", "check_inventory_batch"):
            try:
                result = json.loads(content) if isinstance(content, str) else content
                rows = result.get("orders", []) if name == "check_inventory_batch" else [result]
                for row in rows:
                    values = [row[key] for key in ("predicted_sales", "required_qty", "order_qty")]
                    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                               and math.isfinite(v) and v >= 0 for v in values):
                        continue
                    orders.append({"SKU": row.get("sku", args.get("sku", "미확인")), "지역": "국내",
                                   "창고": "KR_WH", "예상 판매량(개)": values[0],
                                   "부족 수량(개)": values[1], "주문량(개)": values[2]})
            except (ValueError, TypeError, KeyError):
                continue
    return {"sales": sales, "orders": orders}


def build_sales_chart(frame, site):
    """전체 탭은 지역·채널별, 사이트 탭은 날짜·제품별 누적 그래프입니다."""
    quantity = "예상 판매량(개)"
    if site == "전체":
        channels = ["자사몰", "네이버", "쿠팡", "아마존(US)"]
        colors = ["#896451", "#B8836C", "#C99B87", "#F1BCA6"]
        plot = frame.copy()
        plot["채널"] = plot["판매 사이트"].replace({"Amazon US": "아마존(US)"})
        plot["지역"] = plot["채널"].map({
            "자사몰": "국내", "네이버": "국내", "쿠팡": "국내", "아마존(US)": "국외"
        }).fillna("미분류")
        plot = plot.groupby(["지역", "채널"], as_index=False)[quantity].sum()
        plot["순서"] = plot["채널"].map({name: i for i, name in enumerate(channels)}).fillna(4)
        plot = plot.sort_values(["지역", "순서"])
        plot["누적"] = plot.groupby("지역")[quantity].cumsum()
        plot["중앙"] = plot["누적"] - plot[quantity] / 2
        extras = [name for name in plot["채널"].unique() if name not in channels]
        base = alt.Chart(plot).encode(
            x=alt.X("지역:N", sort=["국내", "국외", "미분류"], title=None,
                    axis=alt.Axis(labelAngle=0))
        )
        bars = base.mark_bar(size=72).encode(
            y=alt.Y(f"{quantity}:Q", stack="zero", title="판매량 (개)"),
            color=alt.Color("채널:N", scale=alt.Scale(domain=channels + extras,
                            range=colors + ["#AAAAAA"] * len(extras)),
                            legend=alt.Legend(title=None, orient="right")),
            order=alt.Order("순서:Q", sort="ascending"),
            tooltip=["지역", "채널", alt.Tooltip(f"{quantity}:Q", format=",.0f")]
        )
        labels = base.transform_filter(alt.datum[quantity] > 0).mark_text(fontSize=12).encode(
            y=alt.Y("중앙:Q", title="판매량 (개)"),
            text=alt.Text(f"{quantity}:Q", format=",.0f"),
            color=alt.condition(alt.datum["순서"] < 2, alt.value("white"), alt.value("#392822"))
        )
        totals = base.transform_aggregate(합계=f"sum({quantity})", groupby=["지역"]).mark_text(
            dy=-12, fontSize=15, fontWeight="bold", color="#302C3D"
        ).encode(y=alt.Y("합계:Q", title="판매량 (개)"), text=alt.Text("합계:Q", format=",.0f"))
        chart = bars + labels + totals
    else:
        plot = frame.groupby(["기간", "SKU"], as_index=False)[quantity].sum().sort_values("기간")
        dates = pd.to_datetime(plot["기간"], errors="coerce")
        weekdays = ["월", "화", "수", "목", "금", "토", "일"]
        plot["날짜"] = [
            f"{date:%m-%d} ({weekdays[date.weekday()]})" if pd.notna(date) else str(period)
            for date, period in zip(dates, plot["기간"])
        ]
        date_order = plot["날짜"].drop_duplicates().tolist()
        preferred = ["A-SRM-100", "A-CRM-050", "A-CAP-HY", "A-CAP-VC", "A-CAP-PN"]
        products = preferred + sorted(set(plot["SKU"]) - set(preferred))
        palette = ["#F0B1A5", "#D9A08D", "#C08F7B", "#A77F69", "#896451"]
        palette += ["#707D8C", "#A092A8", "#94A58B"] * ((len(products) // 3) + 1)
        plot["순서"] = plot["SKU"].map({sku: i for i, sku in enumerate(products)})
        chart = alt.Chart(plot).mark_bar(size=22).encode(
            x=alt.X("날짜:N", title="날짜", sort=date_order,
                    axis=alt.Axis(labelAngle=0, labelOverlap=True)),
            y=alt.Y(f"{quantity}:Q", stack="zero", title="판매량 (개)"),
            color=alt.Color("SKU:N", scale=alt.Scale(domain=products, range=palette[:len(products)]),
                            legend=alt.Legend(title=None, orient="right")),
            order=alt.Order("순서:Q", sort="descending"),
            tooltip=["SKU", "기간", alt.Tooltip(f"{quantity}:Q", format=",.0f")]
        )
    return chart.properties(height=280).configure_view(strokeWidth=0).configure_axis(
        gridColor="#F0EEEC", domainColor="#B5B0AD", tickColor="#B5B0AD",
        labelColor="#665F69", titleColor="#665F69", labelFontSize=12, titleFontSize=13
    ).configure_axisX(grid=False).configure_axisY(format=",.0f", tickCount=5).configure_legend(
        labelFontSize=12, symbolType="square", symbolSize=130
    )


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
            chart = build_sales_chart(frame, site)
            st.altair_chart(chart, width="stretch")
            with st.expander("판매량 상세 표", expanded=True):
                st.dataframe(frame.dropna(axis=1, how="all"), hide_index=True, width="stretch")
        st.caption("조회된 예측 결과만 합산합니다. 결과가 없는 채널은 0개를 의미하지 않습니다. 날짜는 예측 주의 시작일입니다.")
        if "Amazon US" in frame["판매 사이트"].values:
            st.caption("아마존 그래프는 입력 조건을 적용한 예상 판매량입니다. 표의 기준 판매량은 광고비 증감 전, 판매량 변화는 광고비 증감 후와의 차이입니다.")
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
            comparison = frame.melt(
                id_vars=["SKU", "지역", "창고"],
                value_vars=["예상 판매량(개)", "부족 수량(개)", "주문량(개)"],
                var_name="항목", value_name="수량(개)"
            )
            chart = alt.Chart(comparison).mark_bar().encode(
                x=alt.X("SKU:N", title="제품", axis=alt.Axis(labelAngle=0)),
                xOffset="항목:N",
                y=alt.Y("수량(개):Q", title="수량(개)"),
                color=alt.Color("항목:N", scale=alt.Scale(range=["#FFBCA6", "#DFA18C", "#8D655B"]),
                                legend=alt.Legend(title=None, orient="bottom")),
                tooltip=["SKU", "지역", "창고", "항목", alt.Tooltip("수량(개):Q", format=",.2f")]
            ).properties(height=260).configure_view(strokeWidth=0)
            st.altair_chart(chart, width="stretch")
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
            st.session_state.order_quantity = sum(row["주문량(개)"] for row in display_results["orders"])

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
    st.session_state.order_quantity = sum(row["주문량(개)"] for row in rows) if rows else None
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
    # 단일 페이지 앱의 현재 메뉴 표시: 페이지 탐색 API를 사용하지 않습니다.
    st.markdown(
        '<div aria-current="page" style="background:#FFF1EB; color:#392822; '
        'border-radius:12px; padding:14px 16px; font-weight:600; text-align:center; margin-bottom:12px;">구매 어시스턴트</div>',
        unsafe_allow_html=True,
    )
    st.button("새 대화", icon=":material/add:", type="primary", width="stretch", on_click=start_new_chat)
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
        st.caption("판매량은 모델 추정치이며 주문 계산은 국내 창고 기준입니다.")
