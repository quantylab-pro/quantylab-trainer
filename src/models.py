# -*- coding: utf-8 -*-
from typing import Optional
from sqlalchemy import BigInteger, Index, String, Text, DateTime, Integer, Float, Boolean, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from .db_types import PortableJSON as JSON, PortableVector


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    create_time: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    update_time: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class QuantylabNpcSubscriptionPayment(Base, TimestampMixin):
    __tablename__ = "quantylab_npc_subscription_payment"

    payment_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    payment_date: Mapped[str] = mapped_column(String)
    product_id: Mapped[str] = mapped_column(String)
    product_type: Mapped[str] = mapped_column(String)
    subscription_round: Mapped[int] = mapped_column(BigInteger)
    product_name: Mapped[str] = mapped_column(String)
    subscription_id: Mapped[int] = mapped_column(BigInteger)
    subscriber_id: Mapped[str] = mapped_column(String)
    product_price: Mapped[int] = mapped_column(BigInteger)
    coupon_discount: Mapped[int] = mapped_column(BigInteger, nullable=True)
    payment_amount: Mapped[int] = mapped_column(BigInteger)
    payment_status: Mapped[str] = mapped_column(String)


class QuantylabMember(Base, TimestampMixin):
    # 퀀티랩 프리미엄콘텐츠 구독자
    __tablename__ = "quantylab_member"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    talktalk_user_key: Mapped[str] = mapped_column(String, nullable=True)
    subscription_id: Mapped[int] = mapped_column(BigInteger, nullable=True)  # 처음 생성 시에는 null, 결제 번호로 구독 번호 확인 후 업데이트
    last_payment_id: Mapped[int] = mapped_column(BigInteger)
    last_payment_date: Mapped[str] = mapped_column(String, nullable=True)
    api_token: Mapped[str] = mapped_column(String, nullable=True)
    access_code: Mapped[str] = mapped_column(String, nullable=True)
    expiration_date: Mapped[str] = mapped_column(String, nullable=True)
    member_type: Mapped[str] = mapped_column(String, nullable=True)  # admin, level1, level2, level3


class QuantylabServiceConsumption(Base, TimestampMixin):
    __tablename__ = "quantylab_service_consumption"

    date: Mapped[str] = mapped_column(String, primary_key=True)
    visit_count: Mapped[int] = mapped_column(BigInteger)
    user_count: Mapped[int] = mapped_column(BigInteger)
    root_visit_count: Mapped[int] = mapped_column(BigInteger)
    root_user_count: Mapped[int] = mapped_column(BigInteger)
    ios_visit_count: Mapped[int] = mapped_column(BigInteger)
    android_visit_count: Mapped[int] = mapped_column(BigInteger)
    pc_visit_count: Mapped[int] = mapped_column(BigInteger)
    other_visit_count: Mapped[int] = mapped_column(BigInteger)
    bot_visit_count: Mapped[int] = mapped_column(BigInteger, default=0)
    page_visit_counts: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    non_service_visit_counts: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    attack_visit_counts: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    referrer_visit_counts: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    pwa_visit_count: Mapped[int] = mapped_column(BigInteger, nullable=True, default=0)
    pwa_user_count: Mapped[int] = mapped_column(BigInteger, nullable=True, default=0)


class NaverFinCommodity(Base, TimestampMixin):
    __tablename__ = "naver_fin_commodity"

    type_name: Mapped[str] = mapped_column(String, primary_key=True)
    date: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[float] = mapped_column(Float)
    diff: Mapped[float] = mapped_column(Float)
    diff_ratio: Mapped[float] = mapped_column(Float)
    extra: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class InvestingCommodity(Base, TimestampMixin):
    __tablename__ = "investing_commodity"

    type_name: Mapped[str] = mapped_column(String, primary_key=True)
    date: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[float] = mapped_column(Float)
    diff: Mapped[float] = mapped_column(Float)
    diff_ratio: Mapped[float] = mapped_column(Float)


class FedSchedule(Base, TimestampMixin):
    __tablename__ = "fed_schedule"

    date: Mapped[str] = mapped_column(String, primary_key=True)
    tag: Mapped[str] = mapped_column(String)


class KrBaseInterestRate(Base, TimestampMixin):
    # 한국 기준금리
    __tablename__ = "kr_base_interest_rate"

    date: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[float] = mapped_column(Float)


class KrYearTrade(Base, TimestampMixin):
    __tablename__ = "kr_year_trade"

    year: Mapped[str] = mapped_column(String, primary_key=True)
    trade_balance: Mapped[float] = mapped_column(Float)  # 무역수지
    import_count: Mapped[int] = mapped_column(Integer)  # 수입건수
    import_amount: Mapped[float] = mapped_column(Float)  # 수입금액
    export_count: Mapped[int] = mapped_column(Integer)  # 수출건수
    export_amount: Mapped[float] = mapped_column(Float)  # 수출금액


class KrMonthTrade(Base, TimestampMixin):
    # 월간 한국 무역
    __tablename__ = "kr_month_trade"

    month: Mapped[str] = mapped_column(String, primary_key=True)  # 월 (예: '200001')
    trade_balance: Mapped[float] = mapped_column(Float)  # 무역수지
    import_count: Mapped[int] = mapped_column(Integer)  # 수입건수
    import_amount: Mapped[float] = mapped_column(Float)  # 수입금액
    export_count: Mapped[int] = mapped_column(Integer)  # 수출건수
    export_amount: Mapped[float] = mapped_column(Float)  # 수출금액


class StockScore(Base, TimestampMixin):
    __tablename__ = "stock_score"

    model: Mapped[str] = mapped_column(String, primary_key=True)
    date: Mapped[str] = mapped_column(String, primary_key=True)
    stock_code: Mapped[str] = mapped_column(String, primary_key=True)
    score: Mapped[float] = mapped_column(Float)


class StockRank(Base, TimestampMixin):
    __tablename__ = "stock_rank"

    code: Mapped[str] = mapped_column(String, primary_key=True)
    date: Mapped[str] = mapped_column(String, primary_key=True)
    model: Mapped[str] = mapped_column(String, primary_key=True)
    stock_name: Mapped[str] = mapped_column(String)
    rank: Mapped[int] = mapped_column(Integer)
    rank_diff: Mapped[int] = mapped_column(Integer)
    stock_count: Mapped[int] = mapped_column(Integer)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)


class FomcDot(Base, TimestampMixin):
    __tablename__ = "fomc_dot"

    year: Mapped[str] = mapped_column(String, primary_key=True)
    target_rate: Mapped[float] = mapped_column(Float, primary_key=True)
    count: Mapped[float] = mapped_column(Float)


class CurrentAccount(Base, TimestampMixin):
    __tablename__ = "current_account"

    month: Mapped[str] = mapped_column(String, primary_key=True)
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class StockMarketHoliday(Base, TimestampMixin):
    __tablename__ = "stock_market_holiday"

    nation: Mapped[str] = mapped_column(String, primary_key=True)
    date: Mapped[str] = mapped_column(String, primary_key=True)
    desc: Mapped[str] = mapped_column(String)


class FutureOptionExpirationDay(Base, TimestampMixin):
    __tablename__ = "future_option_expiration_day"

    nation: Mapped[str] = mapped_column(String, primary_key=True)
    type_name: Mapped[str] = mapped_column(String, primary_key=True)
    date: Mapped[str] = mapped_column(String, primary_key=True)
    desc: Mapped[str] = mapped_column(String)


class M1(Base, TimestampMixin):
    # 광의통화 (M1)
    __tablename__ = "m1"

    month: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[float] = mapped_column(Float)  # 단위: 조원
    diff: Mapped[float] = mapped_column(Float)
    diff_ratio: Mapped[float] = mapped_column(Float)


class M2(Base, TimestampMixin):
    # 광의통화 (M2)
    __tablename__ = "m2"

    month: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[float] = mapped_column(Float)  # 단위: 조원
    diff: Mapped[float] = mapped_column(Float)
    diff_ratio: Mapped[float] = mapped_column(Float)


class FinancialLiquidity(Base, TimestampMixin):
    # 금융기관유동성(Lf = M2 + ...)
    __tablename__ = "financial_liquidity"

    month: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[float] = mapped_column(Float)  # 단위: 조원
    diff: Mapped[float] = mapped_column(Float)
    diff_ratio: Mapped[float] = mapped_column(Float)


class Liquidity(Base, TimestampMixin):
    # 광의유동성(L = Lf + ...)
    __tablename__ = "liquidity"

    month: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[float] = mapped_column(Float)  # 단위: 조원
    diff: Mapped[float] = mapped_column(Float)
    diff_ratio: Mapped[float] = mapped_column(Float)


class Corp(Base, TimestampMixin):
    # 법인정보
    __tablename__ = "corp"

    corp_code: Mapped[str] = mapped_column(String, primary_key=True)  # 법인코드
    corp_name: Mapped[str] = mapped_column(String)  # 법인명
    en_corp_name: Mapped[str] = mapped_column(String, nullable=True)  # 법인영문명
    stock_code: Mapped[str] = mapped_column(String, nullable=True)  # 종목코드


class CorpBizOverview(Base, TimestampMixin):
    # 기업 사업 개요 (DART 사업보고서 "사업의 개요" 섹션)
    __tablename__ = "corp_biz_overview"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목코드
    corp_code: Mapped[str] = mapped_column(String)  # DART 법인고유번호
    corp_name: Mapped[str] = mapped_column(String)  # 법인명
    rcept_no: Mapped[str] = mapped_column(String)  # 접수번호
    rcept_dt: Mapped[str] = mapped_column(String)  # 접수일자
    report_nm: Mapped[str] = mapped_column(String)  # 보고서명
    biz_overview: Mapped[str] = mapped_column(String)  # 사업의 개요 텍스트
    biz_segment_keywords: Mapped[str | None] = mapped_column(String, nullable=True)  # 사업부문 키워드
    biz_product_keywords: Mapped[str | None] = mapped_column(String, nullable=True)  # 주요 제품/서비스 키워드
    listed_affiliates: Mapped[str | None] = mapped_column(String, nullable=True)  # 상장 계열사/자회사/관계사 (코드:종목명, 콤마 구분)


class KrxStockCode(Base, TimestampMixin):
    # 주식 종목 코드
    __tablename__ = "krx_stock_code"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목코드
    std_code: Mapped[str] = mapped_column(String, primary_key=True)  # KRX 종목코드
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class StockCode(Base, TimestampMixin):
    # 주식 종목 코드
    __tablename__ = "stock_code"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목코드
    name: Mapped[str] = mapped_column(String, default="")  # 종목명
    market_code: Mapped[str] = mapped_column(String, default="")  # 시장코드 (1: KOSPI, 2: KOSDAQ, 3: KONEX)
    market_name: Mapped[str] = mapped_column(String, default="")  # 시장명
    status: Mapped[int] = mapped_column(Integer, default=0)  # 종목 상태 (0: 정상, 1: 거래정지, 2: 거래중단)
    control: Mapped[int] = mapped_column(Integer, default=0)  # 제어 (0: 정상, 1: 주의, 2: 경고, 3: 위험예고, 4: 위험)
    supervision: Mapped[int] = mapped_column(Integer, default=0)  # 관리종목 여부 (0: 일반종목, 1: 관리)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)  # 활성화 여부
    extra: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 추가 정보


class UsStockCode(Base, TimestampMixin):
    # 미국 주식 종목 코드
    __tablename__ = "us_stock_code"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드 (예: 'AAPL')
    name: Mapped[str] = mapped_column(String)  # 종목명 (예: 'Apple Inc.')
    sector: Mapped[str] = mapped_column(String)  # 섹터 (예: 'Technology')
    industry: Mapped[str] = mapped_column(String)  # 산업 (예: 'Consumer Electronics')
    country: Mapped[str] = mapped_column(String)  # 국가 (예: 'United States')
    ipo_year: Mapped[str] = mapped_column(String)  # 상장 연도 (예: '1980')
    market_cap: Mapped[str] = mapped_column(String)  # 시가총액 (예: '2.5T')
    close: Mapped[str] = mapped_column(String)  # 최근 거래가 (예: '$150.00')
    diff: Mapped[str] = mapped_column(String)  # 전일 대비 변화량 (예: '+1.50')
    diff_ratio: Mapped[str] = mapped_column(String)  # 전일 대비 변화율 (예: '+1.01%')
    volume: Mapped[str] = mapped_column(String)  # 거래량 (예: '100M')
    url: Mapped[str] = mapped_column(String)  # URL (예: '/market-activity/stocks/aapl')
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)  # 활성화 여부


class Bdi(Base, TimestampMixin):
    # BDI (Baltic Dry Index)
    __tablename__ = "bdi"

    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜
    value: Mapped[float] = mapped_column(Float)  # BDI 값
    diff: Mapped[float] = mapped_column(Float)  # 전일 대비 변화량
    diff_ratio: Mapped[float] = mapped_column(Float)  # 전일 대비 변화율


class KrxMarketCap(Base, TimestampMixin):
    # KRX 시가총액
    __tablename__ = "krx_market_cap"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 시가총액 데이터


class WiseReportStockYearFeature(Base, TimestampMixin):
    # WiseReport 주식 기능 연간 데이터
    # "FCF": 311068.0, "CAPEX": 536143.0, "ROA(%)": 7.85, "ROE(%)": 9.82, "period": "y", "BPS(원)": 73600.0, "EPS(원)": 6802.0, 
    # "PBR(배)": 1.21, "PER(배)": 13.09, "expected": 1, "매출액": 3690950.0, "자본금": 8963.0, "부채비율": 23.82, "부채총계": 1190282.0, 
    # "순이익률": 12.66, "영업이익": 517732.0, "자본총계": 4996100.0, "자산총계": 6186382.0, "현금DPS(원)": 1465.0, 
    # "당기순이익": 467404.0, "영업이익률": 14.03, "자본유보율": null, "이자발생부채": null, "자본총계(지배)": 4879818.0, 
    # "현금배당성향(%)": 18.93, "현금배당수익률": 1.65, "당기순이익(지배)": 458122.0, "자본총계(비지배)": null, "세전계속사업이익": 569236.0, 
    # "영업활동현금흐름": 885000.0, "재무활동현금흐름": -108886.0, "투자활동현금흐름": -605504.0, "당기순이익(비지배)": null, "발행주식수(보통주)": null, "영업이익(발표기준)": null
    __tablename__ = "wise_report_stock_year_feature"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드
    year: Mapped[str] = mapped_column(String, primary_key=True)  # 월 (예: '2015/12')
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 연간 재무 데이터


class BrokerResearchSource(Base, TimestampMixin):
    """증권사 리서치 목록 원천 설정."""
    __tablename__ = "broker_research_source"

    source_id: Mapped[str] = mapped_column(String, primary_key=True)
    broker_name: Mapped[str] = mapped_column(String)
    listing_url: Mapped[str] = mapped_column(String)
    parser: Mapped[str] = mapped_column(String, default="html_links")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    extra: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class BrokerResearchDocument(Base, TimestampMixin):
    """증권사 리서치 메타데이터와 내부 검색용 추출 텍스트."""
    __tablename__ = "broker_research_document"

    document_id: Mapped[str] = mapped_column(String, primary_key=True)
    source_id: Mapped[str] = mapped_column(String)
    broker_name: Mapped[str] = mapped_column(String)
    title: Mapped[str] = mapped_column(String)
    published_at: Mapped[str] = mapped_column(String, nullable=True)
    category: Mapped[str] = mapped_column(String, nullable=True)
    url: Mapped[str] = mapped_column(String)
    content_type: Mapped[str] = mapped_column(String, nullable=True)
    content_sha256: Mapped[str] = mapped_column(String, nullable=True)
    text_content: Mapped[str] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)


class Taylor(Base, TimestampMixin):
    # Taylor 지수
    __tablename__ = "taylor"

    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '19501001')
    value: Mapped[float] = mapped_column(Float)  # Taylor 지수 값


class StockIndex(Base, TimestampMixin):
    # 주식 지수
    __tablename__ = "stock_index"

    group_name: Mapped[str] = mapped_column(String, primary_key=True)  # 지수 그룹 (예: 'kospi')
    name: Mapped[str] = mapped_column(String, primary_key=True)  # 지수 이름 (예: '코스피')
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20150102')
    open: Mapped[float] = mapped_column(Float)  # 시가
    high: Mapped[float] = mapped_column(Float)  # 고가
    low: Mapped[float] = mapped_column(Float)  # 저가
    close: Mapped[float] = mapped_column(Float)  # 종가
    diff: Mapped[float] = mapped_column(Float)  # 전일 대비 변화량
    diff_ratio: Mapped[float] = mapped_column(Float)  # 전일 대비 변화율
    volume: Mapped[float] = mapped_column(Float)  # 거래량
    amount: Mapped[float] = mapped_column(Float)  # 거래 대금
    market_cap: Mapped[float] = mapped_column(Float)  # 시가총액


class StockEvent(Base, TimestampMixin):
    # 주식 이벤트
    __tablename__ = "stock_event"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 이벤트 날짜 (예: '20210115')
    event_tag: Mapped[str] = mapped_column(String, primary_key=True)  # 이벤트 태그 (예: '08')
    event_name: Mapped[str] = mapped_column(String)  # 이벤트 이름 (예: '주총소집에 관한 이사회결의')
    market: Mapped[str] = mapped_column(String)  # 시장 (예: 'kospi')
    name: Mapped[str] = mapped_column(String)  # 종목명


class Tips(Base, TimestampMixin):
    # TIPS (Treasury Inflation-Protected Securities)
    __tablename__ = "tips"

    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20030122')
    value: Mapped[float] = mapped_column(Float)  # TIPS 값


class StockMarketInvestor(Base, TimestampMixin):
    # 주식시장 투자자
    __tablename__ = "stock_market_investor"

    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20150119')
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 투자자 데이터


class Vix(Base, TimestampMixin):
    # VIX (Volatility Index)
    __tablename__ = "vix"

    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20210505')
    open: Mapped[float] = mapped_column(Float)  # 시가
    high: Mapped[float] = mapped_column(Float)  # 고가
    low: Mapped[float] = mapped_column(Float)  # 저가
    close: Mapped[float] = mapped_column(Float)  # 종가
    diff: Mapped[float] = mapped_column(Float)  # 전일 대비 변화량
    diff_ratio: Mapped[float] = mapped_column(Float)  # 전일 대비 변화율


class KospiVix(Base, TimestampMixin):
    # KOSPI VIX (Korea Volatility Index)
    __tablename__ = "kospi_vix"

    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20220103')
    open: Mapped[float] = mapped_column(Float)  # 시가
    high: Mapped[float] = mapped_column(Float)  # 고가
    low: Mapped[float] = mapped_column(Float)  # 저가
    close: Mapped[float] = mapped_column(Float)  # 종가
    diff: Mapped[float] = mapped_column(Float)  # 전일 대비 변화량
    diff_ratio: Mapped[float] = mapped_column(Float)  # 전일 대비 변화율


class InvestorBuySellAmount(Base, TimestampMixin):
    # 투자자 매매 금액
    __tablename__ = "investor_buy_sell_amount"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20211119')
    std_code: Mapped[str] = mapped_column(String)  # 표준 코드
    all: Mapped[float] = mapped_column(Float)  # 전체 매매 금액
    ind: Mapped[float] = mapped_column(Float)  # 기관 매매 금액
    inst: Mapped[float] = mapped_column(Float)  # 개인 매매 금액
    corp_etc: Mapped[float] = mapped_column(Float)  # 기타 법인 매매 금액
    foreign: Mapped[float] = mapped_column(Float)  # 외국인 매매 금액


class InvestorBuySell(Base, TimestampMixin):
    # 투자자 매매
    __tablename__ = "investor_buy_sell"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20200511')
    close: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 현재가
    diff_sign: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 대비기호
    diff: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 전일대비
    diff_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 등락율
    acc_volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 누적거래량
    acc_amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 누적거래대금
    ind: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 개인투자자
    frgn: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 외국인투자자
    inst: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 기관계
    fin_invest: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 금융투자
    insurance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 보험
    trust: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 투신
    etc_fin: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 기타금융
    bank: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 은행
    pension: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 연기금등
    private_fund: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 사모펀드
    nation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 국가
    etc_corp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 기타법인
    natfor: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 내외국인


class StockCredit(Base, TimestampMixin):
    # 주식 신용 거래
    __tablename__ = "stock_credit"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목코드
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 일자
    qry_tp: Mapped[str] = mapped_column(String, primary_key=True)  # 조회구분 (1:융자, 2:대주)
    close: Mapped[float] = mapped_column(Float, nullable=True)  # 현재가
    diff_sign: Mapped[str] = mapped_column(String, nullable=True)  # 전일대비기호
    diff: Mapped[float] = mapped_column(Float, nullable=True)  # 전일대비
    volume: Mapped[int] = mapped_column(BigInteger, nullable=True)  # 거래량
    new: Mapped[float] = mapped_column(Float, nullable=True)  # 신규
    repayment: Mapped[float] = mapped_column(Float, nullable=True)  # 상환
    balance: Mapped[float] = mapped_column(Float, nullable=True)  # 잔고
    amount: Mapped[float] = mapped_column(Float, nullable=True)  # 금액
    balance_diff: Mapped[float] = mapped_column(Float, nullable=True)  # 대비
    supply_ratio: Mapped[float] = mapped_column(Float, nullable=True)  # 공여율
    balance_ratio: Mapped[float] = mapped_column(Float, nullable=True)  # 잔고율


class StockMarketMinuteCandle(Base, TimestampMixin):
    # 주식시장 분봉 캔들
    __tablename__ = "stock_market_minute_candle"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 시장 코드 (예: 'kospi')
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20171220')
    time: Mapped[str] = mapped_column(String, primary_key=True)  # 시간 (HHMM 형식)
    close: Mapped[float] = mapped_column(Float)  # 종가
    diff: Mapped[float] = mapped_column(Float)  # 전일 대비 변화량
    diff_ratio: Mapped[float] = mapped_column(Float)  # 전일 대비 변화율
    volume: Mapped[float] = mapped_column(Float)  # 거래량
    amount: Mapped[float] = mapped_column(Float)  # 거래 대금


class StockMinuteCandle(Base, TimestampMixin):
    # 주식 분봉 캔들
    __tablename__ = "stock_minute_candle"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20220301')
    time: Mapped[str] = mapped_column(String, primary_key=True)  # 시간 (HHMM 형식)
    open: Mapped[float] = mapped_column(Float)  # 시가
    high: Mapped[float] = mapped_column(Float)  # 고가
    low: Mapped[float] = mapped_column(Float)  # 저가
    close: Mapped[float] = mapped_column(Float)  # 종가
    diff: Mapped[float] = mapped_column(Float, nullable=True)  # 전일 대비 변화량
    diff_ratio: Mapped[float] = mapped_column(Float, nullable=True)  # 전일 대비 변화율
    volume: Mapped[float] = mapped_column(Float)  # 거래량
    amount: Mapped[float] = mapped_column(Float)  # 거래 대금


class StockNxtPrice(Base, TimestampMixin):
    # NXT 주식 시세 스냅샷
    __tablename__ = "stock_nxt_price"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20260625')
    time: Mapped[str] = mapped_column(String, primary_key=True)  # 시간 (HHMM 형식)
    price: Mapped[float] = mapped_column(Float)  # 현재가
    diff: Mapped[float] = mapped_column(Float, nullable=True)  # 전일 대비
    diff_ratio: Mapped[float] = mapped_column(Float, nullable=True)  # 전일 대비 변화율 (ratio)
    volume: Mapped[float] = mapped_column(Float, nullable=True)  # 누적 거래량
    amount: Mapped[float] = mapped_column(Float, nullable=True)  # 누적 거래대금
    market_session: Mapped[str] = mapped_column(String, default="")  # pre/main/after
    extra: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


Index("idx_stock_nxt_price_date_time", StockNxtPrice.date, StockNxtPrice.time)
    

class StockDayCandle(Base, TimestampMixin):
    # 주식 일봉 캔들
    __tablename__ = "stock_day_candle"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20220103')
    open: Mapped[float] = mapped_column(Float)  # 시가
    high: Mapped[float] = mapped_column(Float)  # 고가
    low: Mapped[float] = mapped_column(Float)  # 저가
    close: Mapped[float] = mapped_column(Float)  # 종가
    diff: Mapped[float] = mapped_column(Float)  # 전일 대비 변화량
    diff_ratio: Mapped[float] = mapped_column(Float)  # 전일 대비 변화율
    diff_sign: Mapped[str] = mapped_column(String)  # 변화 기호 (예: '0', '5', '1' 등)
    volume: Mapped[float] = mapped_column(Float)  # 거래량
    amount: Mapped[float] = mapped_column(Float)  # 거래 대금
    

class KrMarketCap(Base, TimestampMixin):
    # 한국 시장 시가총액
    __tablename__ = "kr_market_cap"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20220101')
    name: Mapped[str] = mapped_column(String)  # 종목명
    close: Mapped[float] = mapped_column(Float)  # 종가
    diff: Mapped[float] = mapped_column(Float)  # 전일 대비 변화량
    diff_ratio: Mapped[float] = mapped_column(Float)  # 전일 대비 변화율
    volume: Mapped[float] = mapped_column(Float)  # 거래량
    market: Mapped[str] = mapped_column(String)  # 시장 (예: 'KOSPI', 'KOSDAQ')
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 시가총액 데이터


class KrBond3YearCandle(Base, TimestampMixin):
    # 한국 국채 3년물 캔들
    __tablename__ = "kr_bond_3_year_candle"

    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20190920')
    open: Mapped[float] = mapped_column(Float)  # 시가
    high: Mapped[float] = mapped_column(Float)  # 고가
    low: Mapped[float] = mapped_column(Float)  # 저가
    close: Mapped[float] = mapped_column(Float)  # 종가
    volume: Mapped[float] = mapped_column(Float)  # 거래량


class CitiSurprise(Base, TimestampMixin):
    # 시티 서프라이즈 지수
    __tablename__ = "citi_surprise"
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20220101')
    group_name: Mapped[str] = mapped_column(String, primary_key=True)  # 그룹 이름 (예: 'global', 'us', 'eurozone' 등)
    value: Mapped[float] = mapped_column(Float)  # 서프라이즈 지수 값


class CnnFearGreed(Base, TimestampMixin):
    # CNN Fear & Greed 지수
    __tablename__ = "cnn_fear_greed"
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20220101')
    value: Mapped[float] = mapped_column(Float)  # 서프라이즈 지수 값


class KrxPerPbr(Base, TimestampMixin):
    # KRX PER/PBR 데이터
    __tablename__ = "krx_per_pbr"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20150105')
    name: Mapped[str] = mapped_column(String)  # 종목명
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # PER/PBR 데이터


class OecdCliG20(Base, TimestampMixin):
    # OECD Composite leading indicator (CLI) G20
    __tablename__ = "oecd_cli_g20"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # G20 국가 코드
    month: Mapped[str] = mapped_column(String, primary_key=True)  # 월 (예: '2008-01')
    value: Mapped[float] = mapped_column(Float)  # CLI 값


class CompGuideStockFeature(Base, TimestampMixin):
    # WiseReport 컴퍼니 가이드 주식 피처
    __tablename__ = "comp_guide_stock_feature"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드
    month: Mapped[str] = mapped_column(String, primary_key=True)  # 월 (예: '2015/12')
    name: Mapped[str] = mapped_column(String)  # 종목명
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 주식 피처 데이터


class CompGuideCompanyInfoSnapshot(Base, TimestampMixin):
    # WiseReport 컴퍼니 가이드 Snapshot
    __tablename__ = "comp_guide_company_info_snapshot"

    code: Mapped[str] = mapped_column(String, primary_key=True)
    date: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    raw_html: Mapped[str] = mapped_column(String)


class CompGuideCompanyInfoSnapshotFeature(Base, TimestampMixin):
    # WiseReport 컴퍼니 가이드 Snapshot Feature
    __tablename__ = "comp_guide_company_info_snapshot_feature"

    code: Mapped[str] = mapped_column(String, primary_key=True)
    date: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # Snapshot Feature 데이터


class StockMarketDayCandle(Base, TimestampMixin):
    # 주식시장 일봉 캔들
    __tablename__ = "stock_market_day_candle"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 시장 코드 (예: 'kospi')
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20171220')
    open: Mapped[float] = mapped_column(Float)  # 시가
    high: Mapped[float] = mapped_column(Float)  # 고가
    low: Mapped[float] = mapped_column(Float)  # 저가
    close: Mapped[float] = mapped_column(Float)  # 종가
    diff: Mapped[float] = mapped_column(Float)  # 전일 대비 변화량
    diff_ratio: Mapped[float] = mapped_column(Float)  # 전일 대비 변화율
    diff_sign: Mapped[str] = mapped_column(String)  # 변화 기호 (예: '0', '5', '1' 등)
    volume: Mapped[float] = mapped_column(Float)  # 거래량
    amount: Mapped[float] = mapped_column(Float)  # 거래 대금


class ForeignStockMarketDayCandle(Base, TimestampMixin):
    # 주식시장 일봉 캔들
    __tablename__ = "foreign_stock_market_day_candle"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 시장 코드 (예: 'kospi')
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20171220')
    open: Mapped[float] = mapped_column(Float)  # 시가
    high: Mapped[float] = mapped_column(Float)  # 고가
    low: Mapped[float] = mapped_column(Float)  # 저가
    close: Mapped[float] = mapped_column(Float)  # 종가
    diff: Mapped[float] = mapped_column(Float)  # 전일 대비 변화량
    diff_ratio: Mapped[float] = mapped_column(Float)  # 전일 대비 변화율
    volume: Mapped[float] = mapped_column(Float)  # 거래량


class MarketDayFeature(Base, TimestampMixin):
    # 시장 지표 및 특징
    __tablename__ = "market_day_feature"

    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20210901')
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 시장 지표 데이터


class UsFedInterestRate(Base, TimestampMixin):
    # 미국 기준금리
    __tablename__ = "us_fed_interest_rate"

    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20220101')
    name: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[float] = mapped_column(Float)  # 기준금리 값
    

class UsFredTargetInterestRate(Base, TimestampMixin):
    # 미국 연준 목표 금리
    __tablename__ = "us_fred_target_interest_rate"

    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20220101')
    value: Mapped[float] = mapped_column(Float)  # 목표 금리 값


class KrPriceIndex(Base, TimestampMixin):
    # 한국 물가 지수
    # content: {"- 집세": -0.2, "근원물가": 2.6, "생활물가": 3.4, "- 공업제품": 1.8, "소비자물가": 2.8, "- 개인서비스": 3.5, "- 공공서비스": 2.2, "- 농축수산물": 8.0}
    __tablename__ = "kr_price_index"

    month: Mapped[str] = mapped_column(String, primary_key=True)  # 월 (예: '202201')
    type_name: Mapped[str] = mapped_column(String, primary_key=True)  # 지수 종류 (예: 'cpi', 'ppi')
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 물가 지수 데이터


class StockMarketCapital(Base, TimestampMixin):
    # 주식시장 자본
    __tablename__ = "stock_market_capital"

    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20151216')
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 자본 데이터


class FutureDayCandle(Base, TimestampMixin):
    # 선물 일봉 캔들
    __tablename__ = "future_day_candle"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 선물 코드 (예: 'kospi200f')
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20200604')
    open: Mapped[float] = mapped_column(Float)  # 시가
    high: Mapped[float] = mapped_column(Float)  # 고가
    low: Mapped[float] = mapped_column(Float)  # 저가
    close: Mapped[float] = mapped_column(Float)  # 종가
    diff: Mapped[float] = mapped_column(Float)  # 전일 대비 변화량
    diff_ratio: Mapped[float] = mapped_column(Float)  # 전일 대비 변화율
    volume: Mapped[float] = mapped_column(Float)  # 거래량


class StockSubscription(Base, TimestampMixin):
    # 공모주 청약 정보
    __tablename__ = "stock_subscription"

    name: Mapped[str] = mapped_column(String, primary_key=True)  # 종목명
    start_date: Mapped[str] = mapped_column(String, primary_key=True)  # 청약 시작일 (예: '20210121')
    end_date: Mapped[str] = mapped_column(String, primary_key=True)  # 청약 종료일 (예: '20210122')
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 청약 정보 데이터


class UsBond10YearYield(Base, TimestampMixin):
    # 미국 국채 10년물 수익률
    __tablename__ = "us_bond_10_year_yield"

    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20151231')
    open: Mapped[float] = mapped_column(Float)  # 시가
    high: Mapped[float] = mapped_column(Float)  # 고가
    low: Mapped[float] = mapped_column(Float)  # 저가
    close: Mapped[float] = mapped_column(Float)  # 종가
    diff: Mapped[float] = mapped_column(Float)  # 전일 대비 변화량
    diff_ratio: Mapped[float] = mapped_column(Float)  # 전일 대비 변화율


class ShortBalance(Base, TimestampMixin):
    # 공매도 잔고
    __tablename__ = "short_balance"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20221205')
    name: Mapped[str] = mapped_column(String)  # 종목명
    bal_qty: Mapped[float] = mapped_column(Float)  # 공매도 잔고 수량
    bal_amt: Mapped[float] = mapped_column(Float)  # 공매도 잔고 금액
    bal_rto: Mapped[float] = mapped_column(Float)  # 공매도 잔고 비율
    shares: Mapped[float] = mapped_column(Float)  # 발행 주식 수
    market_cap: Mapped[float] = mapped_column(Float)  # 시가총액


class StockIndexPerPbr(Base, TimestampMixin):
    # 주식 지수 PER/PBR
    __tablename__ = "stock_index_per_pbr"

    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20220103')
    name: Mapped[str] = mapped_column(String, primary_key=True)  # 지수 이름 (예: '코스피')
    group_name: Mapped[str] = mapped_column(String)  # 지수 그룹 (예: 'krx')
    per: Mapped[float] = mapped_column(Float, nullable=True)  # PER (주가수익비율)
    pbr: Mapped[float] = mapped_column(Float, nullable=True)  # PBR (주가순자산비율)
    dyr: Mapped[float] = mapped_column(Float, nullable=True)  # 배당수익률
    extra: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 추가 정보


class Gdp(Base, TimestampMixin):
    # 국내총생산 (GDP)
    __tablename__ = "gdp"

    country: Mapped[str] = mapped_column(String, primary_key=True)  # 국가 (예: 'Korea')
    unit: Mapped[str] = mapped_column(String, primary_key=True)  # 단위 (예: 'National currency')
    year: Mapped[str] = mapped_column(String, primary_key=True)  # 연도 (예: '1981')
    value: Mapped[float] = mapped_column(Float)  # GDP 값
    scale: Mapped[str] = mapped_column(String)  # 단위 (예: 'Billions')
    extra: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 추가 정보


class ImfWeo(Base, TimestampMixin):
    # IMF World Economic Outlook
    __tablename__ = "imf_weo"

    country: Mapped[str] = mapped_column(String, primary_key=True)  # 국가 (예: 'US')
    indicator: Mapped[str] = mapped_column(String, primary_key=True)  # 지표 (예: 'GDP')
    year: Mapped[str] = mapped_column(String, primary_key=True)  # 연도 (예: '1981')
    value: Mapped[float] = mapped_column(Float)  # GDP 값
    scale: Mapped[str] = mapped_column(String, nullable=True)  # 
    unit: Mapped[str] = mapped_column(String, nullable=True)  # 단위 (예: 'USD')
    extra: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 추가 정보


class FsBat(Base, TimestampMixin):
    # 재무제표 (배치)
    __tablename__ = "fs_bat"

    corp_code: Mapped[str] = mapped_column(String, primary_key=True)  # 기업 코드
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 결산 기준일 (예: '2017-12-31')
    report_type: Mapped[str] = mapped_column(String, primary_key=True)  # 보고서 종류 (예: '사업보고서')
    fs_type: Mapped[str] = mapped_column(String, primary_key=True)  # 재무제표 종류 (예: '재무상태표')
    corp_name: Mapped[str] = mapped_column(String)  # 기업명
    code: Mapped[str] = mapped_column(String, nullable=True)  # 종목 코드
    content: Mapped[list] = mapped_column(JSON, nullable=False, default=list)  # 재무 데이터 리스트
    file_date: Mapped[str] = mapped_column(String)  # 파일 날짜 (예: '20241115')
    file_name: Mapped[str] = mapped_column(String)  # 파일 이름 (예: '2017_사업보고서_01_재무상태표_연결_20241115.txt')
    month: Mapped[str] = mapped_column(String)  # 결산 월 (예: 12)
    extra: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 추가 정보


class Fs(Base, TimestampMixin):
    # 재무제표
    __tablename__ = "fs"

    rcept_no: Mapped[str] = mapped_column(String, primary_key=True)  # 접수 번호
    reprt_code: Mapped[str] = mapped_column(String, primary_key=True)  # 보고서 코드
    bsns_year: Mapped[str] = mapped_column(String, primary_key=True)  # 사업 연도
    corp_code: Mapped[str] = mapped_column(String, primary_key=True)  # 기업 코드
    stock_code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드
    fs_div: Mapped[str] = mapped_column(String, primary_key=True)  # 재무제표 구분 (연결 여부)
    fs_nm: Mapped[str] = mapped_column(String, primary_key=True)  # 재무제표 이름
    sj_div: Mapped[str] = mapped_column(String, primary_key=True)  # 재무제표 구분 (재무상태표, 손익계산서, 현금흐름표)
    sj_nm: Mapped[str] = mapped_column(String, primary_key=True)  # 재무제표 이름
    account_nm: Mapped[str] = mapped_column(String, primary_key=True)  # 계정명
    thstrm_nm: Mapped[str] = mapped_column(String)  # 당기명
    thstrm_dt: Mapped[str] = mapped_column(String)  # 당기 날짜
    thstrm_amount: Mapped[float] = mapped_column(Float)  # 당기 금액
    frmtrm_nm: Mapped[str] = mapped_column(String)  # 전기명
    frmtrm_dt: Mapped[str] = mapped_column(String)  # 전기 날짜
    frmtrm_amount: Mapped[float] = mapped_column(Float)  # 전기 금액
    bfefrmtrm_nm: Mapped[str] = mapped_column(String)  # 전전기명
    bfefrmtrm_dt: Mapped[str] = mapped_column(String)  # 전전기 날짜
    bfefrmtrm_amount: Mapped[float] = mapped_column(Float)  # 전전기 금액
    currency: Mapped[str] = mapped_column(String)  # 통화
    reprt_name: Mapped[str] = mapped_column(String)  # 보고서 이름


class StockDemScore(Base, TimestampMixin):
    # 주식 수요 점수
    __tablename__ = "stock_dem_score"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: 20181226)
    unit: Mapped[str] = mapped_column(String, primary_key=True)  # 단위 (예: 'day', 'min')
    window: Mapped[int] = mapped_column(Integer, primary_key=True)  # 윈도우 크기 (예: 240)
    score: Mapped[float] = mapped_column(Float)  # 점수


class UsConsensus(Base, TimestampMixin):
    # 미국 주식 컨센서스
    __tablename__ = "us_consensus"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드 (예: 'AAPL')
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20241002')
    current: Mapped[float] = mapped_column(Float)  # 현재가
    high: Mapped[float] = mapped_column(Float)  # 고가
    low: Mapped[float] = mapped_column(Float)  # 저가
    mean: Mapped[float] = mapped_column(Float)  # 평균가
    median: Mapped[float] = mapped_column(Float)  # 중앙값


class BacktestChart(Base, TimestampMixin):
    # 백테스트 차트
    __tablename__ = "backtest_chart"

    model: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)  # 모델 이름 (예: 'quantylab')
    start_date: Mapped[str] = mapped_column(String, primary_key=True)  # 시작 날짜 (예: '20200101')
    end_date: Mapped[str] = mapped_column(String, primary_key=True)  # 종료 날짜 (예: '20201231')
    kospi_baseline: Mapped[list] = mapped_column(JSON, nullable=False, default=list)  # KOSPI 기준선
    kosdaq_baseline: Mapped[list] = mapped_column(JSON, nullable=False, default=list)  # KOSDAQ 기준선
    histories: Mapped[list] = mapped_column(JSON, nullable=False, default=list)  # 백테스트 히스토리 데이터
    holdings: Mapped[list] = mapped_column(JSON, nullable=False, default=list)  # 보유 종목 데이터
    reports: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 백테스트 리포트
    profit: Mapped[float] = mapped_column(Float, default=0.0)  # 총 수익률
    mdd: Mapped[float] = mapped_column(Float, default=0.0)  # 최대 낙폭 (MDD)
    mdd_start_index: Mapped[str] = mapped_column(String, nullable=True)  # MDD 시작 날짜
    mdd_end_index: Mapped[str] = mapped_column(String, nullable=True)  # MDD 종료 날짜
    std: Mapped[float] = mapped_column(Float, default=0.0)  # 표준편차
    

class Sox(Base, TimestampMixin):
    # SOX (Philadelphia Semiconductor Index)
    __tablename__ = "sox"

    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20210728')
    open: Mapped[float] = mapped_column(Float)  # 시가
    high: Mapped[float] = mapped_column(Float)  # 고가
    low: Mapped[float] = mapped_column(Float)  # 저가
    close: Mapped[float] = mapped_column(Float)  # 종가
    diff: Mapped[float] = mapped_column(Float)  # 전일 대비 변화량
    diff_ratio: Mapped[float] = mapped_column(Float)  # 전일 대비 변화율
    volume: Mapped[float] = mapped_column(Float, default=0.0)  # 거래량


class MsciIndex(Base, TimestampMixin):
    # MSCI (Morgan Stanley Capital International)
    __tablename__ = "msci_index"

    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20100121')
    name: Mapped[str] = mapped_column(String, primary_key=True)  # 지수 이름 (예: 'WORLD')
    value: Mapped[float] = mapped_column(Float)  # 지수 값
    diff: Mapped[float] = mapped_column(Float)  # 전일 대비 변화량
    diff_ratio: Mapped[float] = mapped_column(Float)  # 전일 대비 변화율


class UsStockDayCandle(Base, TimestampMixin):
    # 외국 주식 일봉 캔들
    __tablename__ = "us_stock_day_candle"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드 (예: 'AAPL')
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20220103')
    open: Mapped[float] = mapped_column(Float)  # 시가
    high: Mapped[float] = mapped_column(Float)  # 고가
    low: Mapped[float] = mapped_column(Float)  # 저가
    close: Mapped[float] = mapped_column(Float)  # 종가
    diff: Mapped[float] = mapped_column(Float)  # 전일 대비 변화량
    diff_ratio: Mapped[float] = mapped_column(Float)  # 전일 대비 변화율
    volume: Mapped[float] = mapped_column(Float)  # 거래량


class StockIndexComposition(Base, TimestampMixin):
    # 주식 지수 구성 종목
    __tablename__ = "stock_index_composition"

    code_name: Mapped[str] = mapped_column(String, primary_key=True)  # 지수 코드명 (예: 'KRX 300')
    composition: Mapped[list] = mapped_column(JSON, nullable=False, default=list)  # 구성 종목 리스트
    group_name: Mapped[str] = mapped_column(String)  # 지수 그룹 (예: 'krx')
    full_code: Mapped[str] = mapped_column(String)  # 전체 코드 (예: '5')
    short_code: Mapped[str] = mapped_column(String)  # 짧은 코드 (예: '300')
    market_code: Mapped[str] = mapped_column(String)  # 시장 코드 (예: 'KRX')
    market_name: Mapped[str] = mapped_column(String)  # 시장 이름 (예: 'KRX')


class KrBond3YearYield(Base, TimestampMixin):
    # 한국 국채 3년물 수익률
    __tablename__ = "kr_bond_3_year_yield"

    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20230217')
    open: Mapped[float] = mapped_column(Float)  # 시가
    high: Mapped[float] = mapped_column(Float)  # 고가
    low: Mapped[float] = mapped_column(Float)  # 저가
    close: Mapped[float] = mapped_column(Float)  # 종가
    diff: Mapped[float] = mapped_column(Float)  # 전일 대비 변화량
    diff_ratio: Mapped[float] = mapped_column(Float)  # 전일 대비 변화율


class StockBid(Base, TimestampMixin):
    # 주식 호가 잔량
    __tablename__ = "stock_bid"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드 (예: '002360')
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20220729')
    time: Mapped[str] = mapped_column(String, primary_key=True)  # 시간 (HHMM 형식)
    content: Mapped[list] = mapped_column(JSON, nullable=False, default=list)  # 호가 잔량 데이터 리스트
    extra: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    

class HouseholdLoan(Base, TimestampMixin):
    # 가계대출
    __tablename__ = "household_loan"

    month: Mapped[str] = mapped_column(String, primary_key=True)  # 월 (예: '200210')
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 가계대출 데이터
    

class Dx(Base, TimestampMixin):
    # DX (DAX Index)
    __tablename__ = "dx"

    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20210726')
    open: Mapped[float] = mapped_column(Float)  # 시가
    high: Mapped[float] = mapped_column(Float)  # 고가
    low: Mapped[float] = mapped_column(Float)  # 저가
    close: Mapped[float] = mapped_column(Float)  # 종가
    diff: Mapped[float] = mapped_column(Float)  # 전일 대비 변화량
    diff_ratio: Mapped[float] = mapped_column(Float)  # 전일 대비 변화율
    volume: Mapped[float] = mapped_column(Float)  # 거래량


class MarketScore(Base, TimestampMixin):
    # 시장 점수
    __tablename__ = "market_score"

    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20150106')
    model: Mapped[str] = mapped_column(String, primary_key=True)  # 모델 버전 (예: 'qrdm')
    score: Mapped[float] = mapped_column(Float)  # 점수
    extra: Mapped[dict] = mapped_column(JSON, nullable=True, default=dict)  # 추가 데이터


class Ipo(Base, TimestampMixin):
    # 공모주 정보
    __tablename__ = "ipo"

    corp_name: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)  # 기업명
    date: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)  # 신규 상장일 (예: '20210611')
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 공모주 데이터


class BokSchedule(Base, TimestampMixin):
    # 한국은행행 스케줄
    __tablename__ = "bok_schedule"

    year: Mapped[str] = mapped_column(String, primary_key=True)  # 연도 (예: '2020')
    schedule: Mapped[list] = mapped_column(JSON, nullable=False, default=list)  # 일정 리스트
    d_day: Mapped[list] = mapped_column(JSON, nullable=False, default=list)  # D-day 리스트


class StockFeature(Base, TimestampMixin):
    # 주식 피처
    __tablename__ = "stock_feature"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20150202')
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 피처 데이터


class Posting(Base, TimestampMixin):
    # 공시 정보
    __tablename__ = "posting"

    report_id: Mapped[str] = mapped_column(String, primary_key=True)  # 보고서 ID
    corp_code: Mapped[str] = mapped_column(String)  # 기업 코드
    date: Mapped[str] = mapped_column(String)  # 날짜 (예: '20150102')
    link: Mapped[str] = mapped_column(String)  # 공시 링크
    market: Mapped[str] = mapped_column(String)  # 시장 (예: 'kospi')
    extra: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 공시 데이터


class KindEarningsAnnouncement(Base, TimestampMixin):
    """KIND의 잠정 영업실적 공시 메타데이터.

    공시 원문은 KIND가 제공하므로 수집본에는 조회·표시를 위한 식별자와
    메타데이터만 보관한다.
    """
    __tablename__ = "kind_earnings_announcement"

    disclosure_id: Mapped[str] = mapped_column(String, primary_key=True)
    code: Mapped[str] = mapped_column(String, nullable=False)
    company_name: Mapped[str] = mapped_column(String, nullable=False)
    market: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    disclosed_at: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    submitter: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    is_corrected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    kind_url: Mapped[str] = mapped_column(String, nullable=False)
    extra: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    __table_args__ = (
        Index("ix_kind_earnings_announcement_code_disclosed_at", "code", "disclosed_at"),
    )


class Gsci(Base, TimestampMixin):
    # GSCI (Goldman Sachs Commodity Index)
    __tablename__ = "gsci"

    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20210512')
    open: Mapped[float] = mapped_column(Float)  # 시가
    high: Mapped[float] = mapped_column(Float)  # 고가
    low: Mapped[float] = mapped_column(Float)  # 저가
    close: Mapped[float] = mapped_column(Float)  # 종가
    diff: Mapped[float] = mapped_column(Float)  # 변화량
    diff_ratio: Mapped[float] = mapped_column(Float)  # 변화율


class FamaLsv(Base, TimestampMixin):
    # Fama-French 5-factor model
    __tablename__ = "fama_lsv"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드
    month: Mapped[str] = mapped_column(String, primary_key=True)  # 월 (예: '2017/12')
    name: Mapped[str] = mapped_column(String, nullable=True)  # 종목명
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # Fama-French 데이터


class UsPriceIndex(Base, TimestampMixin):
    # 미국 물가 지수
    __tablename__ = "us_price_index"

    type_name: Mapped[str] = mapped_column(String, primary_key=True)  # 지수 종류 (예: 'cpi')
    month: Mapped[str] = mapped_column(String, primary_key=True)  # 월 (예: '202108')
    value: Mapped[float] = mapped_column(Float)  # 지수 값
    footnote: Mapped[str] = mapped_column(String, nullable=True)  # 주석


class Scfi(Base, TimestampMixin):
    # SCFI (Shanghai Containerized Freight Index)
    __tablename__ = "scfi"

    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20141106')
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # SCFI 데이터


class StockAcqDispStatus(Base, TimestampMixin):
    # 주식 취득 및 처분 상태
    __tablename__ = "stock_acq_disp_status"

    rcept_no: Mapped[str] = mapped_column(String, primary_key=True)  # 접수 번호
    code: Mapped[str] = mapped_column(String)  # 종목 코드
    corp_code: Mapped[str] = mapped_column(String)  # 기업 코드
    corp_name: Mapped[str] = mapped_column(String)  # 기업명
    reprt_code: Mapped[str] = mapped_column(String)  # 보고서 코드
    reprt_name: Mapped[str] = mapped_column(String)  # 보고서 이름
    acqs_mth1: Mapped[str] = mapped_column(String)  # 취득 방법 1
    acqs_mth2: Mapped[str] = mapped_column(String)  # 취득 방법 2
    acqs_mth3: Mapped[str] = mapped_column(String)  # 취득 방법 3
    bsis_qy: Mapped[str] = mapped_column(String)  # 기준 수량
    change_qy_acqs: Mapped[str] = mapped_column(String)  # 취득 변경 수량
    change_qy_dsps: Mapped[str] = mapped_column(String)  # 처분 변경 수량
    change_qy_incnr: Mapped[str] = mapped_column(String)  # 증가 수량
    stock_knd: Mapped[str] = mapped_column(String)  # 주식 종류
    trmend_qy: Mapped[str] = mapped_column(String)  # 기말 수량
    year: Mapped[int] = mapped_column(Integer)  # 연도
    rm: Mapped[str] = mapped_column(String, nullable=True)


class Fx(Base, TimestampMixin):
    # 외환 환율 정보
    __tablename__ = "fx"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 외환 코드 (예: 'FX_USDKRW')
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20190923')
    base: Mapped[float] = mapped_column(Float)  # 기준 환율
    diff: Mapped[float] = mapped_column(Float)  # 전일 대비 변화량
    diff_ratio: Mapped[float] = mapped_column(Float)  # 전일 대비 변화율
    buy_cach: Mapped[float] = mapped_column(Float, nullable=True)  # 현찰 매입 환율
    sell_cach: Mapped[float] = mapped_column(Float, nullable=True)  # 현찰 매도 환율
    receive_trans: Mapped[float] = mapped_column(Float, nullable=True)  # 수출 환율
    send_trans: Mapped[float] = mapped_column(Float, nullable=True)  # 수입 환율


class Hai(Base, TimestampMixin):
    # HAI (household activity index)
    __tablename__ = "hai"

    month: Mapped[str] = mapped_column(String, primary_key=True)  # 월 (예: '202004')
    value: Mapped[float] = mapped_column(Float)  # HAI 값


class DayOvertimeUni(Base, TimestampMixin):
    # 시간외 단일가
    __tablename__ = "day_overtime_uni"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드 (예: '082740')
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: 20221124)
    open: Mapped[int] = mapped_column(Integer)  # 시가
    high: Mapped[int] = mapped_column(Integer)  # 고가
    low: Mapped[int] = mapped_column(Integer)  # 저가
    close: Mapped[int] = mapped_column(Integer)  # 종가
    diff: Mapped[int] = mapped_column(Integer)  # 전일 대비 변화량
    diff_ratio: Mapped[float] = mapped_column(Float)  # 전일 대비 변화율
    volume: Mapped[int] = mapped_column(Integer)  # 거래량
    sign: Mapped[int] = mapped_column(Integer)  # 사인 값


class DartFsXbrlTaxonomy(Base, TimestampMixin):
    # sj_div	재무제표구분	재무제표구분
    # account_id	계정ID	계정 고유명칭
    # account_nm	계정명	계정명
    # bsns_de	기준일	적용 기준일
    # label_kor	한글 출력명	한글 출력명
    # label_eng	영문 출력명	영문 출력명
    # data_tp	데이터 유형	※ 데이타 유형설명 - text block : 제목 - Text : Text - yyyy-mm-dd : Date - X : Monetary Value - (X): Monetary Value(Negative) - X.XX : Decimalized Value - Shares : Number of shares (주식 수) - For each : 공시된 항목이 전후로 반복적으로 공시될 경우 사용 - 공란 : 입력 필요 없음
    # ifrs_ref	IFRS Reference	IFRS Reference
    # ※ 출력예시
    # K-IFRS 1001 문단 54 (9),K-IFRS 1007 문단 45
    __tablename__ = "dart_fs_xbrl_taxonomy"

    sj_div: Mapped[str] = mapped_column(String, primary_key=True)  # 재무제표구분
    account_id: Mapped[str] = mapped_column(String, primary_key=True)  # 계정ID
    account_nm: Mapped[str] = mapped_column(String)  # 계정명
    bsns_de: Mapped[str] = mapped_column(String, primary_key=True)  # 기준일
    label_kor: Mapped[str] = mapped_column(String)  # 한글 출력명
    label_eng: Mapped[str] = mapped_column(String)  # 영문 출력명
    data_tp: Mapped[str] = mapped_column(String, nullable=True)  # 데이터 유형
    ifrs_ref: Mapped[str] = mapped_column(String, nullable=True)  # IFRS Reference


class DartReport(Base, TimestampMixin):
    # DART 보고서 정보
    __tablename__ = "dart_report"

    corp_cls: Mapped[str] = mapped_column(String)  # 기업 구분 (예: 'Y')
    corp_code: Mapped[str] = mapped_column(String)  # 기업 코드 (예: '00126380')
    corp_name: Mapped[str] = mapped_column(String)  # 기업명 (예: '삼성전자')
    flr_nm: Mapped[str] = mapped_column(String)  # 보고서 작성자 이름
    rcept_dt: Mapped[str] = mapped_column(String)  # 접수 날짜 (예: '20201118')
    rcept_no: Mapped[str] = mapped_column(String, primary_key=True)  # 접수 번호 (예: '20201118000053')
    report_nm: Mapped[str] = mapped_column(String)  # 보고서 이름 (예: '임원ㆍ주요주주특정증권등소유상황보고서')
    rm: Mapped[str] = mapped_column(String, nullable=True)  # 비고
    stock_code: Mapped[str] = mapped_column(String)  # 주식 코드 (예: '005930')


class DayEtfCandle(Base, TimestampMixin):
    # ETF 일봉 캔들
    __tablename__ = "day_etf_candle"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # ETF 코드 (예: '069500')
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20150102')
    open: Mapped[float] = mapped_column(Float)  # 시가
    high: Mapped[float] = mapped_column(Float)  # 고가
    low: Mapped[float] = mapped_column(Float)  # 저가
    close: Mapped[float] = mapped_column(Float)  # 종가
    diff: Mapped[float] = mapped_column(Float)  # 전일 대비 변화량
    diff_ratio: Mapped[float] = mapped_column(Float)  # 전일 대비 변화율
    diff_sign: Mapped[str] = mapped_column(String)  # 변화 부호 ('1': 상승, '2': 하락)
    volume: Mapped[int] = mapped_column(BigInteger)  # 거래량
    price: Mapped[int] = mapped_column(BigInteger)  # 거래 금액


class StockReclamation(Base, TimestampMixin):
    # 주식 청구 정보
    __tablename__ = "stock_reclamation"

    corp_name: Mapped[str] = mapped_column(String, primary_key=True)  # 기업명 (예: '크래프톤(유가)')
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 청구일 (예: '20210525')
    status: Mapped[str] = mapped_column(String, nullable=True)  # 상태 (예: '상장 예정')
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 청구 데이터


class ProgramVolume(Base, TimestampMixin):
    # 프로그램 거래량
    __tablename__ = "program_volume"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드 (예: '082740')
    date: Mapped[int] = mapped_column(Integer, primary_key=True)  # 날짜 (예: 20220224)
    current_price: Mapped[int] = mapped_column(Integer)  # 현재가
    prev_close: Mapped[int] = mapped_column(Integer)  # 전일 종가
    diff: Mapped[int] = mapped_column(Integer)  # 전일 대비 변화량
    volume: Mapped[int] = mapped_column(Integer)  # 거래량
    buy_amount: Mapped[int] = mapped_column(Integer)  # 매수 금액 (단위: 만원)
    sell_amount: Mapped[int] = mapped_column(Integer)  # 매도 금액 (단위: 만원)
    buy_qty: Mapped[int] = mapped_column(Integer)  # 매수량
    sell_qty: Mapped[int] = mapped_column(Integer)  # 매도량
    net_buy_amount: Mapped[int] = mapped_column(Integer)  # 순매수 누적 금액 (단위: 만원)
    net_buy_qty: Mapped[int] = mapped_column(Integer)  # 순매수 누적 수량
    net_buy_change_amount: Mapped[int] = mapped_column(Integer)  # 순매수 증감 금액 (단위: 만원)
    net_buy_change_qty: Mapped[int] = mapped_column(Integer)  # 순매수 증감 수량


class KrNationalDebt(Base, TimestampMixin):
    # 국가 채무 정보
    __tablename__ = "kr_national_debt"

    year: Mapped[str] = mapped_column(String, primary_key=True)  # 연도 (예: '1997')
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 국가 채무 데이터


class StockMarketShortVolume(Base, TimestampMixin):
    # 주식시장 공매도 거래량
    __tablename__ = "stock_market_short_volume"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 시장 코드 (예: KOSPI, KOSDAQ)
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 거래일
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 공매도 거래량 데이터


class StockMarketShortAmount(Base, TimestampMixin):
    # 주식 시장 공매도
    __tablename__ = "stock_market_short_amount"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 시장 코드 (예: 'kospi')
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20230102')
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 공매도 데이터


class StockShort(Base, TimestampMixin):
    # 주식 공매도 정보
    __tablename__ = "stock_short"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드 (예: '082740')
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20200414')
    short_amount: Mapped[int] = mapped_column(Integer)  # 공매도 수량
    short_volume: Mapped[int] = mapped_column(Integer)  # 공매도 거래량
    short_ratio: Mapped[float] = mapped_column(Float)  # 공매도 비율
    avg_price: Mapped[int] = mapped_column(Integer)  # 평균 가격
    avg_price_ratio: Mapped[float] = mapped_column(Float)  # 평균 가격 비율
    close: Mapped[int] = mapped_column(Integer)  # 종가
    diff: Mapped[int] = mapped_column(Integer)  # 전일 대비 변화량
    diff_ratio: Mapped[float] = mapped_column(Float)  # 전일 대비 변화율
    volume: Mapped[int] = mapped_column(Integer)  # 거래량


class CreonStockFeature(Base, TimestampMixin):
    # Creon 주식 피처
    __tablename__ = "creon_stock_feature"
    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드 (예: '001740')
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20200513')
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 피처 데이터


class KosisPriceIndex(Base, TimestampMixin):
    # 소비자 물가지수 (CPI) 데이터
    __tablename__ = "kosis_price_index"
    type_name: Mapped[str] = mapped_column(String, primary_key=True)  # 지수 종류 (예: 'cpi')
    month: Mapped[str] = mapped_column(String, primary_key=True)  # 월 (예: '196501')
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class EtfCode(Base, TimestampMixin):
    # ETF 코드 정보
    __tablename__ = "etf_code"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # ETF 코드 (예: '069500')
    name: Mapped[str] = mapped_column(String)  # ETF 이름 (예: 'TIGER 200')
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)  # 활성화 여부
    extra: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # ETF 데이터


class MinuteBidVolume(Base, TimestampMixin):
    # 분 단위 호가 잔량
    __tablename__ = "minute_bid_volume"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드 (예: '004440')
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20250301')
    time: Mapped[int] = mapped_column(Integer, primary_key=True)  # 시간 (초 단위, 예: 900)
    buy_bid_volume: Mapped[int] = mapped_column(Integer)  # 매수 호가 잔량
    sell_bid_volume: Mapped[int] = mapped_column(Integer)  # 매도 호가 잔량
    buy_bid_volume_ratio: Mapped[float] = mapped_column(Float)  # 매수 호가 잔량 비율
    close: Mapped[int] = mapped_column(Integer)  # 종가
    diff: Mapped[int] = mapped_column(Integer)  # 전일 대비 변화량
    diff_ratio: Mapped[float] = mapped_column(Float)  # 전일 대비 변화율
    volume: Mapped[int] = mapped_column(Integer)  # 거래량


class TradeIntensity(Base, TimestampMixin):
    # 주식 거래 강도
    __tablename__ = "trade_intensity"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20220103')
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class InvestorTopNetBuyStock(Base, TimestampMixin):
    # KRX 투자자별 순매수상위종목
    # 'investor_type', 'year', 'code', 'name', 'sell_volume', 'buy_volume', 'net_buy_volume', 'sell_amount', 'buy_amount', 'net_buy_amount',
    __tablename__ = "investor_top_net_buy_stock"

    investor_type: Mapped[str] = mapped_column(String, primary_key=True)  # 투자자 구분 (예: '개인')
    year: Mapped[str] = mapped_column(String, primary_key=True)  # 연도 (예: '2022')
    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드 (예: '005930')
    name: Mapped[str] = mapped_column(String)  # 종목명 (예: '삼성전자')
    rank: Mapped[int] = mapped_column(Integer)  # 순위 (예: 1)
    sell_volume: Mapped[float] = mapped_column(Float)  # 매도 거래량 (단위: 백만)
    buy_volume: Mapped[float] = mapped_column(Float)  # 매수 거래량 (단위: 백만)
    net_buy_volume: Mapped[float] = mapped_column(Float)  # 순매수 거래량 (단위: 백만)
    sell_amount: Mapped[float] = mapped_column(Float)  # 매도 거래대금 (단위: 억)
    buy_amount: Mapped[float] = mapped_column(Float)  # 매수 거래대금 (단위: 억)
    net_buy_amount: Mapped[float] = mapped_column(Float)  # 순매수 거래대금 (단위: 억)


class InvestorYearAvgProfit(Base, TimestampMixin):
    __tablename__ = "investor_year_avg_profit"

    year: Mapped[str] = mapped_column(String, primary_key=True)  # 연도 (예: '2022')
    investor_type: Mapped[str] = mapped_column(String, primary_key=True)
    avg_profit: Mapped[float] = mapped_column(Float)


class KrMarketStockNumCap(Base, TimestampMixin):
    # 한국 시장 상장주식수 및 시가총액
    # 	month	상장회사 수 합계	상장회사 수 유가증권시장	상장회사 수 코스닥시장	시가총액 합계	시가총액 유가증권시장	시가총액 코스닥시장
    __tablename__ = "kr_market_stock_num_cap"

    month: Mapped[str] = mapped_column(String, primary_key=True)  # 월 (예: '202212')
    total_listed_companies: Mapped[int] = mapped_column(Integer)  # 상장회사 수 합계
    kospi_listed_companies: Mapped[int] = mapped_column(Integer)  # 상장회사 수 유가증권시장
    kosdaq_listed_companies: Mapped[int] = mapped_column(Integer)  # 상장회사 수 코스닥시장
    total_market_cap: Mapped[float] = mapped_column(Float)  # 시가총액 합계 (단위: 조원)
    kospi_market_cap: Mapped[float] = mapped_column(Float)  # 시가총액 유가증권시장 (단위: 조원)
    kosdaq_market_cap: Mapped[float] = mapped_column(Float)  # 시가총액 코스닥시장 (단위: 조원)


class Csi(Base, TimestampMixin):
    # CSI(소비자심리지수, Consumer Sentiment Index)
    __tablename__ = "csi"

    month: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[float] = mapped_column(Float)
    

class Cofix(Base, TimestampMixin):
    # COFIX(자금조달비용지수, Cost of Funds Index)
    # 공시일      대상월  신규취급액기준 COFIX  잔액기준 COFIX  신 잔액기준 COFIX
    __tablename__ = "cofix"

    publish_date: Mapped[str] = mapped_column(String, primary_key=True)  # 공시일
    target_month: Mapped[str] = mapped_column(String, primary_key=True)  # 대상월
    new_amount_cofix: Mapped[float] = mapped_column(Float)  # 신규취급액기준 COFIX
    balance_cofix: Mapped[float] = mapped_column(Float)  # 잔액기준 COFIX
    new_balance_cofix: Mapped[float] = mapped_column(Float)  # 신 잔액기준 COFIX


class ShortTermCofix(Base, TimestampMixin):
    # COFIX(자금조달비용지수, Cost of Funds Index)
    # 공시일                     대상기간  단기 COFIX
    __tablename__ = "short_term_cofix"
    
    publish_date: Mapped[str] = mapped_column(String, primary_key=True)  # 공시일
    target_start_date: Mapped[str] = mapped_column(String, primary_key=True)  # 대상기간
    target_end_date: Mapped[str] = mapped_column(String, primary_key=True)  # 대상기간
    value: Mapped[float] = mapped_column(Float)  # 단기 COFIX


class StockIncreaseRate(Base, TimestampMixin):
    # 종목 상승률
    __tablename__ = "stock_increase_rate"

    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20220103')
    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드
    name: Mapped[str] = mapped_column(String)  # 종목명
    close: Mapped[float] = mapped_column(Float)  # 종가
    diff_ratio: Mapped[float] = mapped_column(Float)  # 전일 대비 변화율
    diff_ratio_5: Mapped[float] = mapped_column(Float)  # 5일 상승률
    diff_ratio_10: Mapped[float] = mapped_column(Float)  # 10일 상승률
    diff_ratio_20: Mapped[float] = mapped_column(Float)  # 20일 상승률
    diff_ratio_60: Mapped[float] = mapped_column(Float)  # 60일 상승률


class PensionFundHolding(Base, TimestampMixin):
    # 연기금 보유 현황 (dart 대량)
    __tablename__ = "pension_fund_holding"

    rcept_no: Mapped[str] = mapped_column(String, primary_key=True)  # 접수 번호
    dcm_no: Mapped[str] = mapped_column(String)  # 문서 번호
    date: Mapped[str] = mapped_column(String)  # 보고의무발생일 (예: '20220103')
    report_base_date: Mapped[str] = mapped_column(String, nullable=True)  # 보고서작성기준일 (예: '20220103')
    rcept_dt: Mapped[str] = mapped_column(String, nullable=True)  # 접수일 (예: '20220103')
    corp_code: Mapped[str] = mapped_column(String)  # 기업 코드
    corp_name: Mapped[str] = mapped_column(String)  # 기업명
    stock_code: Mapped[str] = mapped_column(String)  # 종목 코드
    stock_num: Mapped[int] = mapped_column(Integer)  # 보유 주식 수
    stock_ratio: Mapped[float] = mapped_column(Float)  # 보유 주식 비율


class DartTaskStatus(Base, TimestampMixin):
    # DART 점검 상태
    __tablename__ = "dart_task_status"

    corp_code: Mapped[str] = mapped_column(String, primary_key=True)  # 기업 코드
    date: Mapped[str] = mapped_column(String)  # 점검 날짜 (예: '20220103')
    task: Mapped[str] = mapped_column(String)  # 점검 작업 이름
    status: Mapped[str] = mapped_column(String)  # 상태 (예: 'completed', 'pending')


class StockVolumeProfile(Base, TimestampMixin):
    # 주식 매물대
    __tablename__ = "stock_volume_profile"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20220103')
    poc_price: Mapped[float] = mapped_column(Float)  # POC (Point of Control)
    poc_volume: Mapped[float] = mapped_column(Float)  # POC 거래량
    close: Mapped[float] = mapped_column(Float)  # 종가
    close_poc_diff: Mapped[float] = mapped_column(Float)  # 종가와 POC 가격 차이
    close_poc_diff_ratio: Mapped[float] = mapped_column(Float)  # 종가와 POC 가격 차이 비율


class UpbitTicker(Base, TimestampMixin):
    # 업비트 틱 데이터
    __tablename__ = "upbit_ticker"
    
    code: Mapped[str] = mapped_column(String, primary_key=True)  # 마켓 코드 (예: 'KRW-BTC')
    timestamp: Mapped[int] = mapped_column(BigInteger, primary_key=True)  # 타임스탬프 (ms)
    opening_price: Mapped[float] = mapped_column(Float)  # 시가
    high_price: Mapped[float] = mapped_column(Float)  # 고가
    low_price: Mapped[float] = mapped_column(Float)  # 저가
    trade_price: Mapped[float] = mapped_column(Float)  # 현재가
    prev_closing_price: Mapped[float] = mapped_column(Float)  # 전일 종가
    change: Mapped[str] = mapped_column(String)  # 전일 종가 대비 가격 변동 방향
    change_price: Mapped[float] = mapped_column(Float)  # 전일 대비 가격 변동의 절대값
    signed_change_price: Mapped[float] = mapped_column(Float)  # 전일 대비 가격 변동 값
    change_rate: Mapped[float] = mapped_column(Float)  # 전일 대비 등락율의 절대값
    signed_change_rate: Mapped[float] = mapped_column(Float)  # 전일 대비 등락율
    trade_volume: Mapped[float] = mapped_column(Float)  # 가장 최근 거래량
    acc_trade_volume: Mapped[float] = mapped_column(Float)  # 누적 거래량(UTC 0시 기준)
    acc_trade_volume_24h: Mapped[float] = mapped_column(Float)  # 24시간 누적 거래량
    acc_trade_price: Mapped[float] = mapped_column(Float)  # 누적 거래대금(UTC 0시 기준)
    acc_trade_price_24h: Mapped[float] = mapped_column(Float)  # 24시간 누적 거래대금
    trade_date: Mapped[str] = mapped_column(String)  # 최근 거래 일자(UTC)
    trade_time: Mapped[str] = mapped_column(String)  # 최근 거래 시각(UTC)
    trade_timestamp: Mapped[int] = mapped_column(BigInteger)  # 체결 타임스탬프(ms)
    ask_bid: Mapped[str] = mapped_column(String)  # 매수/매도 구분
    acc_ask_volume: Mapped[float] = mapped_column(Float)  # 누적 매도량
    acc_bid_volume: Mapped[float] = mapped_column(Float)  # 누적 매수량
    highest_52_week_price: Mapped[float] = mapped_column(Float)  # 52주 최고가
    highest_52_week_date: Mapped[str] = mapped_column(String)  # 52주 최고가 달성일
    lowest_52_week_price: Mapped[float] = mapped_column(Float)  # 52주 최저가
    lowest_52_week_date: Mapped[str] = mapped_column(String)  # 52주 최저가 달성일
    market_state: Mapped[str] = mapped_column(String)  # 거래상태
    delisting_date: Mapped[str] = mapped_column(String, nullable=True)  # 거래지원 종료일
    stream_type: Mapped[str] = mapped_column(String)  # 스트림 타입


class UpbitOrderbook(Base, TimestampMixin):
    # 업비트 호가 잔량
    __tablename__ = "upbit_orderbook"
    
    code: Mapped[str] = mapped_column(String, primary_key=True)  # 마켓 코드 (예: 'KRW-BTC')
    timestamp: Mapped[int] = mapped_column(BigInteger, primary_key=True)  # 타임스탬프 (예: 1618820023456)
    total_ask_size: Mapped[float] = mapped_column(Float)  # 총 매도 잔량
    total_bid_size: Mapped[float] = mapped_column(Float)  # 총 매수 잔량
    orderbook_units: Mapped[list] = mapped_column(JSON, nullable=False, default=list)  # 호가 단위 리스트 (30개)
    stream_type: Mapped[str] = mapped_column(String)  # 스트림 타입 (예: 'SNAPSHOT')
    level: Mapped[int] = mapped_column(Integer)  # 레벨


class UpbitMinuteCandle(Base, TimestampMixin):
    # 업비트 분봉 캔들
    __tablename__ = "upbit_minute_candle"
    # {
    #     "market": "KRW-BTC",
    #     "candle_date_time_utc": "2025-07-01T12:00:00",
    #     "candle_date_time_kst": "2025-07-01T21:00:00",
    #     "opening_price": 145831000,
    #     "high_price": 145831000,
    #     "low_price": 145752000,
    #     "trade_price": 145759000,
    #     "timestamp": 1751327999833,
    #     "candle_acc_trade_price": 4022470467.03403,
    #     "candle_acc_trade_volume": 27.58904602,
    #     "unit": 1
    # }
    market: Mapped[str] = mapped_column(String, primary_key=True)  # 마켓 코드 (예: 'KRW-BTC')
    candle_date_time_utc: Mapped[str] = mapped_column(String, primary_key=True)  # 캔들 기준 시각 (UTC)
    candle_date_time_kst: Mapped[str] = mapped_column(String)  # 캔들 기준 시각 (KST)
    opening_price: Mapped[float] = mapped_column(Float)  # 시가
    high_price: Mapped[float] = mapped_column(Float)  # 고가
    low_price: Mapped[float] = mapped_column(Float)  # 저가
    trade_price: Mapped[float] = mapped_column(Float)  # 현재가
    timestamp: Mapped[int] = mapped_column(BigInteger, primary_key=True)  # 타임스탬프 (예: 1618820020000)
    candle_acc_trade_price: Mapped[float] = mapped_column(Float)  # 누적 거래대금
    candle_acc_trade_volume: Mapped[float] = mapped_column(Float)  # 누적 거래량
    unit: Mapped[int] = mapped_column(Integer)  # 분 단위 (예: 1, 3, 5, 10, 15, 30, 60, 240)


class AiStockAnalysis(Base, TimestampMixin):
    # AI 기반 종목 분석
    __tablename__ = "ai_stock_analysis"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # 종목 코드
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20230615')
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 종목명
    investment_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 투자 점수 (0~100)
    investment_opinion: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 투자 의견 (-2~2)
    factor_scores: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # 팩터별 평가점수·산정근거
    factor_explanations: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # 6개 팩터별 상세 설명
    score_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 점수 신뢰도 (0~1)
    upside: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 가격 상단 예측 (원)
    downside: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 가격 하단 예측 (원)
    price_trend: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 가격 동향 분석
    trading_volume: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 거래량 분석
    technical_indicators: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 기술적 지표 분석
    support_resistance: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 지지선·저항선 분석
    pattern_recognition: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 패턴 인식 분석
    fundamental_analysis: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 기본적 지표 분석
    supply_demand: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 수급 분석
    sector_analysis: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 섹터 분석
    comprehensive_evaluation: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 종합 평가
    model: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 사용 모델명
    version: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 프롬프트 버전
    tags: Mapped[list[str]] = mapped_column(JSON, nullable=False, server_default='{}')  # SNS 게시 태그 (예: ['threads'])


class AiEtfAnalysis(Base, TimestampMixin):
    # AI 기반 ETF 분석
    __tablename__ = "ai_etf_analysis"

    code: Mapped[str] = mapped_column(String, primary_key=True)  # ETF 코드 (예: '069500')
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20230615')
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # ETF명
    investment_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 투자 점수 (0~100)
    investment_opinion: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 투자 의견 (-2~2)
    factor_scores: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # 팩터별 평가점수·산정근거
    score_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 점수 신뢰도 (0~1)
    upside: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 가격 상단 예측 (원)
    downside: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 가격 하단 예측 (원)
    price_trend: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 가격 동향 분석
    trading_volume: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 거래량 분석
    technical_indicators: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 기술적 지표 분석
    support_resistance: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 지지선·저항선 분석
    pattern_recognition: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 패턴 인식 분석
    fundamental_analysis: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 포트폴리오·구성 분석
    supply_demand: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 수급 분석
    sector_analysis: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 시장 포지셔닝 분석
    comprehensive_evaluation: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 종합 평가
    model: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 사용 모델명
    version: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 프롬프트 버전
    tags: Mapped[list[str]] = mapped_column(JSON, nullable=False, server_default='{}')  # SNS 게시 태그 (예: ['threads'])


class AiStockMarketAnalysis(Base, TimestampMixin):
    # AI 기반 주식 시장(코스피/코스닥) 분석
    __tablename__ = "ai_stock_market_analysis"

    market: Mapped[str] = mapped_column(String, primary_key=True)  # 시장 구분 ('kospi' / 'kosdaq')
    date: Mapped[str] = mapped_column(String, primary_key=True)    # 날짜 (예: '20260426')
    market_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)     # 시장 매력도 점수 (0~100)
    market_direction: Mapped[Optional[int]] = mapped_column(Integer, nullable=True) # 시장 방향 의견 (-2~2)
    upside: Mapped[Optional[float]] = mapped_column(Float, nullable=True)           # 지수 상단 예측
    downside: Mapped[Optional[float]] = mapped_column(Float, nullable=True)         # 지수 하단 예측
    price_trend: Mapped[Optional[str]] = mapped_column(String, nullable=True)       # 지수 가격 추세
    trading_volume: Mapped[Optional[str]] = mapped_column(String, nullable=True)    # 시장 거래량 분석
    technical_indicators: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 기술적 지표 분석
    support_resistance: Mapped[Optional[str]] = mapped_column(String, nullable=True)    # 주요 지지·저항 레벨
    market_breadth: Mapped[Optional[str]] = mapped_column(String, nullable=True)    # 시장 폭 (상승/하락 종목 분석)
    investor_activity: Mapped[Optional[str]] = mapped_column(String, nullable=True) # 투자자별 매매 동향
    credit_balance: Mapped[Optional[str]] = mapped_column(String, nullable=True)    # 신용거래잔고 분석
    macro_environment: Mapped[Optional[str]] = mapped_column(String, nullable=True) # 거시 환경 (금리/환율/VIX 등)
    sector_rotation: Mapped[Optional[str]] = mapped_column(String, nullable=True)   # 섹터 순환 및 리더십
    comprehensive_evaluation: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 종합 시장 평가
    model: Mapped[Optional[str]] = mapped_column(String, nullable=True)    # 사용 모델명
    version: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 프롬프트 버전


class AiFinancialNewsSummary(Base, TimestampMixin):
    # AI 기반 금융 주요뉴스 주제별 요약
    __tablename__ = "ai_financial_news_summary"

    date: Mapped[str] = mapped_column(String, primary_key=True)  # 날짜 (예: '20260304')
    summary_id: Mapped[str] = mapped_column(String, primary_key=True)  # 요약 ID (순번, 예: '001')
    article_ids: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # 포함된 기사 ID 목록

    __table_args__ = (
        Index('ix_ai_fns_article_ids', article_ids, postgresql_using='gin'),
        {'extend_existing': True},
    )
    
    published_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 최초 기사 게시 시각 (초 제외, 예: '2026-03-13 09:30')
    title: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 주제 대표 제목
    summary: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # AI 요약
    market_impact: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 시장 영향도 (-2~2)
    related_codes: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 관련 종목 코드 (콤마 구분)
    keywords: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 핵심 키워드 (콤마 구분)
    category: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 기사 분류
    model: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 사용 모델명
    tags: Mapped[list[str]] = mapped_column(JSON, nullable=False, server_default='{}')  # SNS 게시 태그 (예: ['threads'])


class AiFinancialNewsArticle(Base, TimestampMixin):
    """수집된 개별 금융뉴스 원문 메타데이터. 서비스 조회용."""

    __tablename__ = "ai_financial_news_article"

    article_id: Mapped[str] = mapped_column(String, primary_key=True)
    date: Mapped[str] = mapped_column(String, nullable=False)
    query: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    related_codes: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    published_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    office_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    __table_args__ = (
        Index("ix_ai_fna_date_published", date, published_at),
        Index("ix_ai_fna_related_codes", related_codes),
        Index("ix_ai_fna_query", query),
    )


class ThreadsPostLog(Base, TimestampMixin):
    """Threads 게시 이력 및 중복/빈도 제한용 로그."""

    __tablename__ = "threads_post_log"

    post_id: Mapped[str] = mapped_column(String, primary_key=True)
    post_type: Mapped[str] = mapped_column(String, nullable=False)  # news / stock / etf
    posting_date: Mapped[str] = mapped_column(String, nullable=False)  # KST YYYYMMDD
    posted_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=False)
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    source_hash: Mapped[str] = mapped_column(String, nullable=False)
    source_keys: Mapped[list] = mapped_column(JSON, nullable=False, server_default='[]')

    __table_args__ = (
        Index("ix_threads_post_log_date_type", posting_date, post_type),
        Index("ix_threads_post_log_type_time", post_type, posted_at),
        Index("ix_threads_post_log_content_hash", content_hash),
        Index("ix_threads_post_log_source_hash", source_hash),
    )


class AiFsBatSummary(Base, TimestampMixin):
    # AI 기반 FsBat 재무제표 요약
    __tablename__ = "ai_fs_bat_summary"

    corp_code: Mapped[str] = mapped_column(String, primary_key=True)  # DART 기업 코드
    date: Mapped[str] = mapped_column(String, primary_key=True)        # 분석 기준 최근 결산일 (예: '2024-12-31')
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 기업명
    code: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 종목 코드
    years: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 분석 연수
    date_from: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 분석 시작 결산일 (가장 오래된)
    summary: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # AI 요약 전문 (Markdown)
    model: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 사용 모델명
    version: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 프롬프트 버전
    tags: Mapped[list[str]] = mapped_column(JSON, nullable=False, server_default='{}')  # SNS 게시 태그


class BinanceMinuteCandle(Base, TimestampMixin):
    # 바이낸스 분봉 캔들
    __tablename__ = "binance_minute_candle"
    # Binance GET /api/v3/klines 응답:
    # [open_time, open, high, low, close, volume, close_time,
    #  quote_asset_volume, number_of_trades,
    #  taker_buy_base_volume, taker_buy_quote_volume, ignore]
    symbol: Mapped[str] = mapped_column(String, primary_key=True)  # 심볼 (예: 'ETHUSDT')
    open_time: Mapped[int] = mapped_column(BigInteger, primary_key=True)  # 캔들 시작 시각 (ms)
    open_time_utc: Mapped[str] = mapped_column(String)  # 캔들 시작 시각 (UTC ISO str)
    open_price: Mapped[float] = mapped_column(Float)  # 시가
    high_price: Mapped[float] = mapped_column(Float)  # 고가
    low_price: Mapped[float] = mapped_column(Float)  # 저가
    close_price: Mapped[float] = mapped_column(Float)  # 종가
    volume: Mapped[float] = mapped_column(Float)  # 거래량 (base asset)
    close_time: Mapped[int] = mapped_column(BigInteger)  # 캔들 종료 시각 (ms)
    quote_volume: Mapped[float] = mapped_column(Float)  # 거래대금 (quote asset)
    trade_count: Mapped[int] = mapped_column(Integer)  # 거래 횟수
    unit: Mapped[int] = mapped_column(Integer, primary_key=True)  # 분 단위 (1, 3, 5, 15, 30, 60, ...)


class BinanceLongShortRatio(Base, TimestampMixin):
    # 바이낸스 선물 롱숏비율 스냅샷
    # globalLongShortAccountRatio: 전체 계정 기준
    # topLongShortAccountRatio:   상위 트레이더 계정 기준 (Smart Money)
    # topLongShortPositionRatio:  상위 트레이더 포지션 규모 기준
    # GET https://fapi.binance.com/futures/data/globalLongShortAccountRatio
    # GET https://fapi.binance.com/futures/data/topLongShortAccountRatio
    # GET https://fapi.binance.com/futures/data/topLongShortPositionRatio
    __tablename__ = "binance_long_short_ratio"

    symbol: Mapped[str] = mapped_column(String, primary_key=True)
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    timestamp_utc: Mapped[str] = mapped_column(String)
    # 전체 계정 기준
    long_account_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # 롱 계정 비율
    short_account_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 숏 계정 비율
    long_short_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)     # 롱/숏 비율
    # 상위 트레이더 계정 기준
    top_long_account_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    top_short_account_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    top_long_short_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # 상위 트레이더 포지션 규모 기준
    top_pos_long_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    top_pos_short_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    top_pos_long_short_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class BinanceTakerVolume(Base, TimestampMixin):
    # 바이낸스 선물 테이커 매수/매도 거래량 비율
    # 시장가 주문(즉각 체결)의 방향 비율 — 단기 수요/공급 압력의 가장 직접적인 지표
    # GET https://fapi.binance.com/futures/data/takerlongshortRatio
    __tablename__ = "binance_taker_volume"

    symbol: Mapped[str] = mapped_column(String, primary_key=True)
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    timestamp_utc: Mapped[str] = mapped_column(String)
    buy_vol: Mapped[float] = mapped_column(Float)   # 테이커 매수 거래량 (base)
    sell_vol: Mapped[float] = mapped_column(Float)  # 테이커 매도 거래량 (base)
    buy_sell_ratio: Mapped[float] = mapped_column(Float)  # 매수/매도 비율 (>1: 매수 우세)


class BinanceOpenInterest(Base, TimestampMixin):
    # 바이낸스 선물 미결제약정 스냅샷 (1분 실시간 폴링 또는 집계 히스토리)
    # Real-time: GET https://fapi.binance.com/fapi/v1/openInterest
    # History:   GET https://fapi.binance.com/futures/data/openInterestHist (최소 5m 단위)
    __tablename__ = "binance_open_interest"

    symbol: Mapped[str] = mapped_column(String, primary_key=True)  # 심볼 (예: 'ETHUSDT')
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, primary_key=True)  # 시각 (ms epoch)
    timestamp_utc: Mapped[str] = mapped_column(String)  # 시각 (UTC ISO str)
    open_interest: Mapped[float] = mapped_column(Float)  # 미결제약정 수량 (코인 단위)
    open_interest_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 미결제약정 금액 (USDT)


class BinanceFundingRate(Base, TimestampMixin):
    # 바이낸스 선물 펀딩비 (8시간마다 정산 히스토리 + 실시간 예측값 스냅샷)
    # History:    GET https://fapi.binance.com/fapi/v1/fundingRate
    # Real-time:  GET https://fapi.binance.com/fapi/v1/premiumIndex (lastFundingRate + nextFundingTime)
    __tablename__ = "binance_funding_rate"

    symbol: Mapped[str] = mapped_column(String, primary_key=True)  # 심볼 (예: 'ETHUSDT')
    funding_time_ms: Mapped[int] = mapped_column(BigInteger, primary_key=True)  # 펀딩 시각 (ms epoch)
    funding_time_utc: Mapped[str] = mapped_column(String)  # 펀딩 시각 (UTC ISO str)
    funding_rate: Mapped[float] = mapped_column(Float)  # 펀딩비율 (예: 0.0001 = 0.01%)
    mark_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 마크 가격 (USDT)


class FeatureVector(Base, TimestampMixin):
    # 피처 벡터
    __tablename__ = "feature_vector"

    name: Mapped[str] = mapped_column(String, primary_key=True)  # 피처 이름 (예: 'TIGER 200')
    code: Mapped[str] = mapped_column(String, primary_key=True)  # 코드 (예: '069500')
    version: Mapped[str] = mapped_column(String, primary_key=True)  # 버전 (예: '1')
    x: Mapped[str] = mapped_column(String, primary_key=True)  # 키 (예: '20150101')
    y: Mapped[list] = mapped_column(PortableVector, nullable=False)  # 피처 벡터
    meta: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # 메타 정보 (예: {"unit": "day"})


class MarketCreditBalance(Base, TimestampMixin):
    # 시장 전체 신용거래잔고 (KOFIA freesis STATSCU0100000070)
    # 단위: 백만원
    __tablename__ = "market_credit_balance"

    date: Mapped[str] = mapped_column(String, primary_key=True)  # 일자 (YYYYMMDD)
    loan_balance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)       # 신용융자잔고 합계
    loan_balance_kospi: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 신용융자잔고 유가증권
    loan_balance_kosdaq: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 신용융자잔고 코스닥
    short_balance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)       # 신용대주잔고 합계
    short_balance_kospi: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 신용대주잔고 유가증권
    short_balance_kosdaq: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 신용대주잔고 코스닥


class EtfPortfolioPaperTradeRun(Base, TimestampMixin):
    # ETF 포트폴리오 스윙 모의투자 실행 이력
    __tablename__ = "etf_portfolio_paper_trade_run"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    rebalance_key: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    trade_date: Mapped[str] = mapped_column(String, nullable=False)
    market_date: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    market_status: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    action: Mapped[str] = mapped_column(String, nullable=False)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    account_type: Mapped[str] = mapped_column(String, nullable=False)  # mock / real
    dry_run: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(String, nullable=False)  # started / skipped / completed / failed
    skip_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    stale_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    feature_source: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    feature_date: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    latest_market_date: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    order_blocked_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    cancel_open_orders: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    open_order_count_before: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    open_order_count_after_cancel: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    requested_sell_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    requested_buy_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    executed_sell_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    executed_buy_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cash_before: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cash_after_estimate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    managed_value_before: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_equity_before: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    capital_base: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    invest_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cash_weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trading_fee: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    min_order_amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_sell_amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_buy_amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    post_trade_cash: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    post_trade_managed_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    post_trade_total_equity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    started_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    finished_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    raw_summary: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_eptr_trade_date", "trade_date"),
        Index("ix_eptr_status_trade_date", "status", "trade_date"),
        Index("ix_eptr_model_trade_date", "model_name", "trade_date"),
        {"extend_existing": True},
    )


class EtfPortfolioPaperTradePosition(Base, TimestampMixin):
    # 실행 시점 포지션 및 목표 비중 스냅샷
    __tablename__ = "etf_portfolio_paper_trade_position"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    code: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    current_qty: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    target_qty: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    desired_qty: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    diff_qty: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    current_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    target_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    target_weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    last_buy_trade_date: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    holding_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    sell_rank_limit: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    order_side: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    order_amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    order_blocked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    order_block_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    sell_guard_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    sell_guard_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    __table_args__ = (
        Index("ix_eptp_run_id", "run_id"),
        Index("ix_eptp_code_trade", "code"),
        {"extend_existing": True},
    )


class EtfPortfolioPaperTradeSignal(Base, TimestampMixin):
    # 실행 시점 ETF별 원신호/제약반영/최종결정 스냅샷
    __tablename__ = "etf_portfolio_paper_trade_signal"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    code: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    raw_rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    raw_target_weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    limited_target_weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    final_target_weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_qty: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    current_weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    desired_qty: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    final_target_qty: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_held_before: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_top_entry: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_selected_after_limit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    decision: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    decision_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    __table_args__ = (
        Index("ix_epts_run_id", "run_id"),
        Index("ix_epts_code", "code"),
        Index("ix_epts_raw_rank", "raw_rank"),
        {"extend_existing": True},
    )


class EtfPortfolioPaperTradeOrder(Base, TimestampMixin):
    # 주문 및 미체결 취소 이력
    __tablename__ = "etf_portfolio_paper_trade_order"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    order_key: Mapped[str] = mapped_column(String, primary_key=True)
    category: Mapped[str] = mapped_column(String, nullable=False)  # cancel_open / rebalance
    side: Mapped[str] = mapped_column(String, nullable=False)  # buy / sell / cancel
    code: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    qty: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    remaining_qty: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dry_run: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    order_no: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    original_order_no: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    filled_qty: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    filled_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    filled_amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    filled_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    cost_basis_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    realized_return_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    final_status: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    raw_response: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_epto_run_id", "run_id"),
        Index("ix_epto_code", "code"),
        Index("ix_epto_order_no", "order_no"),
        {"extend_existing": True},
    )


if __name__ == "__main__":
    from .db import psql
    psql.create_tables(Base)
    psql.create_trigger_for_update_time()
    print("테이블 생성 완료")
