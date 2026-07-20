"""RAGAS 스타일 RAG 정량평가 — 생성 품질을 수치화(직접 구현).

순수 지표(토큰 F1·context precision/recall·faithfulness)는 네트워크 없이 단위테스트되고,
claim 분해·근거 판정 등 LLM 의존부는 judge를 주입한다(무료 Gemini 등 / 테스트는 stub).
"""
