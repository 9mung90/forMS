# -*- coding: utf-8 -*-
import sys
import re
import torch
import logging

# 트랜스포머 라이브러리 (Hugging Face)
from transformers import AutoTokenizer, BertForSequenceClassification

# 경고 메시지 숨기기 (데모 화면을 깔끔하게 하기 위함)
logging.getLogger("transformers").setLevel(logging.ERROR)

# =============================================================================
# 1. 설정 및 데이터 (욕설 사전, 피드백 메시지)
# =============================================================================

# [설정] 사용할 모델 (Smilegate AI의 UnSmile 데이터셋으로 학습된 모델)
MODEL_NAME = "smilegate-ai/kor_unsmile"

# [데이터] 혐오 유형 라벨 (모델 출력 순서와 동일해야 함)
LABELS = [
    "여성/가족", "남성", "성소수자", "인종/국적",
    "연령", "지역", "종교", "기타 혐오",
    "악플/욕설", "clean"
]

# [데이터] 욕설/비하 표현 치환 사전 (Badword Masking)
# 시연을 위해 대표적인 단어들로 구성
BADWORD_DICT = {
    "병신": "사람",
    "ㅂㅅ": "사람",
    "미친놈": "사람",
    "꺼져": "저리 가 줘",
    "쓰레기": "좋지 않은 행동을 하는 사람",
    "짱깨": "중국인",
    "쪽발이": "일본인",
    "느금마": "가족",
    "개같은": "나쁜",
}

# [데이터] 혐오 유형별 순화 가이드 (Rewrite Suggestion)
FEEDBACK_DICT = {
    "여성/가족": "성별이나 가족 전체를 일반화하기보다는, 특정 상황이나 행동에 대해 설명하는 표현이 좋습니다.",
    "남성": "특정 성별을 싸잡아 비난하기보다는, 문제라고 느낀 행동에 대해 구체적으로 이야기해보세요.",
    "성소수자": "성적 지향이나 정체성을 공격하기보다, 서로의 다름을 인정하고 존중하는 표현을 사용해보세요.",
    "인종/국적": "국적이나 인종 전체를 비하하지 말고, 객관적인 상황이나 제도를 설명하는 것이 좋습니다.",
    "연령": "나이만을 이유로 비하하기보다는, 세대 간 차이를 이해하려는 태도가 필요합니다.",
    "지역": "특정 지역 사람 전체를 매도하기보다는, 본인이 겪은 개별적인 경험으로 국한하여 표현해 주세요.",
    "종교": "신앙 자체를 공격하기보다는, 동의하지 않는 의견에 대해 논리적으로 반박해 보세요.",
    "기타 혐오": "집단 전체를 향한 혐오 표현은 지양하고, 구체적인 사실에 기반해 건전하게 대화해 주세요.",
    "악플/욕설": "강한 감정이 들 때는 잠시 진정하고, 욕설 대신 불만 사항을 구체적으로 적어보세요.",
    "clean": "상대를 존중하는 표현입니다. 지금처럼 건강한 온라인 소통을 이어가 주세요."
}


# 정규식 패턴 컴파일 (치환 속도 향상)
def build_badword_patterns(badword_dict):
    patterns = {}
    for bad in badword_dict.keys():
        # 욕설 뒤에 붙는 조사(이, 가, 은, 는 등)를 유연하게 처리하기 위한 정규식
        escaped = re.escape(bad)
        pattern = re.compile(rf"({escaped})([이가은는을를]*)")
        patterns[bad] = pattern
    return patterns


BADWORD_PATTERNS = build_badword_patterns(BADWORD_DICT)


# =============================================================================
# 2. 핵심 기능 함수 (모델 로드, 전처리, 예측)
# =============================================================================

def load_system():
    """모델과 토크나이저를 로드합니다."""
    print("\n[시스템] AI 모델을 로딩 중입니다... (잠시만 기다려 주세요)")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()  # 평가 모드로 설정
        print(f"[시스템] 모델 로딩 완료! (가속 장치: {device})")
        return model, tokenizer, device
    except Exception as e:
        print(f"\n[오류] 모델을 불러오는 데 실패했습니다.\n원인: {e}")
        print("인터넷 연결을 확인하거나 'pip install transformers torch' 설치 여부를 확인해주세요.")
        sys.exit(1)


def replace_badwords_func(text):
    """문장 내 욕설을 순화된 표현으로 치환합니다."""
    cleaned_text = text
    logs = []

    # 긴 단어부터 치환하기 위해 정렬 (예: '개'보다 '개새끼'를 먼저 찾음)
    sorted_keys = sorted(BADWORD_DICT.keys(), key=len, reverse=True)

    for bad in sorted_keys:
        pattern = BADWORD_PATTERNS[bad]
        replacement = BADWORD_DICT[bad]

        def _sub_func(match):
            word = match.group(1)
            tail = match.group(2) or ""  # 조사
            # 로그에 기록 (예: '병신' -> '사람')
            logs.append(f"'{word}' -> '{replacement}'")
            return replacement + tail

        cleaned_text, _ = pattern.subn(_sub_func, cleaned_text)

    return cleaned_text, logs


def analyze_sentence(text, model, tokenizer, device):
    """문장을 분석하여 혐오 유형, 순화 문장, 피드백을 반환합니다."""

    # 1. 모델 예측 (Inference)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # 로짓(Logits)을 시그모이드 함수로 변환하여 0~1 사이 확률값 도출
        probs = torch.sigmoid(outputs.logits[0])

    probs_list = probs.cpu().tolist()

    # 2. 임계값(Threshold) 0.5 이상인 라벨 추출
    detected_labels = []
    label_scores = {}  # 피드백 선정을 위해 점수 저장

    for i, score in enumerate(probs_list):
        if score >= 0.5:
            label_name = LABELS[i]
            detected_labels.append(label_name)
            label_scores[label_name] = score

    # 3. 욕설 치환 수행
    cleaned_text, replace_logs = replace_badwords_func(text)

    # 4. 맞춤형 피드백 선정
    # 감지된 혐오 라벨 중 '점수가 가장 높은' 라벨의 피드백을 대표로 보여줌
    main_feedback = ""

    # clean이 아니면서 감지된 라벨이 있는 경우
    hate_labels = [l for l in detected_labels if l != "clean"]

    if hate_labels:
        # 혐오 점수가 가장 높은 라벨 찾기
        top_label = max(hate_labels, key=lambda l: label_scores[l])
        main_feedback = FEEDBACK_DICT.get(top_label, "상대를 존중하는 고운 말을 써주세요.")
    elif "clean" in detected_labels:
        main_feedback = FEEDBACK_DICT["clean"]
    else:
        # 아무것도 감지되지 않았거나(Threshold 미만) 애매한 경우
        detected_labels.append("정상(Clean)")
        main_feedback = FEEDBACK_DICT["clean"]

    return {
        "original": text,
        "cleaned": cleaned_text,
        "labels": detected_labels,
        "replace_logs": replace_logs,
        "feedback": main_feedback
    }


# =============================================================================
# 3. 메인 실행 루프 (데모 UI)
# =============================================================================

def run_demo():
    # 모델 로드
    model, tokenizer, device = load_system()

    print("\n" + "=" * 70)
    print("      [ 기계학습 8조: 문맥 기반 실시간 혐오 표현 다중 분류 시스템 ]")
    print("=" * 70)
    print(" ※ 종료하려면 'q' 또는 'exit'을 입력하세요.\n")

    while True:
        try:
            # 사용자 입력
            print("-" * 70)
            user_input = input("📝 분석할 문장을 입력하세요: ")

            # 종료 조건
            if user_input.strip().lower() in ['q', 'exit', 'quit']:
                print("\n[시스템] 데모를 종료합니다. 감사합니다.")
                break

            # 빈 입력 처리
            if not user_input.strip():
                continue

            # 분석 수행
            result = analyze_sentence(user_input, model, tokenizer, device)

            # --- 결과 출력 화면 (가독성 최적화) ---
            print("\n   [ 분석 결과 ]")

            # 1. 원문
            print(f"   ▶ 원문 문장 :  {result['original']}")

            # 2. 순화 (치환된 경우에만 표시)
            if result['replace_logs']:
                print(f"   ▶ 순화 문장 :  {result['cleaned']}")
                print(f"   ▶ 치환 내역 :  {', '.join(result['replace_logs'])}")
            else:
                print(f"   ▶ 순화 문장 :  (변동 사항 없음)")

            # 3. 감지 유형
            # 리스트를 보기 좋게 문자열로 변환
            labels_str = ", ".join(result['labels'])
            print(f"   ▶ 감지 유형 :  [{labels_str}]")

            # 4. AI 피드백
            print(f"   ▶ AI 피드백 :  \"{result['feedback']}\"")
            print("")  # 공백 라인

        except KeyboardInterrupt:
            print("\n\n[시스템] 강제 종료되었습니다.")
            break
        except Exception as e:
            print(f"\n[오류] 처리 중 문제가 발생했습니다: {e}")


if __name__ == "__main__":
    run_demo()