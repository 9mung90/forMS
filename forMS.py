# --------------------------------------------------------------------------
# 1. 라이브러리 및 기본 설정
# --------------------------------------------------------------------------
import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# GPU 사용 설정 (사용 가능하면 GPU, 아니면 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용할 디바이스: {device}")


# --------------------------------------------------------------------------
# 2. 데이터셋 준비
# --------------------------------------------------------------------------
print("Hugging Face에서 'smilegate-ai/kor_unsmile' 데이터셋을 로드합니다...")
dataset = load_dataset("smilegate-ai/kor_unsmile")

# 사용할 혐오 종류 라벨 정의
hate_labels = ['여성/가족', '남성', '인종/국적', '연령', '지역', '종교', '성소수자', '기타 혐오']
num_labels = len(hate_labels)


def preprocess_data(examples):
    texts = examples['문장']
    # 다중 라벨 분류를 위해 각 라벨을 float 리스트로 변환
    labels = [
        [float(examples[label][i]) for label in hate_labels]
        for i in range(len(texts))
    ]
    return {'text': texts, 'labels': labels}


print("데이터셋을 다중 라벨 형식으로 전처리합니다...")
processed_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)


# --------------------------------------------------------------------------
# 3. 모델, 토크나이저, 데이터 로더 준비
# --------------------------------------------------------------------------
MODEL_NAME = 'klue/bert-base'
print(f"'{MODEL_NAME}' 모델과 토크나이저를 로드합니다...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    problem_type="multi_label_classification"
)
model.to(device)


def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)


print("텍스트 데이터를 토큰화합니다...")
tokenized_dataset = processed_dataset.map(tokenize_function, batched=True)

# 데이터셋을 학습용과 검증용으로 분리합니다.
train_dataset = tokenized_dataset['train']
eval_dataset = tokenized_dataset['valid']

# PyTorch가 데이터를 배치로 묶을 때 라벨 타입을 float32로 변환하는 함수
def collate_fn(batch):
    # 수동으로 텐서 변환
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    attention_mask = torch.tensor([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch]).to(torch.float32)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


# DataLoader 정의 시 collate_fn을 적용
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, collate_fn=collate_fn)
eval_dataloader = DataLoader(eval_dataset, batch_size=16, collate_fn=collate_fn)


# --------------------------------------------------------------------------
# 4. 모델 학습
# --------------------------------------------------------------------------
optimizer = AdamW(model.parameters(), lr=5e-6)
num_epochs = 5  # 충분한 학습을 위해 5 에폭으로 설정
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

progress_bar = tqdm(range(num_training_steps))

print("\n--- 모델 학습을 시작합니다 ---")
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    print(f"Epoch {epoch + 1} 완료 | Loss: {loss.item()}")

print("--- 모델 학습 완료 ---\n")


# --------------------------------------------------------------------------
# 5. 새로운 문장으로 예측하는 함수
# --------------------------------------------------------------------------
def classify_sentence(sentence):
    print(f"\n--- 입력 문장: '{sentence}' ---")
    model.eval()  # 모델을 평가 모드로 설정

    # 문장 토큰화 및 디바이스로 이동
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # 로짓에 시그모이드 함수를 적용하여 각 라벨의 확률을 계산
    probs = torch.sigmoid(outputs.logits).squeeze()
    # 0.5를 기준으로 혐오 여부 판단 (임계값은 조정 가능)
    predictions = (probs > 0.5).int()

    detected_hates = []
    # cpu()를 호출하여 GPU 텐서를 CPU로 이동시킨 후 반복
    for i, label in enumerate(hate_labels):
        if predictions[i].cpu().item() == 1:
            detected_hates.append(label)

    # 최종 결과 출력
    if not detected_hates:
        print(">> 예측: [정상 발언]")
    else:
        print(f">> 예측: [혐오 표현]")
        print(f">> 혐오 종류: {', '.join(detected_hates)}")


# --- 함수 테스트 ---
classify_sentence("동남아 사람들은 미개해서 우리나라에 오면 안 돼.")
classify_sentence("외국인 노동자들은 다 추방해야 한다.")
classify_sentence("오늘 날씨 정말 좋아서 기분이 상쾌해요!")
classify_sentence("전라도 사람들은 뒤통수치니까 믿으면 안 된다")