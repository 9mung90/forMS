import torch  # 파이토치 라이브러리 (오픈 소스 머신 러닝 라이브러리) 를 가져옴
from datasets import load_dataset  # 허깅페이스 데이터셋을 쉽게 불러오는 기능

# 트랜스포머 라이브러리 업데이트 내용을 알려주는거네
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup # 트랜스포머 라이브러리에서 필요한 클래스들 가져오기

from torch.optim import AdamW #그래서 파이토치에서 직접 AdamW 옵티마이저를 가져와야 함
from torch.utils.data import DataLoader # 데이터를 배치 단위로 묶어주는 데이터로더
from tqdm.auto import tqdm # 학습 진행 상황을 막대그래프로 보여줘서 보기 편하게 해주는 기능

# GPU 사용 설정 (사용 가능하면 GPU, 아니면 CPU)
# 앵간하면 무조건 GPU 사용해야함, CPU말도 안되게 오래걸림
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 학습할 때 GPU 쓸 수 있으면 쓰고, 없으면 CPU 쓰라고 설정하는 부분
print(f"사용할 디바이스: {device}") # 그래서 뭘로 학습하는지 한번 찍어보는 코드



print("Hugging Face에서 'smilegate-ai/kor_unsmile' 데이터셋을 로드합니다...")
dataset = load_dataset("smilegate-ai/kor_unsmile") # 스마일게이트에서 만든 한국어 악성댓글 데이터셋을 다운로드함

# 사용할 혐오 종류 라벨 정의
hate_labels = ['여성/가족', '남성', '인종/국적', '연령', '지역', '종교', '성소수자', '기타 혐오'] # 이 모델이 분류해야 할 혐오 표현의 종류들임
num_labels = len(hate_labels) # 라벨이 총 몇 개인지 세어보는거


def preprocess_data(examples): # 데이터를 모델이 쓰기 좋은 형태로 바꿔주는 함수로 정의함
    texts = examples['문장'] # 데이터셋에서 '문장' 열을 가져옴
    # 다중 라벨 분류를 위해 각 라벨을 float 리스트로 변환 # ex) [1, 0, 0, 1, 0, 0, 0, 0] 이런 식으로
    labels = [
        [float(examples[label][i]) for label in hate_labels]
        for i in range(len(texts))
    ]
    return {'text': texts, 'labels': labels} # 전처리가 끝난 텍스트와 라벨을 돌려줌


print("데이터셋을 다중 라벨 형식으로 전처리합니다...")
processed_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names) # 위에서 만든 함수로 전체 데이터셋을 한방에 처리하고 쓸모 없어진 원래 열들은 지워버림



#모델 정리 및 전처리
MODEL_NAME = 'klue/bert-base' #한국어 처리용 klue/bert-base 모델
print(f"'{MODEL_NAME}' 모델과 토크나이저를 로드합니다...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME) # 문장을 잘게 쪼개서 숫자로 바꿔주는 역할
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, # 위에서 정한 BERT 모델을 불러오고
    num_labels=num_labels, # 라벨 개수(8개)를 알려주고
    problem_type="multi_label_classification" # 여러 개의 정답을 동시에 가질 수 있는 문제(다중 라벨 분류)라고 알려주는거
)
model.to(device) # 만들어진 모델을 위에서 설정한 GPU로 보냄


def tokenize_function(examples): # 문장을 토크나이저로 처리하는 함수를 또 만듬
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128) # 문장을 쪼개고 길이를 128로 맞춤(짧으면 채우고 길면 자름)


print("텍스트 데이터를 토큰화합니다...")
tokenized_dataset = processed_dataset.map(tokenize_function, batched=True) # 위에서 만든 토큰화 함수로 데이터셋 전체를 또 한번에 처리

# 데이터셋을 학습용과 검증용으로 분리합니다.
train_dataset = tokenized_dataset['train'] # 학습에 쓸 데이터
eval_dataset = tokenized_dataset['valid'] # 모델 성능 검증에 쓸 데이터

# PyTorch가 데이터를 배치로 묶을 때 라벨 타입을 float32로 변환하는 함수
def collate_fn(batch): # 데이터로더가 데이터를 배치로 만들 때 어떻게 처리할지 정해주는 함수
    # set_format을 사용하지 않으므로 수동으로 텐서 변환
    # 데이터들을 파이토치가 계산할 수 있는 '텐서' 형태로 직접 바꿔줘야 함
    input_ids = torch.tensor([item['input_ids'] for item in batch]) # 토큰화된 숫자들(input_ids)을 텐서로
    attention_mask = torch.tensor([item['attention_mask'] for item in batch]) # 어텐션 마스크도 텐서로 (어디가 진짜 단어고 어디가 패딩인지 알려주는거)
    labels = torch.tensor([item['labels'] for item in batch]).to(torch.float32) # 라벨도 텐서로 바꿔주는데, 모델이 계산하려면 float 타입이어야 함
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels} # 딕셔너리 형태로 묶어서 돌려줌


# DataLoader 정의 시 collate_fn을 적용
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, collate_fn=collate_fn) # 학습용 데이터로더. 16개씩 묶고, 순서는 매번 섞어줌
eval_dataloader = DataLoader(eval_dataset, batch_size=16, collate_fn=collate_fn) # 검증용 데이터로더. 이건 굳이 안 섞어도 됨



optimizer = AdamW(model.parameters(), lr=5e-6) # 모델의 파라미터를 최적화할 AdamW 옵티마이저를 정의. lr은 학습률!
num_epochs = 3  # 전체 데이터를 몇 번 반복해서 학습할지 정하는거
#1은 낮고 5는 너무 높음 3~4가 적당한듯

num_training_steps = num_epochs * len(train_dataloader) # 총 학습 스텝이 몇 번인지 계산
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps) # 학습률을 점차적으로 조절해주는 스케줄러. 안정적인 학습을 도와줌

progress_bar = tqdm(range(num_training_steps)) # tqdm으로 진행상황 바를 만듦

print("\n--- 모델 학습을 시작합니다 ---")
model.train() # 모델을 학습모드로 바꿈
for epoch in range(num_epochs): # 정해진 에포크만큼 반복
    for batch in train_dataloader: # 데이터로더에서 배치(16개 묶음)를 하나씩 꺼내옴
        # collate_fn에서 이미 텐서로 변환했으므로 디바이스로만 이동
        batch = {k: v.to(device) for k, v in batch.items()} # 배치를 통째로 GPU(또는 CPU)로 보냄
        outputs = model(**batch) # 모델에 데이터를 넣어서 결과를 계산함
        loss = outputs.loss # 결과에서 손실(loss) 값을 뽑아냄, 모델이 얼마나 틀렸는지를 나타내는 지표
        loss.backward() # 손실을 기반으로 역전파를 수행, 각 파라미터를 어떻게 업데이트할지 계산하는 과정

        optimizer.step() # 옵티마이저가 역전파 결과를 바탕으로 모델의 파라미터를 업데이트함
        lr_scheduler.step() # 스케줄러도 업데이트
        optimizer.zero_grad() # 다음 배치를 위해 기울기(gradient)를 초기화. 이걸 안하면 기울기가 계속 쌓임
        progress_bar.update(1) # 진행상황 바를 한 칸 업데이트

    print(f"Epoch {epoch + 1} 완료 | Loss: {loss.item()}") # 한 에포크가 끝나면 마지막 손실 값을 출력

print("--- 모델 학습 완료 ---\n")


def classify_sentence(sentence): # 새로운 문장이 들어오면 혐오 표현인지 아닌지 판단하는 함수
    print(f"\n--- 입력 문장: '{sentence}' ---")
    model.eval()  # 모델을 평가 모드로 설정함 학습할 때와는 다르게 동작함

    # 문장 토큰화 및 디바이스로 이동
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128) # 입력 문장을 토큰화하고 파이토치 텐서('pt')로 만듬
    inputs = {k: v.to(device) for k, v in inputs.items()} # 토큰화된 데이터도 GPU(또는 CPU)로 보냄

    with torch.no_grad(): # 이 블록 안에서는 기울기 계산을 안함. 예측만 할거니까 계산량을 줄이는거
        outputs = model(**inputs) # 모델에 입력을 넣어서 결과 출력

    # 로짓에 시그모이드 함수를 적용하여 각 라벨의 확률을 계산
    probs = torch.sigmoid(outputs.logits).squeeze() # 모델이 뱉은 날것의 점수를 0과 1 사이의 확률 값으로 바꿔줌
    # 0.5를 기준으로 혐오 여부 판단 (임계값은 조정 가능)
    predictions = (probs > 0.5).int() # 확률이 0.5를 넘으면 해당 혐오 표현이 있다고(1), 아니면 없다고(0) 판단

    detected_hates = [] # 어떤 종류의 혐오가 감지됐는지 담을 리스트 함
    # cpu()를 호출하여 GPU 텐서를 CPU로 이동시킨 후 반복 # 결과를 처리하려면 GPU에 있는 텐서를 CPU로 가져와야 함
    for i, label in enumerate(hate_labels): # 라벨 하나하나 다 돌면서 확인
        if predictions[i].cpu().item() == 1: # 예측값이 1이면 = 혐오 표현이 있다고 판단되면
            detected_hates.append(label) # 리스트에 해당 라벨을 추가

    # 결과 출력
    if not detected_hates: # 감지된 혐오 표현 리스트가 비어있으면
        print(">> 예측: [정상 발언]") # 정상 발언이라고 출력
    else: # 리스트에 뭐라도 들어있으면
        print(f">> 예측: [혐오 표현]") # 혐오 표현이라고 출력함
        print(f">> 혐오 종류: {', '.join(detected_hates)}") # 어떤 종류의 혐오인지도 알려줌


# 테스트
classify_sentence("동남아 사람들은 미개해서 우리나라에 오면 안 돼")
classify_sentence("외국인 노동자들은 다 추방해야 한다.")
classify_sentence("오늘 날씨 정말 좋아서 기분이 상쾌해요!")
classify_sentence("전라도 사람들은 뒤통수치니까 믿으면 안 된다")