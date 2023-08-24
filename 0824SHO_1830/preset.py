class Config:
    n_samples = 1000
    num_layers = 3  # 실제 레이어의 수. 코드의 for문에서 -1을 이미 적용함
    hidden_dim = 32
    learning_rate = 0.005
    epochs = 20
    batch_size = 70  # 배치 사이즈
    model_try = 3  # 해당 값으로 트라이할 모델 수
    min_epoch = 500  # model당 최소 epoch. 해당 값 이전까지는 stop하지않음
    ReLU_On = False  # True 적용시 레이어의 ReLU 활성화
    cuda_On = False  # cuDNN 설치 전까지 False 사용. 혹시 사용할 수도 있으니 다들 사전에 설치하면 좋겠음
    patience = 1  # 이 epoch동안 val_loss 기록이 단 한 번도 개선되지 않으면 iteration을 종료
    scaler = 1  # quiver scale 조정 값
    amplification_factor = 5  # 증폭계수 적용
