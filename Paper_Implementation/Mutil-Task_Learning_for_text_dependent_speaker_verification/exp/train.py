import torch


device = torch.device('cuda') if torch.cuda.is_available(
) else torch.device('cpu')  # GPU 할당


# 에포크 설정
# num_epochs = 10
# 배치 사이즈 설정
# batch_size = 100  # 여기서 질문 :  여기서 에포크랑 밴치를 주는 게 맞을까? 아니면 어떻게 해야할지??  : 스스로가 찾은 내 대답은 여기가 맞음


#criterion = torch.nn.CrossEntropyLoss().to(device)


def train(model, optimizer, train_loader, device, epoch, scheduler=None):
    # global criterion # , num_epochs

    best_acc = 0

    # 에포크 설정
    model.train()  # 모델 학습
    #running_loss = 0.0

    for b, (k, x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()  # 배치마다 optimizer 초기화


        # # Data -> Model -> Output
        # logit = model(x)  # 예측값 산출
        # #loss = criterion(logit, label)  # 손실함수 계산
        # loss = torch.nn.functional.cross_entropy(logit, y)
        # # loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, label)


        # Data -> Model -> Output
        logit_task1, logit_task2 = model(x)  # Task 1, Task 2의 예측값 산출

        loss_task1 = torch.nn.functional.cross_entropy(logit_task1, y)  # Task 1의 손실 계산
        loss_task2 = torch.nn.functional.cross_entropy(logit_task2, y)  # Task 2의 손실 계산


        # 두 작업의 손실을 결합하여 최종 손실 계산
        loss = loss_task1 + loss_task2

        # 역전파
        loss.backward()  # 손실함수 기준 역전파
        optimizer.step()  # 가중치 최적화
        #running_loss += loss.item()

        #print('[%d] Train loss: %.10f' % (epoch+1, running_loss / len(train_loader)))
        print(f'\rEpoch {epoch:3d}\tloss: {loss.item():.4f}\t{b+1:3d} / {len(train_loader):3d}', end='\t')

        if scheduler is not None:
            scheduler.step()

            
    """
    # Validation set 평가
    model.eval()  # evaluation 과정에서 사용하지 않아야 하는 layer들을 알아서 off 시키도록 하는 함수
    val_loss = 0.0
    correct = 0
    
    with torch.no_grad():  # 파라미터 업데이트 안하기 때문에 no_grad 사용
        for wav, label in dev_loader:
            wav, label = wav.to(device), torch.tensor([int(x) for x in label]).to(device)
            logit = model(wav)
            val_loss += criterion(logit, label)
            pred = logit.argmax(dim=1, keepdim=True)  # 10개의 class 중 가장 값이 높은 것을 예측 label로 추출
            correct += pred.eq(label.view_as(pred)).sum().item()  # 예측값과 실제값이 맞으면 1, 아니면 0으로 합산
            
    val_acc = 100 * correct / len(dev_loader.dataset)
    print('dev set: Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(val_loss / len(dev_loader), correct, len(dev_loader.dataset), val_acc))
    
    # Best 모델 저장
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print('Best model saved.')
    """
