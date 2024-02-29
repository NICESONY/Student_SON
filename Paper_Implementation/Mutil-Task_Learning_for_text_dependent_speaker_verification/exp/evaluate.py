import torch


def predict_eval_module(model, test_loader,device):
    model.eval()  # 모델을 평가 모드로 설정
    #total_loss = 0.0
    correct_task1 = 0
    correct_task2 = 0
    # correct = 0
    total = 0

    with torch.no_grad():  # 그라디언트 계산 비활성화
        for k, x, y in test_loader:
            x = x.to(device)
            # y = y.to(device)


            logit_task1, logit_task2 = model(x)  # Task 1, Task 2의 예측값 산출

            _, pred_indices_task1 = torch.max(logit_task1, 1) 
            _, pred_indices_task2 = torch.max(logit_task2, 1)  


            total += y.size(0)
            correct_task1 += (pred_indices_task1.cpu() == y).sum().item()
            correct_task2 += (pred_indices_task2.cpu() == y).sum().item()

    accuracy_task1 = correct_task1 / total
    accuracy_task2 = correct_task2 / total

    print(f"Task 1 Accuracy: {accuracy_task1:.4f}", end = "\n")
    print(f"Task 2 Accuracy: {accuracy_task2:.4f}", end = "\n")







            # logit = model(x)
            # _, pred_indices = torch.max(logit, 1)

            #total_loss += torch.nn.functional.cross_entropy(logit, labels, reduction='sum').item()
    #         total += y.size(0)
    #         correct += (pred_indices.cpu() == y).sum().item()

    # #avg_loss = total_loss / total
    # accuracy = correct / total

    # # return avg_loss , accuracy

    # #print(f"평균 손실: {avg_loss:.4f}")
    # print(f"accuracy: {accuracy:.4f}", end='\t')











# def predict_eval_module(model, eva_loader):
#     model.eval()
#     model_pred = []
#     with torch.no_grad():
#         for wav in eva_loader:
#             # wav = wav
#             wav = wav  # 테스트 해보자 이렇게 만들면 to 사용할 수 있다고 함.
#             # 여기서 라벨도 같이 불러오는 것이 맞을까? 아닐까? 
#             pred_logit = model(wav)
#             pred_logit = pred_logit.argmax(dim=1, keepdim=True).squeeze(1)

#             model_pred.extend(pred_logit.tolist())
#     return model_pred
#     # 다음으로 바꿀 것은 y값을 None으로 주고 해보자 => 실패  
    