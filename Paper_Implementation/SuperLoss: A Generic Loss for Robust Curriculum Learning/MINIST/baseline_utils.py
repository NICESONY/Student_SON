import numpy as np
import matplotlib.pyplot as plt
import torch




def torch_seed(seed = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_determinstic_algoriths = True







def evaluate_history(history):
    print(f"inital state val_loss : {history[0,2]:.5f}"
          ,f"inital stast val_acc :  {history[0,4] :.5f}")
    print(f"final state val_loss : {history[-1,2]:.5f}"
          ,f"final stast val_acc :  {history[-1,4] :.5f}")

    # 최고 정확도와 최소 손실 찾기
    best_accuracy = max(history[:, 4])
    best_loss = min(history[:, 2])
    print(f"Best accuracy : {best_accuracy:.5f}")
    print(f"Best loss : {best_loss:.5f}")


    num_epochs = len(history)
    unit = num_epochs / 10
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,1], "b",label = "train")
    plt.plot(history[:, 0], history[:, 3], "k", label="val")
    plt.xticks(np.arange(0, num_epochs+1, unit))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("learning curve(loss)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(9,8))
    plt.plot(history[:, 0], history[:,2], "b", label = "train")
    plt.plot(history[:, 0], history[:, 4], "k", label = "val")
    plt.xticks(np.arange(0, num_epochs+1 , unit))
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("learning curve(acc)")
    plt.legend()
    plt.show()
