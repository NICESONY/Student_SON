import torch
import numpy as np


class CustomDataset(torch.utils.data.Dataset):
	def __init__(self, X, y, transforms = None):  # 필요한 변수들을 선언		
		self.X = X.reshape(-1, 1, X.shape[1], X.shape[2] )
		self.y = y
		self.transforms = transforms


	def __getitem__(self, index): #index번째 data를 return
		X_i = self.X[index]
		y_i = self.y[index]

		if self.transforms is not None:
			X_i = self.transforms(X_i)

		return X_i, y_i
			

	def __len__(self) -> int:
		return len(self.X)
