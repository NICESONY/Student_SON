import torch
import numpy as np


class CustomDataset(torch.utils.data.Dataset):
	def __init__(self, key2mfcc: dict, label_fn, target_length: int, transforms=None):  # 필요한 변수들을 선언		
		self.keys = sorted(list(key2mfcc.keys()))

		self.key2mfcc = key2mfcc
	

		#self.key2label = {}
		#for key in self.keys:
		#	label = label_fn(key)
		#	self.key2label[key] = label

		self.key2label = {k: label_fn(k) for k in self.keys}

		'''
		[k for k in self.keys]
		==
		a = []
		for k in self.keys:
			a.append(k)
		'''

		if transforms is not None:
			for key in self.keys:
				self.key2mfcc[key] = transforms(self.key2mfcc[key])


		for key in self.keys:
			mfcc = self.key2mfcc[key]
			D, T = mfcc.shape

			if T < target_length:
				p = target_length - T
				mfcc = np.pad(mfcc, ((0, 0), (0, p)), mode = "wrap" )
				self.key2mfcc[key] = mfcc
				
			else:
				self.key2mfcc[key] = mfcc[ : , : target_length]


		for key in self.keys:
			self.key2mfcc[key] = np.expand_dims(self.key2mfcc[key], 0)


		"""
		self.key2mfcc = {}
		self.key2label = {}

		self.key_list = self.key2mfcc.keys()
		"""

	def __getitem__(self, index):  # index번째 data를 return
		"""
		key = self.key_list[index]
		mfcc = self.key2mfcc[key]
		label = self.key2label[key]


		mfcc = set_frames(mfcc, target_frames)

		return mfcc, label
		"""
		"""
		X_i = self.X[index]
		y_i = self.y[index]

		if self.transforms is not None:
			X_i = self.transforms(X_i)

		return X_i, y_i
		"""
		key = self.keys[index]
		mfcc = np.ascontiguousarray(self.key2mfcc[key], dtype=np.float32)
		label = self.key2label[key]

		return key, mfcc, label



	def __len__(self) -> int:
		return len(self.keys)
