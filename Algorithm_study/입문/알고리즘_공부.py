'''문제 설명
정수 n이 주어질 때, n이하의 짝수를 모두 더한 값을 return 하도록 solution 함수를 작성해주세요.

제한사항
0 < n ≤ 1000

입출력 예
n	result
10	30
4	6
입출력 예 설명
입출력 예 #1

n이 10이므로 2 + 4 + 6 + 8 + 10 = 30을 return 합니다.
입출력 예 #2

n이 4이므로 2 + 4 = 6을 return 합니다.'''

### 1
def solution(n):
    k = 0
    for i in range(n+1):
        if i % 2 == 0 :
            k = k+i
    return k
print(solution(4))

### 2
def solution(n):
    return sum(i for i in range(2, n + 1, 2))
print(solution(4))

### 3
def solution(n):
    return sum([i for i in range(2, n + 1, 2)])
print(solution(4))





'''문제 설명
정수 배열 numbers가 매개변수로 주어집니다. numbers의 원소의 평균값을 return하도록 solution 함수를 완성해주세요.

제한사항
0 ≤ numbers의 원소 ≤ 1,000
1 ≤ numbers의 길이 ≤ 100
정답의 소수 부분이 .0 또는 .5인 경우만 입력으로 주어집니다.
입출력 예
numbers	result
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]	5.5
[89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]	94.0
입출력 예 설명
입출력 예 #1

numbers의 원소들의 평균 값은 5.5입니다.
입출력 예 #2

numbers의 원소들의 평균 값은 94.0입니다.'''

### 1
def solution(numbers):
    l = len(numbers)
    s = sum(numbers)
    r = s / l
    return r

### 2
import numpy as np
def solution(numbers):
    return np.mean(numbers)




'''정수가 들어 있는 배열 num_list가 매개변수로 주어집니다. num_list의 원소의 순서를 거꾸로 뒤집은 배열을 return하도록 solution 함수를 완성해주세요.

제한사항
1 ≤ num_list의 길이 ≤ 1,000
0 ≤ num_list의 원소 ≤ 1,000
입출력 예
num_list	result
[1, 2, 3, 4, 5]	[5, 4, 3, 2, 1]
[1, 1, 1, 1, 1, 2]	[2, 1, 1, 1, 1, 1]
[1, 0, 1, 1, 1, 3, 5]	[5, 3, 1, 1, 1, 0, 1]
입출력 예 설명
입출력 예 #1

num_list가 [1, 2, 3, 4, 5]이므로 순서를 거꾸로 뒤집은 배열 [5, 4, 3, 2, 1]을 return합니다.
입출력 예 #2

num_list가 [1, 1, 1, 1, 1, 2]이므로 순서를 거꾸로 뒤집은 배열 [2, 1, 1, 1, 1, 1]을 return합니다.
입출력 예 #3

num_list가 [1, 0, 1, 1, 1, 3, 5]이므로 순서를 거꾸로 뒤집은 배열 [5, 3, 1, 1, 1, 0, 1]을 return합니다.'''

## 1
def solution(num_list):
    answer = num_list[: : -1]
    return answer

print(solution([1,2,3]))



## 2
def solution(num_list):
    num_list.reverse()
    return num_list



## 3
from collections import deque
def solution(num_list):
    answer = deque([])
    for i in num_list:
        answer.appendleft(i)
    return list(answer)
