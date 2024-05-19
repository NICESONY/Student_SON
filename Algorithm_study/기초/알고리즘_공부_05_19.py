"""배열 만들기 2


정수 l과 r이 주어졌을 때, l 이상 r이하의 정수 중에서 숫자 "0"과 "5"로만 이루어진 모든 정수를 오름차순으로 저장한 배열을 return 하는 solution 함수를 완성해 주세요.

만약 그러한 정수가 없다면, -1이 담긴 배열을 return 합니다.

입출력 예
l	r	result
5	555	[5, 50, 55, 500, 505, 550, 555]
10	20	[-1]

"""

from collections import deque

def solution(l, r):
    queue = deque([5])
    results = []
    
    while queue:
        num = queue.popleft()
        
        if num > r:
            continue
        
        if l <= num <= r:
            results.append(num)
        
        # 새 숫자를 큐에 추가
        queue.append(num * 10)
        queue.append(num * 10 + 5)
    
    results = sorted(results)
    
    if not results:
        return [-1]
    
    return results

# 예제 테스트
print(solution(5, 555))  # [5, 50, 55, 500, 505, 550, 555]
print(solution(10, 20))  # [-1]



