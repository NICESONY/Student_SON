"""문제 설명
어떤 자연수를 제곱했을 때 나오는 정수를 제곱수라고 합니다. 
정수 n이 매개변수로 주어질 때, 
n이 제곱수라면 1을 아니라면 2를 return하도록 solution 함수를 완성해주세요."""



# import math

# def solution(n):
#     # 주어진 수의 제곱근을 구합니다.
#     root = math.sqrt(n)
#     print(root)
#     # 제곱근이 정수인지 확인하여 제곱수 여부를 판단합니다.
#     if root == int(root):
#         return 1  # 제곱수
#     else:
#         return 2  # 제곱수가 아님

# # 예시로 16을 제곱수인지 확인합니다.
# n = 15
# result = solution(n)
# print(result)




def solution(n):
    # n이 1보다 작으면 2를 반환합니다.
    if n < 1:
        return 2
    
    # 주어진 수를 1부터 차례대로 제곱하여 비교합니다.
    for i in range(1, n+1):
        if i * i == n:
            return 1  # 제곱수
        elif i * i > n:
            break  # 주어진 수를 넘어선 경우 반복문을 종료합니다.
    
    return 2  # 제곱수가 아님

# 예시로 16을 제곱수인지 확인합니다.
n = 16
result = solution(n)
print(result)

