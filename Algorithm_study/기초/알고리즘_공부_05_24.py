"""
flag에 따라 다른 값 반환하기


두 정수 a, b와 boolean 변수 flag가 매개변수로 주어질 때, flag가
 true면 a + b를 false면 a - b를 return 하는 solution 함수를 작성해 주세요.
 
 
 입출력 예
a	b	flag	result
-4	7	true	3
-4	7	false	-11
 """




def solution(a, b, flag):
    if flag == True :
        return a + b
    else :
        return a -b
