"""문자열 my_string이 매개변수로 주어집니다. 
my_string안의 모든 자연수들의 합을 return하도록 solution 함수를 완성해주세요"""

""" 새롭게 배운 것 

isdigit()를 사용하면 str()로 입력된 문자 중에서 숫자를 찾을 수 있음"""

""" 1. 문자열만 포함되어있는지 확인하기
문자열의 구성이 알파벳 or 한글인지 확인하는 방법 [isalpha]
isalpha()라는 내장함수 사용, 단 문자열에 공백,기호 and 숫자가 있을시 False를 리턴한다"""


"""2. 숫자인지 확인하는 방법 [isdigit]
위와 비슷하게 isdigit()라는 내장함수를 사용하여 확인한다."""

"""3. 알파벳(한글) 또는 숫자인지 확인하는법[isalnum]
isalnum()라는 내장함수를 사용하여 확인한다."""


def solution(my_string):
    num_sum = 0
    for i in my_string :
        if i.isdigit():
            num_sum += int(i)
    return num_sum    

print(solution("1a2b3c4d123"))














