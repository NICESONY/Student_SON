"""대문자와 소문자

문자열 my_string이 매개변수로 주어질 때, 
대문자는 소문자로 소문자는 대문자로 변환한 문자열을 return하도록 solution 함수를 완성해주세요.


my_string	result
"cccCCC"	"CCCccc"
"abCdEfghIJ"	"ABcDeFGHij"
"""


def solution(my_string):
    answer = ''
    for i in my_string :
        if i.isupper():
            answer += i.lower()
        else :
            answer += i.upper()
    return answer


""" 배운점 isupper() 내장 함수와 upper(), lower() 내장함수를 이해할 수 있고 사용할 수 있어야함"""



######################

"""정수 배열 array가 매개변수로 주어질 때, 가장 큰 수와 그 수의 인덱스를 담은 배열을 return 하도록 
solution 함수를 완성해보세요.


제한사항

1 ≤ array의 길이 ≤ 100
0 ≤ array 원소 ≤ 1,000
array에 중복된 숫자는 없습니다.

입출력 예

array	result
[1, 8, 3]	[8, 1]
[9, 10, 11, 8]	[11, 2]"""


def solution(array):
    max_box = []
    array_max = array
    array_max.sort()
    max_box.append(array_max[-1])
    for i , value in enumerate(array) :
        if value == max_box[0]:
            max_box.append(i)
    return max_box


print(solution([1, 8, 3]))