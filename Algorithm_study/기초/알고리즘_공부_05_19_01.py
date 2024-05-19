

"""문자열 섞기

문제 설명
길이가 같은 두 문자열 str1과 str2가 주어집니다.

두 문자열의 각 문자가 앞에서부터 
서로 번갈아가면서 한 번씩 등장하는 문자열을 만들어 return 하는 solution 함수를 완성해 주세요.



입출력 예
str1	str2	result
"aaaaa"	"bbbbb"	"ababababab"
"""





def solution(str1, str2):
    new_string = ''
    for i in range(len(str1)):
        new_string += str1[i] + str2[i]
    return new_string

