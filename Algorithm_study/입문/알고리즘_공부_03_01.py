"""문제 설명
문자열 배열 strlist가 매개변수로 주어집니다. strlist 각 원소의 길이를 담은 배열을 retrun하도록 solution 함수를 완성해주세요.

제한사항
1 ≤ strlist 원소의 길이 ≤ 100
strlist는 알파벳 소문자, 대문자, 특수문자로 구성되어 있습니다.
입출력 예
strlist	result
["We", "are", "the", "world!"]	[2, 3, 3, 6]
["I", "Love", "Programmers."]	[1, 4, 12]
입출력 예 설명
입출력 예 #1

["We", "are", "the", "world!"]의 각 원소의 길이인 [2, 3, 3, 6]을 return합니다.
입출력 예 #2

["I", "Love", "Programmers."]의 각 원소의 길이인 [1, 4, 12]을 return합니다."""



def solution(strlist):
    length = []
    
    for w in strlist :
        length.append(len(w))
    return


print(solution(["we","ok"]))




"""문제 설명
문자열 my_string이 매개변수로 주어집니다. my_string을 거꾸로 뒤집은 문자열을 return하도록 solution 함수를 완성해주세요.

제한사항
1 ≤ my_string의 길이 ≤ 1,000
입출력 예
my_string	return
"jaron"	"noraj"
"bread"	"daerb"
입출력 예 설명
입출력 예 #1

my_string이 "jaron"이므로 거꾸로 뒤집은 "noraj"를 return합니다.
입출력 예 #2

my_string이 "bread"이므로 거꾸로 뒤집은 "daerb"를 return합니다."""


def solution(my_string):
    return my_string[::-1]


def solution(my_string):
    strin = ""  # 빈 문자열로 초기화
    for char in my_string:
        strin += char  # 문자열로 연결
    return strin

# 예시 입력에 대한 테스트
print(solution("jaron"))  # 출력: "noraj"
print(solution("bread"))  # 출력: "daerb"
