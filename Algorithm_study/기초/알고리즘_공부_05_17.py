"""문자열 겹쳐쓰기

문자열 my_string, overwrite_string과 정수 s가 주어집니다. 문자열 my_string의 인덱스 
s부터 overwrite_string의 길이만큼을 문자열 overwrite_string으로 바꾼 
문자열을 return 하는 solution 함수를 작성해 주세요.


my_string	overwrite_string	s	result
"He11oWor1d"	"lloWorl"	2	"HelloWorld"
"Program29b8UYP"	"merS123"	7	"ProgrammerS123"
"""


def solution(my_string, overwrite_string, s):
    idx_string = my_string[: s]
    new = overwrite_string[:]
    add_string = my_string[s + len(overwrite_string) :]
    return idx_string + new + add_string