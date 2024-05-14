"""암호 해독"""

def solution(cipher, code):
    result = ""
    for i in range(code - 1, len(cipher), code):
        result += cipher[i]
    return result


print(solution("dfjardstddetckdaccccdegk", 4))
