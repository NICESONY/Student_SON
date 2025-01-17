"""전화번호 목록

문제 설명
전화번호부에 적힌 전화번호 중, 한 번호가 다른 번호의 접두어인 경우가 있는지 확인하려 합니다.
전화번호가 다음과 같을 경우, 구조대 전화번호는 영석이의 전화번호의 접두사입니다.

구조대 : 119
박준영 : 97 674 223
지영석 : 11 9552 4421
전화번호부에 적힌 전화번호를 담은 배열 phone_book 이 solution 함수의 매개변수로 주어질 때, 어떤 번호가 

다른 번호의 접두어인 경우가 있으면 false를 그렇지 않으면 true를 return 하도록 solution 함수를 작성해주세요.



입출력 예제
phone_book	return
["119", "97674223", "1195524421"]	false
["123","456","789"]	true
["12","123","1235","567","88"]	false

"""


def solution(phone_book):

    phone_book.sort()
    
    for i in range(len(phone_book) - 1):
        if phone_book[i+1].startswith(phone_book[i]): ## phone_book[i+1]이 phone_book[i]의 접두어면 startswith로 인해서 true
            return False
    return True


## 아래 풀이로 하면 시간 초과로 안돼는 것이 존재함

def solution(phone_book):
    phone_book = "".join(phone_book)
    hash_map = {}
    for i in phone_book :
        if i in hash_map :
            hash_map[i] += 1
        else :
            hash_map[i] = 1
            
    for key in hash_map :
        if hash_map[key] != 1 :
            return False
        else :
            return True
