def solution(n):
    k = 0
    for i in range(n+1):
        if i % 2 == 0 :
            k = k+i

    return k

print(solution(4))



def solution(n):
    return sum(i for i in range(2, n + 1, 2))


print(solution(4))


def solution(n):
    return sum([i for i in range(2, n + 1, 2)])


print(solution(4))