"""
완주하지 못한 선수


수많은 마라톤 선수들이 마라톤에 참여하였습니다. 단 한 명의 선수를 제외하고는 모든 선수가 마라톤을 완주하였습니다.

마라톤에 참여한 선수들의 이름이 담긴 배열 participant와 완주한 선수들의 이름이 담긴 배열 completion이 주어질 때,
 완주하지 못한 선수의 이름을 return 하도록 solution 함수를 작성해주세요.



입출력 예
participant	completion	return
["leo", "kiki", "eden"]	["eden", "kiki"]	"leo"
["marina", "josipa", "nikola", "vinko", "filipa"]	["josipa", "filipa", "marina", "nikola"]	"vinko"
["mislav", "stanko", "mislav", "ana"]	["stanko", "ana", "mislav"]	"mislav"
 """



def solution(participant, completion):
    hash_map = {}
    for p in participant :
        if p in hash_map :
            hash_map[p] += 1
        else :
            hash_map[p] = 1
            
    for c in completion:
        if c in hash_map :
            hash_map[c] -= 1
            
    for key in hash_map :
        if hash_map[key] != 0 :
            return key
            
            