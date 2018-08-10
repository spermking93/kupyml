Github를 왜 사용해야 할까요?
1. 버전 업데이트 용이 
2. 소스코드 여러번 아니고 한 번만 배포하면 됨 

Github에서 하는 작업은 똑같은 코드를 동시에 고치는 것은 아님. 여러 개의 파일을 파트로 따로 맡아서 올리고 merge하는 것. 
Sourcetree에서 맨 왼쪽 선 - 마스터 브렌치  
마스터 브렌지에 연결되어 튀어 나오는 선: 새로운 브렌치 
Why 새로운 브렌치? 
a—>a’으로 시도해서 잘 작동하면 기존의 마스터 브렌치 project로 merge하게 됨. 동시에 똑같은 작업을 하고 있을 때 브렌치가 발생. 

명령어
Git clone url(복사한것)  
—> repository 다운로드 
Git add filename 
—> 어떤 파일을 올릴지 명시해주기(업로드 하고 싶은 파일 골라서 선택해놓기) 
Git commit -m “설명” 
—> 파일에 대한 설명 
Git push  
—> 실제로 파일을 업로드 하는 것 
Git pull  
—>중앙 저장소에서 업데이트 코드를 가져옴 

개인 프로젝트에서도 브렌치를 할 수 있다. 어떤 코드가 더 성능이 좋은지 비교하기 위해서.  
개인 프로젝트 하는 식으로 예쁘게 꾸미기 추천. 
남의 GitHub도 복사해올 수 있다.(fork) 

Wiki에는 공부하다가 기억하고 싶은 이론들 넣어놓기 
Issues - bug들 고쳐야할 to do list처럼 올리기, 남의 작업 보고 버그 보고하기 - 커뮤니티 페이지 
github를 보면 문제해결 능력을 알수 있음 

Sourcetree 사용법
체크박스 누르면 commit 시작. 다 적고 커밋 누르면 푸시 됨. 
빨간색라인은 기존의 파일에서 내가 없앤 것. 

Original branch 는 release용, 새 branch는 develop용.  
Release용(Original branch)은 완전히 돌아가는 코드만 올리고, 새 branch는 시험, 디버그용. 

Tensorflow 쓸 때, colab쓰는게 편할 듯. 자동 재생도 되고, 주석 달기도 쉬움. 
