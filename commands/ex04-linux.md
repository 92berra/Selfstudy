# Linux commands

### 현재 디렉토리 파일 갯수

```
ls -l | grep ^- | wc -l
```

- ls -l : 현재 파일, 폴더 목록 출력
- grep ^- : 폴더 제외
- wc -l : 라인 수 세기

<br/>

### 특정 디렉토리 내 몇 개의 파일만 복사하기

```
ls source_directory | head -n 19 | xargs -I {} cp source_directory/{} destination_directory/
ls ../Decompose/datasets/fonts/target | head -n 19 | xargs -I {} cp ../Decompose/datasets/fonts/target/{} data/ttf/validation
```

- ls source_directory: source_directory에 있는 파일 리스트 출력
- head -n 19: 출력된 파일 리스트에서 첫 19개의 파일만 선택
- xargs -I {} cp source_directory/{} destination_directory/: 선택된 각 파일을 source_directory에서 destination_directory로 복사

<br/>

```
ls source_directory | tail -n +20 | xargs -I {} cp source_directory/{} destination_directory/
ls ../Decompose/datasets/fonts/target | tail -n +20 | xargs -I {} cp ../Decompose/datasets/fonts/target/{} data/ttf/train
```

- ls source_directory: source_directory에 있는 파일 리스트 출력
- tail -n +20: 파일 리스트에서 첫 19개의 파일을 제외하고 나머지 파일을 선택합니다. tail -n +20은 20번째 파일부터 끝까지 출력
- xargs -I {} cp source_directory/{} destination_directory/: 선택된 각 파일을 source_directory에서 destination_directory로 복사