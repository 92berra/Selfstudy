# GitHub Trouble Shooting

#### error: RPC failed; curl 18 transfer closed with outstanding read data remaining

```
git clone https://github.com/repo --depth 1
cd repo
git fetch --unshallow
```

- 인터넷 연결이 불안정해서 생긴 문제
- 점진적으로 가져오기 방식으로 clone 하면 해결 가능

<br/>
<br/>

#### Apply .gitignore

```
git rm -r --cached .
git add .
git commit -m "Apply .gitignore"
```

<br/>
<br/>
<br/>
<br/>

<div align='center'>
92berra ©2024
</div>