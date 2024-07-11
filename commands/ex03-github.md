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
git push -u origin main
```

<br/>
<br/>

#### commit 

```
git reset HEAD~0
git reset HEAD~1
git reset HEAD~2

git status
git push -u origin main

git pull origin main

git add .
git status

git commit -m "commit message"
git push -u origin main

git pull origin main
git push -u origin main
```

<br/>
<br/>

#### Generate Branch

```
git branch dev
git checkout dev
git merge dev
```
<br/>
<br/>
<br/>
<br/>

<div align='center'>
92berra ©2024
</div>