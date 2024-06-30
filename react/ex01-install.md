# Install 

#### Install

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install node
node -v
npm -v
```

<br/>

#### Create project and Install required module

```
npx create-react-app {project-name}
cd {project-name}
npm install -g gh-pages --save-dev
```

<br/>

#### Commit and Push

```
git init 
git add .
git commit -m "first commit"
git branch -M main
git remote add origin {Repository Remote URL} 
git push -u origin main
```

<br/>

#### Edit packages.json

```
# scripts
"deploy": "gh-pages -d build"
```

```
# scripts 와 같은 레벨로 homepages 항목 추가
"homepage": "https://{username}.github.io",
```

<br/>

#### Apply deploy

```
npm run build
npm run deploy
```

<br/>

#### Modify git branch

Github profile > github Pages > Branch > gh-pages

<br/>
<br/>
<br/>

<div align='center'>
92 berra ©2024
</div>
