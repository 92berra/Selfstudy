# Install

## Homebrew

<a href='https://brew.sh/ko/'>URL</a>

```
vi ~/.zshrc
```

```
export PATH=/opt/homebrew/bin:$PATH
```

```
source ~/.zshrc
```

<li>Homebrew URL 로 들어가서 명령어 복사 후 터미널에 붙여넣기 > 아래 세 개 명령어를 통한 zshrc 에 경로 추가하여 brew 명령어를 쓸 수 있도록 설정</li>

<br/>
<br/>
<br/>
<br/>

## Anaconda 

```
brew install --cask anaconda
export PATH="/opt/homebrew/anaconda3/bin:$PATH"
source ~/.zshrc
conda init zsh
conda update -n root conda
```

```
conda info --envs
```

<br/>
<br/>
<br/>
<br/>

## React

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install node
node -v
npm -v
```

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
<br/>

## Django
#### Environment Setting

```
conda create -n django python=3.9
conda activate django
```

<br/>

#### Install

```
pip install django
python -m django --version
```

<br/>

#### Create django project

```
django-admin startproject {project name}
```

<br/>

#### Settings.py

```
# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/

LANGUAGE_CODE = 'ko-kr'
TIME_ZONE = 'Asia/Seoul'
USE_I18N = True
USE_TZ = False
```

<br/>

#### Varification

```
cd {project directory}
python manage.py runserver
```

<li>https://127.0.0.1:8000</li>

<br/>
<br/>
<br/>
<br/>

## Tensorboard

```
conda activate tensorflow
tensorboard --logdir pbtxt-path
```
<li>http://localhost:6006/</li>

<br/>
<br/>
<br/>
<br/>

<div align='center'>
92berra ©2024
</div>