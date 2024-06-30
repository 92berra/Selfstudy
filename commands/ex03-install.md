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

## Tensorflow 

```
python3 -m pip install tensorflow
python3 -m pip install tensorflow-metal
```

```
import tensorflow as tf
tf.__version__
tf.config.list_physical_devices('GPU')
quit()
```

<li>tf.config.list_physical_devices('GPU') 실행 결과 [ ] 만 뜨면 GPU 사용이 불가한 상태이다. 아래 코드를 test.py 로 저장 해 실행하면 훈련이 되는 것을 확인할 수 있다. 여기까지 된다면 성공적으로 설치된 것이다. </li>

<br/>
<br/>

#### Tensorflow Installation Varification

```
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
     tf.keras.layers.Flatten(input_shape=(28, 28)),
     tf.keras.layers.Dense(128, activation='relu'),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam',
     loss='sparse_categorical_crossentropy',
     metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)
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
92 berra ©2024
</div>