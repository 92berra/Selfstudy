# Install

#### Homebrew

<a href='https://brew.sh/ko/'>URL</a>

```
$ vi ~/.zshrc
$ export PATH=/opt/homebrew/bin:$PATH
$ source ~/.zshrc
```

<li>Homebrew URL 로 들어가서 명령어 복사 후 터미널에 붙여넣기 > 아래 세 개 명령어를 통한 zshrc 에 경로 추가하여 brew 명령어를 쓸 수 있도록 설정</li>

<br/>

#### Anaconda 

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

#### Tensorflow 

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

#### Varification

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
