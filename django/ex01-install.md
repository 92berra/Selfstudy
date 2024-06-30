# Django Commands

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

<div align='center'>
92 berra ©2024
</div>