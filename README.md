# quick_draw_classifier
## Постановка задачи
### Описание задачи
В качестве задачи компьютерного зрения была выбрана задача классификации рукописных изображений, представленная набором данных The Quick Draw dataset – коллекция рисунков пользователей игры [Quick, Draw!](https://quickdraw.withgoogle.com/), содержащая 345 категорий изображений, из которых для лабораторной работы были взяты следующие 5 категорий по 5 000 примеров:
 - spoon - ложка
 - paintbrush - кисть
 - smiley face - улыбающееся лицо
 - wheel - колесо
 - bush - куст
 
Рисунки представлены набором точек – `(x, y, t)` - вектор координат пикселей на плоскости со значением времени первой точки и разметкой, включающей в себя информацию о стране игрока и о том, что требовалось нарисовать.

### Выбор библиотеки
Для выполнения данной практической работы была выбрана библиотека глубокого обучения `TensorFlow`, использующая в качестве интерфейса язык программирования `Python`.

Для проверки корректности установки библиотеки была выполнена разработка и запуск тестового примера сети для задачи классификации рукописных цифр из набора данных `MNIST`, достигнутая точность на котором равна `0,931`.

### Тренировочные и тестовые наборы данных 
В используемом наборе данных `25 000 = 5 категорий х 5 000 примеров` изображений, из которых использовали как:
 - тренировочных – 17 500 (70%);
 - валидационных – 2 500 (10%);
 - тестовых – 5 000 (20%)

В ходе обучения наборы тщательно перемешиваются перед выборками.

### Метрики качества решения задачи 
Качество решения выбранной задачи оценивается с использованием различных метрик:
- `accuracy` – точность – это отношение числа верно классифицированных изображений к общему числу изображений в выборке

$$точность={верноклассифицированные\over всеизображения}$$

Введем некоторые обозначения для определения следующих величин:
 - `TP` — истино-положительное решение
 - `TN` — истино-отрицательное решение
 - `FP` — ложно-положительное решение
 - `FN` — ложно-отрицательное решение

тогда можем определить вычисление следующих метрик по формулам:
 - `precision` – точность – это доля изображений действительно принадлежащих данному классу относительно количества всех изображений, которые сеть отнесла к этому классу
 
$$precision = {TP\over {TP + FP}}$$

 - `recall` – полнота – это доля найденных сетью изображений, принадлежащих классу относительно количества всех изображений этого класса в тестовой выборке
 
$$recall = {TP\over {TP + FN}}$$

 - `f1-score` – F-мера – это гармоническое среднее между точностью и полнотой
 
$$F = 2 * {{precision * recall}\over {precision + recall}}$$

### Формат хранения данных
Исходный формат хранения данных - бинарные файлы

	https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/binary 

 Атрибут    | Тип                     | Описание 
------------|-------------------------|-------------------------------------------
key_id      | 64-bit unsigned integer | Уникальный идентификатор
word        | string                  | Категория изображения
recognized  | boolean                 | Указано, что слово было опознано в игре
timestamp   | datetime                | Дата создания рисунка
countrycode | string                  | Код страны участника
drawing     | string                  | JSON массив, содержащий вектор рисования

 
Формат данных на входе сети – в ходе проведения экспериментов данные подавались в следующих видах:
 - 3-мерный массив нормализованных координат данных, длины равной максимальной длине набора точек рисунка (недостающие элементы заполнялись нулями);
 - нормализованные бинарные изображения, отмасштабированные по размеру `28х28`, полученные с помощью библиотеки `OpenCV` из данных источника;
 - в ходе подготовки данных все данные разметки были векторизованы.

### Разработанная программа
Программа содержит следующие файлы в директории `src`: 
 - `parse_data.py` - подготовка данных 
 - `run_create_dataset.py` - выбор данных, которые распознала сеть Google, распределение на run, train и validate выборки
 - `run_train_conv.py` - непосредственно обучение сети, использует класс `NetworkBase`
 - `NetworkBase.py` - описание нейронной сети
 - `run_statisctic_conv.py` - получение статистики по итогам обучения
 
## Тестовая инфраструктура
Эксперименты  проводились на машине со следующими характеристиками:
 - Ubuntu 16.04, CUDA 9.1
 - RAM 60 GB
 - Geforce GTX 1060 6 GB x2 Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz. 

## Итоговая таблица
Ниже представлена сводная таблица, содержащая значения самые высокие значения метрики f1-score по проведенным лабораторным работам в курсе

Lab| № experiment | Spoon | Paintbrush | Smiley face | Wheel | Bush | Micro avg | Macro avg | Weighted avg | Accuracy, %  | Cross entropy | Time, min
---|--------------|-------|------------|-------------|-------|------|-----------|-----------|--------------|--------------|---------------|-----------
 2 | 3            | 0.95  | 0.94       | 0.92        | 0.96  | 0.84 | 0.92      | 0.92      | 0.93         | 98           | 0.1           |    77min  
 3 | 3            | 0.94  | 0.94       | 0.94        | 0.93  | 0.92 | 0.91      | 0.91      | 0.91         | 95           | 0.18          |   105min  
 4 | 3            | 0.93  | 0.92       | 0.91        | 0.92  | 0.92 | 0.92      | 0.92      | 0.92         | 97           | 0.18          |   107min  
 5 | 3            | 0.96  | 0.95       | 0.96        | 0.97  | 0.92 | 0.95      | 0.95      | 0.95         | 99           | 0.1           |  3025min  

Наилучший результат был показан в лабораторной работе №5, где использовалась более сложная архитектура нейронной сети при Transfer Learning'е, а именно при переносе архитектуры сети.

# Manuals
Ман по гиту
 
0.Установить GitBash для Windows
Ссылка на скачивание: http://git-scm.com/download/win
При установке все опции ставим по умолчанию.
 
0.Запустить GitBash, и сделать следующее:
·         $git config --global user.email " Your.Name@gmail.com"
·         $git config --global user.name "Your Name"
·         $git config core.autocrlf false
 
1.Cоздать свой собственный fork

2.Cклонировать fork проекта
git clone <project_name> https_или_http_или_ssh: //github.com/<GITHUBLOGIN>/<project_name>.git)

3.Добавить в этот проект remote на свой оригинальный проект
git remote add <project_name> https_или_http_или_ssh://github.com/VictorBebnev/<project_name>.git

4.Зафетчить код с оригинального проекта
git fetch <project_name>

Зафетчить код со своего fork можно командой
git fetch origin

Вообще это действие рекомендуется делать перед каждым действием с чего либо, что содержит слово remote
Если есть какие-то сомнения то лучше сделать

5.Создать бранч на котором вы будите работать от оригинального проекта
git chekout -b <your_br_name> <project_name>/master
или
git chekout -b <your_br_name>  <project_name>/developer
или другой бранч. полный список бранчей  можно посмотреть набрав
git branch -a

6.Делаете свои изменения далее обычная работа с git, до момента push а изменений на сервер
Хорошим стилем является привычка работать на своем собственном бранче.
git branch <branch_name> - создает бранч от текущего состояния текущего бранча.
git checkout <branch_name> - переключается на бранч <branch_name>;
Эти 2 команды эквивалентны команде
git checkout -b <branch_name>
 
git fetch забирает обновления из текущего проекта
git checkout origin/master -b <branch_name> создает бранч от последней версии обновлений бранча master
 
git status - позволяет увидеть список изменных вами фалов
git diff <relative_path_to_file> - позволяет увидеть изменения в указанном файле
git diff - позволяет увидеть изменения во всех изменных файлах
 
git add <relative_path_to_file> - помечает файл(ы) как добавленные/изменённые
git rm <relative_path_to_file> - помечает файлы как удаленные
git mv <old_path_to_file> <new_path_to_file> - помечает файлы как перемещённые
 
git commit – применяет все помеченные изменения.

7.Далее пушите свои изменения на свой форк
git push origin <your_br_name>_или_HEAD:<some_br_name>

8.Теперь надо создать pull request, для этого надо пойти на github в свой форк, и жамкнуть кнопочку Compare & pull request.

Важный момент!
После создания пулреквеста НЕЛЗЯ использовать rebase, commit --amend и push -f если вы хотите внести изменения в pull request, это сотрёт все комментарии к pull request.

 
Ман по Докеру
Установка:
docker
https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce-1

Если необходимо работать с gpu:
gpu driver
http://www.nvidia.ru/Download/index.aspx?
nvidia-docker
https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)

Основные команды(могут быть запрошены права суперпользователя):
docker build -t <имя_докер_репозитория>/<имя_контейнера>:<версия> <путь_до_докерфайла> - собрать образ
пример: docker build -t nvidia/ds_cuda8.0cudnn7:1.0 .?

docker images  - вывести список собранных образов в текущей ОС
замеч.: флаг -a отобразит все образы

docker ps – вывести список запущенных контейнеров
замеч.: флаг -a отобразит все контейнеры
 
docker run  <ключи> <полное_имя_контейнера_или_его_id> - запуск контейнера
пример:  docker run -it -p 10.13.12.108:800:80 -v $pwd:/opt/STT/ --runtime=nvidia --entrypoint "/bin/bash" nvidia/ds_cuda8.0cudnn7:1.0?

часто используемые ключи для запуска:
-it - интерактивная сессия, всегда ставим этот флаг если хотим работать сразу внутри запускаемого контейнера
--rm - удаление контейнера после запуска, всегда ставим этот флаг если хотим освободить ресурсы жесткого диска после остановки контейнера
-p<номер_порта_в_текущей_ос>:<номер_порта_внутри_контейнера> - перенаправление порта текущей ос внутрь контейнера
-v<путь_до_директории_в_текущей_ос>:<путь_до_директории_внутри_контейнера> - подключение директории из текущей ос внутрь контейнера
--runtime=nvidia - указание движка для запуска, необходим при использовании gpu
-d - detach контейнера сразу после запуска (для подключения используем примеры с exec или attach ниже)
--name <name_контейнера> - вместо случайно сгенерируемого имени можно указать своё
замеч.: можно пробросить порт N внутрь контейнера перенаправив его на 22 порт, появится возможность подключится через ssh в порт N сразу в контейнер.

docker ps  ?- вывести список запущенных контейнеров в текущей ОС

Если хотим сделать dettach(отключится от контейнера, но оставить его запущенным) последовательно выполняем: ctrl + p, ctrl + q

docker attach <id_запущенного_контейнера> -  подсключение к запущенному контейнеру

docker exec -it <ID_контейнера>\<name_контейнера> bash – подключиться к контейнеру в режиме командной стоки(новая сессия)

docker stop <ID_контейнера>\<name_контейнера> - остановка контейнера

docker rm <ID_контейнера>\<name_контейнера> - удаление контейнера с жесткого диска

docker logs <ID_контейнера>\<name_контейнера> - выводит логи контейнера