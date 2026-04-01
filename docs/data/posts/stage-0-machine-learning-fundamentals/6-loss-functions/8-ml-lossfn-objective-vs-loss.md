# 8. Чем objective function отличается от loss function?

Loss function измеряет **ошибку предсказаний модели**.

Objective function — это более общая функция, которую алгоритм обучения оптимизирует.

## Loss function

Loss показывает, насколько prediction отличается от target.

## Objective function

Objective function может включать:

- loss function  
- регуляризацию  
- дополнительные ограничения

## Пример

objective = loss + λ * regularization

## Основное различие

Loss — это часть objective function.

Objective function — это то, что оптимизируется во время обучения.

## Итог

Objective function — это функция оптимизации, которая обычно включает loss function.