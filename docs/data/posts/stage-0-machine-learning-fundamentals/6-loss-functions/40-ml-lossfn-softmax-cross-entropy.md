# 40. Что такое softmax cross entropy?

Softmax cross entropy — это функция потерь, которая объединяет две операции:

1. softmax  
2. cross entropy

## Основная идея

Сначала logits преобразуются в вероятности с помощью softmax.

Затем вычисляется cross entropy loss.

## Почему это объединяют

Многие библиотеки объединяют эти операции в одну функцию.

Это:

- быстрее  
- численно стабильнее

## Пример

В deep learning фреймворках часто используется:

softmax_cross_entropy_with_logits

## Итог

Softmax cross entropy — это функция потерь для multiclass classification, которая работает напрямую с logits.