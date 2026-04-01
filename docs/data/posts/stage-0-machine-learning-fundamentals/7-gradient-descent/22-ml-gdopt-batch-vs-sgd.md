# 22. Чем batch gradient descent отличается от stochastic gradient descent?

Главное различие между batch gradient descent и stochastic gradient descent заключается в **количестве данных, используемых для одного обновления параметров**.

## Batch gradient descent

Использует **весь dataset** для вычисления градиента.

Особенности:

- точный градиент  
- стабильные обновления  
- медленные итерации при больших данных

## Stochastic gradient descent

Использует **только один пример** для обновления параметров.

Особенности:

- быстрые обновления  
- шумные градиенты  
- более частые обновления параметров

## Главное различие

Batch gradient descent:

обновление после обработки всех данных

Stochastic gradient descent:

обновление после каждого примера

## Итог

Batch gradient descent использует весь dataset, а stochastic gradient descent — один пример за обновление.