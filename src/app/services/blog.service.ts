import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { map, catchError } from 'rxjs/operators';
import { marked } from 'marked';
import hljs from 'highlight.js';
import { ThemeService } from './theme.service';
import { Post, PostMetadata, CategoryTree, SubCategory, PredefinedCategories, PredefinedCategory, PredefinedSubcategory } from '../models/post.interface';

@Injectable({
  providedIn: 'root'
})
export class BlogService {
  private allPostsCache: PostMetadata[] | null = null;

  /** Category order in nav and listings. */
  private readonly studyOrderCategorySlugs: string[] = [
    'stage-0-machine-learning-fundamentals',
    'stage-1-pytorch-engineering',
    'stage-2-gpu-and-performance-engineering',
    'stage-3-distributed-training',
    'stage-4-ml-infrastructure',
    'stage-5-large-model-training',
    'stage-6-advanced-ml-systems'
  ];

  private readonly studyOrderSubcategorySlugs: Record<string, string[]> = {
    'stage-0-machine-learning-fundamentals': [
      '1-fundamentals-of-machine-learning',
      '2-types-of-learning',
      '3-data',
      '4-linear-regression',
      '5-logistic-regression',
      '6-loss-functions',
      '7-gradient-descent',
      '8-overfitting',
      '9-metrics',
      '10-neural-networks'
    ],
    'stage-1-pytorch-engineering': [
      '1-tensor-basics',
      '2-autograd',
      '3-models',
      '4-loss-functions',
      '5-optimizers',
      '6-training-loop',
      '7-data-loading',
      '8-checkpoints-and-infrastructure'
    ],
    'stage-2-gpu-and-performance-engineering': [
      '1-gpu-fundamentals',
      '2-gpu-architecture',
      '3-gpu-memory',
      '4-tensor-operations',
      '5-pytorch-and-gpu',
      '6-memory-bottlenecks',
      '7-gpu-utilization',
      '8-data-loading-performance',
      '9-profiling',
      '10-mixed-precision',
      '11-engineering-questions'
    ],
    'stage-3-distributed-training': [
      '1-basic-concepts-of-distributed-training',
      '2-data-parallelism',
      '3-all-reduce-and-communication',
      '4-pytorch-distributed',
      '5-distributed-data-parallel-ddp',
      '6-batch-size-and-scaling',
      '7-model-parallelism',
      '8-pipeline-parallelism',
      '9-fsdp-and-zero',
      '10-memory-and-optimization',
      '11-multi-node-training',
      '12-debugging-distributed-training',
      '13-engineering-questions'
    ],
    'stage-4-ml-infrastructure': [
      '1-basics-of-ml-infrastructure',
      '2-data-pipelines',
      '3-dataset-storage',
      '4-experiment-tracking',
      '5-hyperparameter-tuning',
      '6-training-orchestration',
      '7-containerization',
      '8-kubernetes-and-gpu-orchestration',
      '9-model-artifacts',
      '10-inference-systems',
      '11-monitoring',
      '12-production-reliability',
      '13-architecture-questions'
    ],
    'stage-5-large-model-training': [
      '1-basics-of-large-models',
      '2-scaling-laws',
      '3-architectures-of-large-models',
      '4-memory-in-training',
      '5-mixed-precision',
      '6-gradient-accumulation',
      '7-checkpointing',
      '8-distributed-training-of-large-models',
      '9-training-stability',
      '10-datasets-for-large-models',
      '11-training-cost',
      '12-engineering-problems'
    ],
    'stage-6-advanced-ml-systems': [
      '1-ai-system-architecture',
      '2-llm-inference',
      '3-llm-serving',
      '4-inference-optimization',
      '5-memory-optimization',
      '6-inference-architectures',
      '7-retrieval-augmented-generation-rag',
      '8-multimodal-systems',
      '9-cost-engineering',
      '10-reliability-in-production',
      '11-architectural-questions'
    ]
  };

  // Paths: /data/posts/{category-slug}/{n-subcategory-slug}/{post-slug}.md
  private predefinedCategories: PredefinedCategories = {
    'stage-0-machine-learning-fundamentals': {
      name: 'Stage 0 — Machine Learning Fundamentals',
      slug: 'stage-0-machine-learning-fundamentals',
      subcategories: [
        {
          name: '1. Fundamentals of Machine Learning',
          slug: '1-fundamentals-of-machine-learning',
          posts: [
            { title: '1. Что такое machine learning?', slug: '1-what-is-machine-learning', excerpt: 'Введение: что называют machine learning и в чём суть подхода.' },
            { title: '2. Чем machine learning отличается от обычного программирования?', slug: '2-how-ml-differs-from-traditional-programming', excerpt: 'Сравнение явных правил и обучения по данным.' },
            { title: '3. В каких задачах machine learning применяется чаще всего?', slug: '3-common-ml-application-tasks', excerpt: 'Типичные области и классы задач для ML.' },
            { title: '4. Какие основные компоненты есть в любой ML-системе?', slug: '4-main-components-of-ml-system', excerpt: 'Данные, модель, обучение, оценка и развёртывание.' },
            { title: '5. Что такое модель (model) в machine learning?', slug: '5-what-is-a-model-in-ml', excerpt: 'Определение модели как функции с обучаемыми параметрами.' },
            { title: '6. Что значит «обучить модель»?', slug: '6-what-does-training-a-model-mean', excerpt: 'Что происходит при training: данные, лосс, оптимизация.' },
            { title: '7. Что такое training data?', slug: '7-what-is-training-data', excerpt: 'Данные, на которых настраиваются параметры модели.' },
            { title: '8. Что такое feature?', slug: '8-what-is-a-feature', excerpt: 'Признак входа: что модель использует как сигнал.' },
            { title: '9. Что такое label (target)?', slug: '9-what-is-a-label-or-target', excerpt: 'Целевая переменная в обучении с учителем.' },
            { title: '10. Что такое prediction?', slug: '10-what-is-prediction', excerpt: 'Выход модели на новых входах.' },
            { title: '11. Что такое inference?', slug: '11-what-is-inference', excerpt: 'Применение обученной модели без обновления весов.' },
            { title: '12. Чем training отличается от inference?', slug: '12-training-vs-inference', excerpt: 'Режимы обучения и применения модели.' },
            { title: '13. Что такое dataset?', slug: '13-what-is-a-dataset', excerpt: 'Набор примеров для обучения и оценки.' },
            { title: '14. Почему качество данных критично для machine learning?', slug: '14-why-data-quality-matters-in-ml', excerpt: 'Связь качества данных и качества модели.' },
            { title: '15. Что такое training set?', slug: '15-what-is-training-set', excerpt: 'Часть данных для обучения параметров.' },
            { title: '16. Что такое validation set?', slug: '16-what-is-validation-set', excerpt: 'Данные для настройки гиперпараметров и ранней остановки.' },
            { title: '17. Что такое test set?', slug: '17-what-is-test-set', excerpt: 'Отложенная выборка для финальной оценки.' },
            { title: '18. Зачем dataset делят на train / validation / test?', slug: '18-why-split-train-validation-test', excerpt: 'Зачем разделять данные и что даёт каждый сплит.' },
            { title: '19. Что произойдёт, если обучать модель на test dataset?', slug: '19-training-on-test-set-problem', excerpt: 'Почему нельзя учить на тесте и что ломается.' },
            { title: '20. Что такое generalization?', slug: '20-what-is-generalization', excerpt: 'Способность модели работать на новых данных.' },
            { title: '21. Что такое distribution данных?', slug: '21-what-is-data-distribution', excerpt: 'Распределение признаков и меток в выборке.' },
            { title: '22. Почему важно, чтобы training и test данные имели похожую distribution?', slug: '22-train-test-distribution-match', excerpt: 'Сдвиг распределения и ошибки оценки.' },
            { title: '23. Что такое data leakage?', slug: '23-what-is-data-leakage', excerpt: 'Когда в признаках «подглядывают» ответ или будущее.' },
            { title: '24. Почему data leakage делает результаты модели недостоверными?', slug: '24-why-data-leakage-invalidates-results', excerpt: 'Завышенные метрики и иллюзия качества.' },
            { title: '25. Что такое bias в данных?', slug: '25-what-is-dataset-bias', excerpt: 'Систематические искажения в сборе и разметке.' },
            { title: '26. Как dataset bias влияет на модель?', slug: '26-how-dataset-bias-affects-models', excerpt: 'Как модель усиливает перекосы и что с этим делать.' },
            { title: '27. Что такое noise в данных?', slug: '27-what-is-noise-in-ml-data', excerpt: 'Случайные ошибки в метках и измерениях.' },
            { title: '28. Как шум в данных влияет на обучение?', slug: '28-how-noise-affects-training', excerpt: 'Переобучение, смещение и нестабильность.' },
            { title: '29. Что такое feature engineering?', slug: '29-what-is-feature-engineering', excerpt: 'Создание и отбор признаков под задачу.' },
            { title: '30. Почему feature engineering может быть важнее модели?', slug: '30-why-feature-engineering-can-outweigh-models', excerpt: 'Когда данные и признаки решают больше, чем архитектура.' },
            { title: '31. Что такое pipeline в machine learning?', slug: '31-what-is-ml-pipeline', excerpt: 'Связка шагов от сырых данных до предсказания.' },
            { title: '32. Какие этапы обычно включает ML pipeline?', slug: '32-typical-stages-of-ml-pipeline', excerpt: 'Сбор, подготовка, обучение, валидация, деплой.' },
            { title: '33. Что такое preprocessing данных?', slug: '33-what-is-data-preprocessing', excerpt: 'Очистка, фильтрация и подготовка признаков.' },
            { title: '34. Что такое normalization признаков?', slug: '34-what-is-feature-normalization', excerpt: 'Приведение признаков к заданному диапазону.' },
            { title: '35. Что такое standardization признаков?', slug: '35-what-is-feature-standardization', excerpt: 'Центрирование и масштаб по стандартному отклонению.' },
            { title: '36. Почему масштабирование признаков важно для некоторых моделей?', slug: '36-why-feature-scaling-matters', excerpt: 'Градиенты, расстояния и сходимость.' },
            { title: '37. Что такое categorical features?', slug: '37-what-are-categorical-features', excerpt: 'Категориальные признаки и отличие от числовых.' },
            { title: '38. Какие способы кодирования категориальных признаков существуют?', slug: '38-categorical-encoding-methods', excerpt: 'Label, one-hot, target encoding и другие.' },
            { title: '39. Что такое one-hot encoding?', slug: '39-what-is-one-hot-encoding', excerpt: 'Вектор из нулей и одной единицы на категорию.' },
            { title: '40. Какие проблемы возникают при большом количестве категорий?', slug: '40-high-cardinality-categorical-problems', excerpt: 'Разреженность, память и переобучение.' },
            { title: '41. Что такое baseline модель?', slug: '41-what-is-a-baseline-model', excerpt: 'Простая модель для сравнения и ориентира.' },
            { title: '42. Почему baseline важен перед построением сложной модели?', slug: '42-why-baseline-before-complex-models', excerpt: 'Контроль сложности и проверка данных.' },
            { title: '43. Что значит «модель переобучилась»?', slug: '43-what-is-overfitting-in-ml', excerpt: 'Запоминание обучающей выборки вместо закономерностей.' },
            { title: '44. Что значит «модель недообучилась»?', slug: '44-what-is-underfitting-in-ml', excerpt: 'Слишком простая модель и высокий bias.' },
            { title: '45. Как определить, что модель обучается правильно?', slug: '45-how-to-tell-model-is-training-well', excerpt: 'Метрики, кривые и сравнение train/val.' },
            { title: '46. Что такое learning curve?', slug: '46-what-is-learning-curve', excerpt: 'Зависимость метрик от объёма данных или итераций.' },
            { title: '47. Почему важно разделять эксперимент и production модель?', slug: '47-experiment-vs-production-models', excerpt: 'Чистота оценки и версионирование.' },
            { title: '48. Что такое reproducibility в machine learning?', slug: '48-what-is-reproducibility-in-ml', excerpt: 'Воспроизводимость кода, данных и результатов.' },
            { title: '49. Почему результаты обучения могут отличаться между запусками?', slug: '49-why-training-runs-differ', excerpt: 'Случайность, сиды и недетерминизм.' },
            { title: '50. Какие основные ошибки чаще всего делают начинающие в machine learning?', slug: '50-common-beginner-mistakes-in-ml', excerpt: 'Типичные ловушки при работе с данными и моделями.' }
          ]
        },
        {
          name: '2. Types of Learning',
          slug: '2-types-of-learning',
          posts: [
            { title: '1. Что такое тип обучения (learning paradigm) в machine learning?', slug: '1-what-is-learning-paradigm-in-ml', excerpt: 'Парадигма обучения: как формулируется задача и что оптимизируется.' },
            { title: '2. Какие основные типы обучения существуют в machine learning?', slug: '2-main-types-of-learning-in-ml', excerpt: 'Обзор supervised, unsupervised, semi-supervised, self-supervised, RL.' },
            { title: '3. Чем supervised learning отличается от unsupervised learning?', slug: '3-supervised-vs-unsupervised-learning', excerpt: 'Разметка, целевая переменная и постановка задачи.' },
            { title: '4. Что такое supervised learning?', slug: '4-what-is-supervised-learning', excerpt: 'Обучение с учителем: входы, метки и функция потерь.' },
            { title: '5. Какие задачи решает supervised learning?', slug: '5-what-tasks-does-supervised-learning-solve', excerpt: 'Классификация, регрессия и смежные постановки.' },
            { title: '6. Что такое training data в supervised learning?', slug: '6-training-data-in-supervised-learning', excerpt: 'Пары пример–метка для настройки модели.' },
            { title: '7. Почему supervised learning требует размеченные данные?', slug: '7-why-supervised-learning-needs-labeled-data', excerpt: 'Связь меток с целевой функцией и оценкой ошибки.' },
            { title: '8. Какие примеры задач supervised learning существуют?', slug: '8-supervised-learning-task-examples', excerpt: 'Практические примеры из разных доменов.' },
            { title: '9. Чем regression отличается от classification?', slug: '9-regression-vs-classification', excerpt: 'Непрерывный выход против дискретных классов.' },
            { title: '10. Какие алгоритмы часто используются для supervised learning?', slug: '10-common-supervised-learning-algorithms', excerpt: 'От линейных моделей до бустинга и нейросетей.' },
            { title: '11. Что такое unsupervised learning?', slug: '11-what-is-unsupervised-learning', excerpt: 'Обучение без меток: структура и представления.' },
            { title: '12. Какие задачи решает unsupervised learning?', slug: '12-what-tasks-does-unsupervised-learning-solve', excerpt: 'Кластеризация, снижение размерности, плотность.' },
            { title: '13. Чем unsupervised learning отличается от supervised learning?', slug: '13-unsupervised-vs-supervised-learning', excerpt: 'Отсутствие меток и другие цели обучения.' },
            { title: '14. Какие данные используются в unsupervised learning?', slug: '14-data-used-in-unsupervised-learning', excerpt: 'Неразмеченные выборки и допущения.' },
            { title: '15. Что такое clustering?', slug: '15-what-is-clustering', excerpt: 'Группировка похожих объектов без заранее заданных классов.' },
            { title: '16. Какие алгоритмы используются для clustering?', slug: '16-clustering-algorithms', excerpt: 'K-means, иерархическая, DBSCAN и др.' },
            { title: '17. Что такое dimensionality reduction?', slug: '17-what-is-dimensionality-reduction', excerpt: 'Сжатие признакового пространства с сохранением сигнала.' },
            { title: '18. Почему dimensionality reduction полезен?', slug: '18-why-dimensionality-reduction-is-useful', excerpt: 'Визуализация, скорость и борьба с проклятием размерности.' },
            { title: '19. Какие алгоритмы используются для уменьшения размерности?', slug: '19-dimensionality-reduction-algorithms', excerpt: 'PCA, t-SNE, UMAP и нейросетевые энкодеры.' },
            { title: '20. В каких задачах используют unsupervised learning?', slug: '20-where-is-unsupervised-learning-used', excerpt: 'Сегментация, поиск аномалий, представления.' },
            { title: '21. Что такое semi-supervised learning?', slug: '21-what-is-semi-supervised-learning', excerpt: 'Сочетание размеченных и неразмеченных данных.' },
            { title: '22. Почему semi-supervised learning используется на практике?', slug: '22-why-semi-supervised-learning-in-practice', excerpt: 'Экономия разметки и масштаб данных.' },
            { title: '23. В каких ситуациях размеченных данных мало?', slug: '23-when-labeled-data-is-scarce', excerpt: 'Дорогая разметка, редкие классы, новые домены.' },
            { title: '24. Как можно использовать неразмеченные данные для обучения?', slug: '24-using-unlabeled-data-in-training', excerpt: 'Псевдометки, согласованность, совместные модели.' },
            { title: '25. Какие методы semi-supervised learning существуют?', slug: '25-semi-supervised-methods', excerpt: 'Self-training, graph-based, consistency regularization.' },
            { title: '26. Чем semi-supervised learning отличается от supervised learning?', slug: '26-semi-supervised-vs-supervised', excerpt: 'Роль неразмеченной части выборки.' },
            { title: '27. Чем semi-supervised learning отличается от unsupervised learning?', slug: '27-semi-supervised-vs-unsupervised', excerpt: 'Наличие хотя бы части меток.' },
            { title: '28. Какие проблемы возникают при использовании semi-supervised learning?', slug: '28-semi-supervised-challenges', excerpt: 'Качество псевдометок и смещения.' },
            { title: '29. Какие преимущества у semi-supervised подходов?', slug: '29-semi-supervised-advantages', excerpt: 'Баланс затрат и качества.' },
            { title: '30. В каких областях применяется semi-supervised learning?', slug: '30-semi-supervised-application-domains', excerpt: 'Медицина, NLP, компьютерное зрение.' },
            { title: '31. Что такое self-supervised learning?', slug: '31-what-is-self-supervised-learning', excerpt: 'Постановка задач с сигналом из самих данных.' },
            { title: '32. Чем self-supervised learning отличается от unsupervised learning?', slug: '32-self-supervised-vs-unsupervised', excerpt: 'Явные proxy-задачи и обучаемый лосс.' },
            { title: '33. Почему self-supervised learning стал популярным?', slug: '33-why-self-supervised-became-popular', excerpt: 'Масштабируемые представления без ручной разметки.' },
            { title: '34. Какие задачи используют self-supervised learning?', slug: '34-self-supervised-tasks', excerpt: 'Предсказание маскированных частей, контраст, ротации.' },
            { title: '35. Что такое pretext task?', slug: '35-what-is-pretext-task', excerpt: 'Вспомогательная задача для обучения представлений.' },
            { title: '36. Как self-supervised learning используется в NLP?', slug: '36-self-supervised-in-nlp', excerpt: 'MLM, next sentence, контрастные цели.' },
            { title: '37. Как self-supervised learning используется в computer vision?', slug: '37-self-supervised-in-vision', excerpt: 'Jigsaw, contrastive, маскирование патчей.' },
            { title: '38. Как self-supervised learning используется в large language models?', slug: '38-self-supervised-in-llms', excerpt: 'Предобучение на тексте с proxy-задачами.' },
            { title: '39. Почему self-supervised learning требует большие datasets?', slug: '39-self-supervised-needs-large-data', excerpt: 'Сложность сигнала и стабильность представлений.' },
            { title: '40. Какие преимущества self-supervised learning по сравнению с supervised learning?', slug: '40-self-supervised-vs-supervised-advantages', excerpt: 'Меньше ручной разметки, перенос представлений.' },
            { title: '41. Что такое reinforcement learning?', slug: '41-what-is-reinforcement-learning', excerpt: 'Агент, среда, награды и последовательности решений.' },
            { title: '42. Какие компоненты входят в reinforcement learning систему?', slug: '42-rl-system-components', excerpt: 'Агент, среда, политика, функции ценности.' },
            { title: '43. Что такое agent в reinforcement learning?', slug: '43-what-is-rl-agent', excerpt: 'Сущность, выбирающая действия по наблюдениям.' },
            { title: '44. Что такое environment?', slug: '44-what-is-environment-in-rl', excerpt: 'Динамика состояний, наград и переходов.' },
            { title: '45. Что такое reward?', slug: '45-what-is-reward-in-rl', excerpt: 'Скалярная обратная связь за шаг или эпизод.' },
            { title: '46. Что такое policy?', slug: '46-what-is-policy-in-rl', excerpt: 'Отображение состояния в действие или распределение.' },
            { title: '47. Что такое exploration и exploitation?', slug: '47-exploration-vs-exploitation', excerpt: 'Баланс поиска нового и использования известного.' },
            { title: '48. В каких задачах используется reinforcement learning?', slug: '48-where-is-reinforcement-learning-used', excerpt: 'Игры, робототехника, рекомендации, системы.' },
            { title: '49. Какие сложности возникают при обучении reinforcement learning моделей?', slug: '49-rl-training-challenges', excerpt: 'Дисперсия, кредит, сэмплы, безопасность.' },
            { title: '50. Чем reinforcement learning принципиально отличается от других типов обучения?', slug: '50-rl-vs-other-paradigms', excerpt: 'Последовательности, награда, взаимодействие со средой.' }
          ]
        },
        {
          name: '3. Data',
          slug: '3-data',
          posts: [
            { title: '1. Что такое данные (data) в machine learning?', slug: '1-ml-data-what-is-data', excerpt: 'Сырые наблюдения, признаки и связь с задачей.' },
            { title: '2. Почему данные являются основой любой ML-модели?', slug: '2-ml-data-why-foundation', excerpt: 'Предел качества модели и типичные узкие места.' },
            { title: '3. Что такое dataset?', slug: '3-ml-data-dataset-concept', excerpt: 'Табличное или иное представление выборки для обучения.' },
            { title: '4. Какие типы данных используются в machine learning?', slug: '4-ml-data-types', excerpt: 'Числовые, категориальные, текст, изображения, время.' },
            { title: '5. Что такое feature в dataset?', slug: '5-ml-data-feature-in-dataset', excerpt: 'Столбец или сигнал, описывающий объект.' },
            { title: '6. Что такое target или label?', slug: '6-ml-data-target-or-label', excerpt: 'Ответ для supervised-задачи.' },
            { title: '7. Чем отличаются признаки (features) и целевая переменная (target)?', slug: '7-ml-data-features-vs-target', excerpt: 'Входы модели и то, что предсказываем.' },
            { title: '8. Что такое observation или sample?', slug: '8-ml-data-observation-sample', excerpt: 'Одна строка или один объект в выборке.' },
            { title: '9. Что такое размерность dataset?', slug: '9-ml-data-dataset-dimensionality', excerpt: 'Число признаков и объём выборки.' },
            { title: '10. Что такое feature space?', slug: '10-ml-data-feature-space', excerpt: 'Пространство, в котором лежат векторы признаков.' },
            { title: '11. Что такое train dataset?', slug: '11-ml-data-train-dataset', excerpt: 'Часть данных для обучения параметров.' },
            { title: '12. Что такое validation dataset?', slug: '12-ml-data-validation-dataset', excerpt: 'Выборка для настройки и ранней остановки.' },
            { title: '13. Что такое test dataset?', slug: '13-ml-data-test-dataset', excerpt: 'Отложенная оценка обобщения.' },
            { title: '14. Почему dataset разделяют на train, validation и test?', slug: '14-ml-data-why-split', excerpt: 'Смещение оценки и честная проверка.' },
            { title: '15. Как обычно делят dataset на части?', slug: '15-ml-data-split-proportions', excerpt: 'Типичные доли и стратификация.' },
            { title: '16. Почему test dataset нельзя использовать при обучении?', slug: '16-ml-data-no-test-in-training', excerpt: 'Утечка и завышенные метрики.' },
            { title: '17. Что такое holdout dataset?', slug: '17-ml-data-holdout', excerpt: 'Отложенная часть без участия в обучении.' },
            { title: '18. Что такое cross-validation?', slug: '18-ml-data-cross-validation', excerpt: 'K-fold и повторное использование данных для оценки.' },
            { title: '19. Когда используют cross-validation?', slug: '19-ml-data-when-cross-validation', excerpt: 'Мало данных, выбор модели, сравнение гиперпараметров.' },
            { title: '20. Какие проблемы возникают при маленьком dataset?', slug: '20-ml-data-small-dataset-issues', excerpt: 'Высокая дисперсия оценок и переобучение.' },
            { title: '21. Что такое distribution данных?', slug: '21-ml-data-distribution', excerpt: 'Закономерности и плотность признаков и меток.' },
            { title: '22. Почему важно, чтобы training и test данные имели похожую distribution?', slug: '22-ml-data-similar-train-test-distribution', excerpt: 'Сдвиг и ошибка в проде.' },
            { title: '23. Что такое dataset shift?', slug: '23-ml-data-dataset-shift', excerpt: 'Изменение распределения между этапами.' },
            { title: '24. Что такое covariate shift?', slug: '24-ml-data-covariate-shift', excerpt: 'Сдвиг P(X) при том же условном законе.' },
            { title: '25. Что такое concept drift?', slug: '25-ml-data-concept-drift', excerpt: 'Меняется связь признаков и метки.' },
            { title: '26. Как drift влияет на модель?', slug: '26-ml-data-drift-impact', excerpt: 'Падение метрик и некалиброванные решения.' },
            { title: '27. Как обнаружить изменение distribution данных?', slug: '27-ml-data-detect-distribution-change', excerpt: 'Мониторинг, статистики, сравнение окон.' },
            { title: '28. Почему модели деградируют со временем?', slug: '28-ml-data-why-models-degrade', excerpt: 'Мир меняется, данные и схемы сбора.' },
            { title: '29. Как часто нужно переобучать модель?', slug: '29-ml-data-retrain-frequency', excerpt: 'Критерии и практика MLOps.' },
            { title: '30. Что такое data leakage?', slug: '30-ml-data-leakage', excerpt: 'Информация из «будущего» или теста в признаках.' },
            { title: '31. Какие типы data leakage существуют?', slug: '31-ml-data-leakage-types', excerpt: 'Train-test, временная, утечка цели.' },
            { title: '32. Почему data leakage делает результаты модели недостоверными?', slug: '32-ml-data-leakage-unreliable', excerpt: 'Завышение качества на валидации.' },
            { title: '33. Как обнаружить data leakage в dataset?', slug: '33-ml-data-detect-leakage', excerpt: 'Проверки корреляций, времени, сплитов.' },
            { title: '34. Как предотвратить data leakage?', slug: '34-ml-data-prevent-leakage', excerpt: 'Пайплайны, правильный сплит, версии данных.' },
            { title: '35. Что такое missing data?', slug: '35-ml-data-missing-values', excerpt: 'Пропуски, NA и их происхождение.' },
            { title: '36. Какие способы обработки пропущенных значений существуют?', slug: '36-ml-data-missing-handling-methods', excerpt: 'Удаление, константы, модели пропусков.' },
            { title: '37. Что такое imputation?', slug: '37-ml-data-imputation', excerpt: 'Заполнение пропусков оценёнными значениями.' },
            { title: '38. Когда лучше удалить данные с пропущенными значениями?', slug: '38-ml-data-when-drop-missing', excerpt: 'Мало строк, MCAR, дубли.' },
            { title: '39. Как пропущенные значения влияют на обучение модели?', slug: '39-ml-data-missing-training-impact', excerpt: 'Смещение, ложные сигналы, NaN в градиентах.' },
            { title: '40. Что такое outliers?', slug: '40-ml-data-outliers', excerpt: 'Точки далеко от основной массы.' },
            { title: '41. Почему выбросы (outliers) могут быть проблемой для моделей?', slug: '41-ml-data-outliers-problem', excerpt: 'Лоссы, масштаб, линейные модели.' },
            { title: '42. Как обнаружить outliers в dataset?', slug: '42-ml-data-detect-outliers', excerpt: 'IQR, z-score, изоляционный лес.' },
            { title: '43. Какие методы обработки outliers существуют?', slug: '43-ml-data-outlier-methods', excerpt: 'Обрезка, winsorize, робастные модели.' },
            { title: '44. Что такое class imbalance?', slug: '44-ml-data-class-imbalance', excerpt: 'Неравные частоты классов.' },
            { title: '45. Почему дисбаланс классов может быть проблемой?', slug: '45-ml-data-why-imbalance-problem', excerpt: 'Метрики и перекос предсказаний.' },
            { title: '46. Какие методы используются для работы с class imbalance?', slug: '46-ml-data-imbalance-methods', excerpt: 'Веса, сэмплирование, пороги.' },
            { title: '47. Что такое data augmentation?', slug: '47-ml-data-augmentation', excerpt: 'Синтетическое расширение обучающей выборки.' },
            { title: '48. Когда data augmentation используется?', slug: '48-ml-data-when-augmentation', excerpt: 'Изображения, текст, мало данных.' },
            { title: '49. Как качество данных влияет на качество модели?', slug: '49-ml-data-quality-vs-model', excerpt: 'Ошибки разметки, шум, покрытие.' },
            { title: '50. Почему в machine learning часто говорят «garbage in, garbage out»?', slug: '50-ml-data-gigo', excerpt: 'Связь входа данных и потолка модели.' }
          ]
        },
        {
          name: '4. Linear Regression',
          slug: '4-linear-regression',
          posts: [
            { title: '1. Что такое linear regression?', slug: '1-ml-linreg-what-is-linear-regression', excerpt: 'Простая линейная модель для предсказания непрерывной цели.' },
            { title: '2. Какие задачи решает linear regression?', slug: '2-ml-linreg-tasks', excerpt: 'Регрессия и базовый baseline для числовых таргетов.' },
            { title: '3. Как выглядит математическая форма линейной регрессии?', slug: '3-ml-linreg-mathematical-form', excerpt: 'Взвешенная сумма признаков плюс смещение.' },
            { title: '4. Что означает коэффициент в линейной регрессии?', slug: '4-ml-linreg-coefficient-meaning', excerpt: 'Вклад признака при прочих равных.' },
            { title: '5. Что такое bias (intercept) в линейной модели?', slug: '5-ml-linreg-bias-intercept', excerpt: 'Свободный член и сдвиг гиперплоскости.' },
            { title: '6. Что такое признаки (features) в линейной регрессии?', slug: '6-ml-linreg-features', excerpt: 'Входы модели: сырые или инженерные.' },
            { title: '7. Что такое целевая переменная (target)?', slug: '7-ml-linreg-target', excerpt: 'Число, которое предсказываем.' },
            { title: '8. Что означает предсказание модели?', slug: '8-ml-linreg-prediction-meaning', excerpt: 'Оценка отклика по обученным весам.' },
            { title: '9. Как linear regression вычисляет предсказание?', slug: '9-ml-linreg-how-prediction-computed', excerpt: 'Скалярное произведение весов и x.' },
            { title: '10. Что такое линейная зависимость между переменными?', slug: '10-ml-linreg-linear-dependence', excerpt: 'Пропорциональное изменение отклика.' },
            { title: '11. Что такое residual (ошибка предсказания)?', slug: '11-ml-linreg-residual', excerpt: 'Разность факта и предсказания.' },
            { title: '12. Что такое функция потерь (loss function) в линейной регрессии?', slug: '12-ml-linreg-loss-function', excerpt: 'Мера ошибки на обучающей выборке.' },
            { title: '13. Что такое mean squared error (MSE)?', slug: '13-ml-linreg-mse', excerpt: 'Средний квадрат остатков.' },
            { title: '14. Почему MSE часто используется как функция потерь?', slug: '14-ml-linreg-why-mse', excerpt: 'Дифференцируемость и штраф крупным ошибкам.' },
            { title: '15. Что такое root mean squared error (RMSE)?', slug: '15-ml-linreg-rmse', excerpt: 'Корень из MSE в тех же единицах, что и таргет.' },
            { title: '16. Чем RMSE отличается от MSE?', slug: '16-ml-linreg-rmse-vs-mse', excerpt: 'Масштаб и интерпретируемость.' },
            { title: '17. Что такое mean absolute error (MAE)?', slug: '17-ml-linreg-mae', excerpt: 'Среднее абсолютное отклонение.' },
            { title: '18. Когда MAE лучше MSE?', slug: '18-ml-linreg-mae-vs-mse', excerpt: 'Устойчивость к выбросам и другая интерпретация.' },
            { title: '19. Как интерпретировать значение ошибки модели?', slug: '19-ml-linreg-interpreting-error', excerpt: 'Сравнение с масштабом таргета и baseline.' },
            { title: '20. Почему минимизация ошибки важна для обучения модели?', slug: '20-ml-linreg-why-minimize-error', excerpt: 'Связь эмпирического риска и качества.' },
            { title: '21. Что такое параметры модели?', slug: '21-ml-linreg-model-parameters', excerpt: 'Веса и смещение, которые настраиваются.' },
            { title: '22. Как параметры линейной регрессии определяются во время обучения?', slug: '22-ml-linreg-how-parameters-learned', excerpt: 'Минимизация лосса по данным.' },
            { title: '23. Что такое gradient descent?', slug: '23-ml-linreg-gradient-descent', excerpt: 'Итеративное движение против градиента.' },
            { title: '24. Как gradient descent используется в линейной регрессии?', slug: '24-ml-linreg-gd-in-linear-regression', excerpt: 'Обновление весов по производным MSE.' },
            { title: '25. Что такое learning rate?', slug: '25-ml-linreg-learning-rate', excerpt: 'Шаг обновления весов.' },
            { title: '26. Как learning rate влияет на обучение?', slug: '26-ml-linreg-learning-rate-effect', excerpt: 'Скорость и стабильность сходимости.' },
            { title: '27. Что произойдёт, если learning rate слишком большой?', slug: '27-ml-linreg-learning-rate-too-large', excerpt: 'Расходимость и осцилляции.' },
            { title: '28. Что произойдёт, если learning rate слишком маленький?', slug: '28-ml-linreg-learning-rate-too-small', excerpt: 'Медленная сходимость.' },
            { title: '29. Что такое сходимость (convergence)?', slug: '29-ml-linreg-convergence', excerpt: 'Стабилизация лосса и весов.' },
            { title: '30. Как понять, что модель сошлась?', slug: '30-ml-linreg-how-to-tell-converged', excerpt: 'Пороги, кривые, ранняя остановка.' },
            { title: '31. Что такое normal equation в линейной регрессии?', slug: '31-ml-linreg-normal-equation', excerpt: 'Замкнутая форма OLS через псевдообратную.' },
            { title: '32. Чем аналитическое решение отличается от gradient descent?', slug: '32-ml-linreg-analytic-vs-gd', excerpt: 'Точность vs масштаб и регуляризация.' },
            { title: '33. Когда используют аналитическое решение?', slug: '33-ml-linreg-when-analytic', excerpt: 'Умеренное p, полный ранг, умеренные n.' },
            { title: '34. Почему gradient descent используется чаще при больших данных?', slug: '34-ml-linreg-why-gd-large-data', excerpt: 'Память и стохастические варианты.' },
            { title: '35. Как количество признаков влияет на линейную регрессию?', slug: '35-ml-linreg-number-of-features', excerpt: 'Сложность, дисперсия и численная устойчивость.' },
            { title: '36. Что происходит при сильной корреляции признаков?', slug: '36-ml-linreg-strong-correlation', excerpt: 'Нестабильные оценки весов.' },
            { title: '37. Что такое multicollinearity?', slug: '37-ml-linreg-multicollinearity', excerpt: 'Линейная зависимость между признаками.' },
            { title: '38. Почему multicollinearity может быть проблемой?', slug: '38-ml-linreg-why-multicollinearity-problem', excerpt: 'Раздутые дисперсии коэффициентов.' },
            { title: '39. Как обнаружить multicollinearity?', slug: '39-ml-linreg-detect-multicollinearity', excerpt: 'VIF, корреляции, условное число.' },
            { title: '40. Какие способы борьбы с multicollinearity существуют?', slug: '40-ml-linreg-multicollinearity-remedies', excerpt: 'Отбор признаков, PCA, регуляризация.' },
            { title: '41. Что такое regularization в линейной регрессии?', slug: '41-ml-linreg-regularization', excerpt: 'Штраф за величину весов.' },
            { title: '42. Что такое Ridge regression (L2 regularization)?', slug: '42-ml-linreg-ridge-l2', excerpt: 'Квадратичный штраф и сжатие весов.' },
            { title: '43. Что такое Lasso regression (L1 regularization)?', slug: '43-ml-linreg-lasso-l1', excerpt: 'L1 и разреживание коэффициентов.' },
            { title: '44. Чем L1 отличается от L2 регуляризации?', slug: '44-ml-linreg-l1-vs-l2', excerpt: 'Разреживание vs гладкое сжатие.' },
            { title: '45. Как регуляризация помогает бороться с переобучением?', slug: '45-ml-linreg-regularization-overfitting', excerpt: 'Ограничение сложности гипотезы.' },
            { title: '46. Как интерпретировать коэффициенты линейной модели?', slug: '46-ml-linreg-coefficient-interpretation', excerpt: 'При масштабировании признаков и осторожности с причинностью.' },
            { title: '47. Какие ограничения есть у линейной регрессии?', slug: '47-ml-linreg-limitations', excerpt: 'Линейность, выбросы, нелинейные эффекты.' },
            { title: '48. Когда линейная регрессия работает плохо?', slug: '48-ml-linreg-when-works-poorly', excerpt: 'Сильная нелинейность и сложные взаимодействия.' },
            { title: '49. Почему линейные модели всё ещё широко используются?', slug: '49-ml-linreg-why-still-used', excerpt: 'Скорость, интерпретируемость, базовый уровень.' },
            { title: '50. В каких задачах линейная регрессия остаётся хорошим baseline?', slug: '50-ml-linreg-baseline-use-cases', excerpt: 'Табличные данные, A/B, простые сигналы.' }
          ]
        },
        {
          name: '5. Logistic Regression',
          slug: '5-logistic-regression',
          posts: [
            { title: '1. Что такое logistic regression?', slug: '1-ml-logreg-what-is-logistic-regression', excerpt: 'Линейная модель с сигмоидой для вероятностей классов.' },
            { title: '2. Почему logistic regression используется для задач классификации?', slug: '2-ml-logreg-why-classification', excerpt: 'Вероятностная интерпретация и гладкий лосс.' },
            { title: '3. Чем logistic regression отличается от linear regression?', slug: '3-ml-logreg-vs-linear-regression', excerpt: 'Бинарный/категориальный таргет и нелинейная связь с линейным скором.' },
            { title: '4. Что является выходом модели logistic regression?', slug: '4-ml-logreg-model-output', excerpt: 'Вероятность или логит в зависимости от постановки.' },
            { title: '5. Почему logistic regression предсказывает вероятность?', slug: '5-ml-logreg-why-probability', excerpt: 'Сигмоида как калибруемая модель шансов.' },
            { title: '6. Что такое бинарная классификация?', slug: '6-ml-logreg-binary-classification', excerpt: 'Два класса и порог решения.' },
            { title: '7. Какие задачи можно решать с помощью logistic regression?', slug: '7-ml-logreg-tasks', excerpt: 'Спам, отток, медицинская диагностика, скоринг.' },
            { title: '8. Как выглядит формула logistic regression?', slug: '8-ml-logreg-formula', excerpt: 'Сигмоида от линейной комбинации признаков.' },
            { title: '9. Что такое logit?', slug: '9-ml-logreg-logit', excerpt: 'Логарифм отношения шансов к линейному скору.' },
            { title: '10. Что такое decision boundary?', slug: '10-ml-logreg-decision-boundary', excerpt: 'Гиперплоскость, разделяющая классы в пространстве признаков.' },
            { title: '11. Что такое sigmoid функция?', slug: '11-ml-logreg-sigmoid', excerpt: 'S-образная карта ℝ → (0,1).' },
            { title: '12. Почему sigmoid используется в logistic regression?', slug: '12-ml-logreg-why-sigmoid', excerpt: 'Связь линейного скора с вероятностью.' },
            { title: '13. Как sigmoid преобразует значения в вероятности?', slug: '13-ml-logreg-sigmoid-to-probability', excerpt: 'Монотонное сжатие в интервал.' },
            { title: '14. Почему выход sigmoid находится между 0 и 1?', slug: '14-ml-logreg-sigmoid-range', excerpt: 'Свойства экспоненты в знаменателе.' },
            { title: '15. Что означает вероятность, которую выдаёт модель?', slug: '15-ml-logreg-probability-meaning', excerpt: 'Оценка P(y=1|x) при корректной калибровке.' },
            { title: '16. Как probability превращается в класс?', slug: '16-ml-logreg-probability-to-class', excerpt: 'Сравнение с порогом или argmax.' },
            { title: '17. Что такое threshold в классификации?', slug: '17-ml-logreg-threshold', excerpt: 'Граница вероятности для метки.' },
            { title: '18. Почему часто используют threshold = 0.5?', slug: '18-ml-logreg-threshold-0-5', excerpt: 'Симметрия для сбалансированных классов и costs.' },
            { title: '19. Когда threshold стоит изменить?', slug: '19-ml-logreg-when-change-threshold', excerpt: 'Дисбаланс, разные цены ошибок, целевой recall/precision.' },
            { title: '20. Как threshold влияет на precision и recall?', slug: '20-ml-logreg-threshold-precision-recall', excerpt: 'Сдвиг компромисса по ROC/PR.' },
            { title: '21. Что такое log loss?', slug: '21-ml-logreg-log-loss', excerpt: 'Кросс-энтропия для бинарного случая.' },
            { title: '22. Почему log loss используется в logistic regression?', slug: '22-ml-logreg-why-log-loss', excerpt: 'Согласованность с вероятностной моделью и выпуклость.' },
            { title: '23. Чем log loss отличается от MSE?', slug: '23-ml-logreg-log-loss-vs-mse', excerpt: 'Штраф за калибровку вероятностей vs квадратичная ошибка.' },
            { title: '24. Почему MSE плохо работает для классификации?', slug: '24-ml-logreg-mse-bad-for-classification', excerpt: 'Плохая градиентная сигнализация у границ.' },
            { title: '25. Как выглядит формула log loss?', slug: '25-ml-logreg-log-loss-formula', excerpt: '−(y log p + (1−y) log(1−p)).' },
            { title: '26. Как log loss штрафует неправильные предсказания?', slug: '26-ml-logreg-log-loss-penalty', excerpt: 'Большой штраф за уверенную ошибку.' },
            { title: '27. Почему уверенные неправильные предсказания сильно увеличивают loss?', slug: '27-ml-logreg-confident-wrong-penalty', excerpt: 'Логарифм стремится к −∞ при p→0 или 1.' },
            { title: '28. Что происходит с loss, когда вероятность близка к правильному классу?', slug: '28-ml-logreg-loss-near-correct', excerpt: 'Малый градиент и низкий вклад.' },
            { title: '29. Как интерпретировать значение log loss?', slug: '29-ml-logreg-interpreting-log-loss', excerpt: 'Сравнение с baseline и относительное улучшение.' },
            { title: '30. Почему минимизация log loss обучает модель?', slug: '30-ml-logreg-why-minimize-log-loss', excerpt: 'Максимизация правдоподобия при биномиальной модели.' },
            { title: '31. Как logistic regression обучается?', slug: '31-ml-logreg-how-trained', excerpt: 'Минимизация log loss по весам.' },
            { title: '32. Почему используется gradient descent?', slug: '32-ml-logreg-why-gd', excerpt: 'Выпуклая задача и масштаб данных.' },
            { title: '33. Как вычисляется градиент функции потерь?', slug: '33-ml-logreg-loss-gradient', excerpt: 'Цепное правило через сигмоиду.' },
            { title: '34. Что происходит с параметрами модели при обучении?', slug: '34-ml-logreg-parameters-update', excerpt: 'Шаг по антиградиенту log loss.' },
            { title: '35. Как learning rate влияет на обучение logistic regression?', slug: '35-ml-logreg-learning-rate-effect', excerpt: 'Скорость и риск расходимости.' },
            { title: '36. Что происходит если модель не сходится?', slug: '36-ml-logreg-non-convergence', excerpt: 'Колебания лосса и нестабильные веса.' },
            { title: '37. Как определить, что обучение прошло успешно?', slug: '37-ml-logreg-training-success', excerpt: 'Стабильный лосс и метрики на валидации.' },
            { title: '38. Что такое регуляризация в logistic regression?', slug: '38-ml-logreg-regularization', excerpt: 'L1/L2 штраф к весам в логите.' },
            { title: '39. Как L1 регуляризация влияет на модель?', slug: '39-ml-logreg-l1-effect', excerpt: 'Разреживание признаков.' },
            { title: '40. Как L2 регуляризация влияет на модель?', slug: '40-ml-logreg-l2-effect', excerpt: 'Сжатие весов без обнуления.' },
            { title: '41. Что такое multiclass classification?', slug: '41-ml-logreg-multiclass', excerpt: 'Более двух меток.' },
            { title: '42. Как logistic regression используется для нескольких классов?', slug: '42-ml-logreg-multiclass-usage', excerpt: 'OvR, OvO или softmax.' },
            { title: '43. Что такое one-vs-rest подход?', slug: '43-ml-logreg-one-vs-rest', excerpt: 'K бинарных классификаторов по одному классу.' },
            { title: '44. Что такое softmax regression?', slug: '44-ml-logreg-softmax-regression', excerpt: 'Многоклассовое обобщение с нормализацией вероятностей.' },
            { title: '45. Чем softmax отличается от sigmoid?', slug: '45-ml-logreg-softmax-vs-sigmoid', excerpt: 'Вектор вероятностей, суммирующийся в 1.' },
            { title: '46. Какие ограничения есть у logistic regression?', slug: '46-ml-logreg-limitations', excerpt: 'Линейная граница, аддитивный логит.' },
            { title: '47. Когда logistic regression работает плохо?', slug: '47-ml-logreg-when-poor', excerpt: 'Сильная нелинейность и взаимодействия без признаков.' },
            { title: '48. Почему logistic regression всё ещё широко используется?', slug: '48-ml-logreg-why-still-used', excerpt: 'Скорость, интерпретируемость, калибровка.' },
            { title: '49. Когда logistic regression может быть хорошим baseline?', slug: '49-ml-logreg-baseline', excerpt: 'Табличные данные и первый сравнимый уровень.' },
            { title: '50. Почему logistic regression считается простой, но мощной моделью?', slug: '50-ml-logreg-simple-but-powerful', excerpt: 'Сильная теория, вероятности, устойчивые практики.' }
          ]
        },
        {
          name: '6. Loss Functions',
          slug: '6-loss-functions',
          posts: [
            { title: '1. Что такое loss function в machine learning?', slug: '1-ml-lossfn-what-is-loss', excerpt: 'Скаляр ошибки между предсказанием и меткой.' },
            { title: '2. Зачем нужна функция потерь?', slug: '2-ml-lossfn-why-needed', excerpt: 'Цель для градиентной оптимизации и измерения качества на шаге.' },
            { title: '3. Чем loss function отличается от метрики качества модели?', slug: '3-ml-lossfn-vs-metric', excerpt: 'Лосс для обучения; метрика — для оценки и бизнес-смысла.' },
            { title: '4. Что означает минимизация функции потерь?', slug: '4-ml-lossfn-minimization-meaning', excerpt: 'Поиск параметров с меньшей средней ошибкой.' },
            { title: '5. Почему функция потерь используется во время обучения модели?', slug: '5-ml-lossfn-why-training', excerpt: 'Сигнал для обновления весов по градиенту.' },
            { title: '6. Как loss function связана с оптимизацией модели?', slug: '6-ml-lossfn-optimization-link', excerpt: 'Она задаёт целевую функцию, которую минимизирует оптимизатор.' },
            { title: '7. Что такое objective function?', slug: '7-ml-lossfn-objective', excerpt: 'Целевая функция: минимизировать loss или максимизировать reward.' },
            { title: '8. Чем objective function отличается от loss function?', slug: '8-ml-lossfn-objective-vs-loss', excerpt: 'Loss — частный случай; objective может включать регуляризацию.' },
            { title: '9. Почему loss функция должна быть вычислима для каждого примера?', slug: '9-ml-lossfn-per-example', excerpt: 'Стохастические градиенты и мини-батчи.' },
            { title: '10. Почему loss функция должна быть дифференцируемой?', slug: '10-ml-lossfn-differentiable', excerpt: 'Для обратного распространения и градиентного спуска.' },
            { title: '11. Что такое средняя ошибка по dataset?', slug: '11-ml-lossfn-mean-error-dataset', excerpt: 'Эмпирическое среднее лосса по всем точкам.' },
            { title: '12. Почему loss обычно усредняется по batch?', slug: '12-ml-lossfn-why-average-batch', excerpt: 'Масштаб градиента и оценка полного риска.' },
            { title: '13. Что такое batch loss?', slug: '13-ml-lossfn-batch-loss', excerpt: 'Средний или суммарный лосс на подмножестве данных.' },
            { title: '14. Что такое expected loss?', slug: '14-ml-lossfn-expected-loss', excerpt: 'Математическое ожидание ошибки по распределению данных.' },
            { title: '15. Почему минимизация loss приводит к улучшению модели?', slug: '15-ml-lossfn-minimization-improves', excerpt: 'Снижение эмпирического риска при корректной постановке.' },
            { title: '16. Как loss function влияет на поведение модели?', slug: '16-ml-lossfn-affects-behavior', excerpt: 'Задаёт, что считать ошибкой и как штрафовать.' },
            { title: '17. Почему разные задачи требуют разные функции потерь?', slug: '17-ml-lossfn-different-tasks', excerpt: 'Разные распределения таргета и цели бизнеса.' },
            { title: '18. Что происходит если выбрать неподходящую loss функцию?', slug: '18-ml-lossfn-wrong-choice', excerpt: 'Смещённые градиенты и плохая калибровка.' },
            { title: '19. Как loss функция влияет на градиенты?', slug: '19-ml-lossfn-loss-gradients', excerpt: 'Определяет направление и величину обновления весов.' },
            { title: '20. Почему градиенты важны для обучения?', slug: '20-ml-lossfn-gradients-importance', excerpt: 'Локальное направление к минимуму целевой функции.' },
            { title: '21. Что такое Mean Squared Error (MSE)?', slug: '21-ml-lossfn-mse', excerpt: 'Среднее квадрата отклонения предсказания от истины.' },
            { title: '22. В каких задачах используется MSE?', slug: '22-ml-lossfn-mse-tasks', excerpt: 'Регрессия с гауссовским шумом и квадратичная штрафовка.' },
            { title: '23. Почему MSE чувствительна к выбросам?', slug: '23-ml-lossfn-mse-outliers', excerpt: 'Квадрат усиливает большие ошибки.' },
            { title: '24. Что такое Mean Absolute Error (MAE)?', slug: '24-ml-lossfn-mae', excerpt: 'Среднее модуля ошибки.' },
            { title: '25. Чем MAE отличается от MSE?', slug: '25-ml-lossfn-mae-vs-mse', excerpt: 'Линейный штраф и устойчивость к выбросам.' },
            { title: '26. Когда MAE предпочтительнее MSE?', slug: '26-ml-lossfn-when-mae', excerpt: 'Тяжёлые хвосты и робастность к аномалиям.' },
            { title: '27. Что такое Huber loss?', slug: '27-ml-lossfn-huber', excerpt: 'Гибрид MAE и MSE с порогом переключения.' },
            { title: '28. Когда Huber loss полезен?', slug: '28-ml-lossfn-when-huber', excerpt: 'Регрессия с умеренными выбросами и гладкость у нуля.' },
            { title: '29. Чем Huber loss отличается от MSE и MAE?', slug: '29-ml-lossfn-huber-vs-mse-mae', excerpt: 'Квадратично мало, линейно далеко.' },
            { title: '30. В каких задачах используется Huber loss?', slug: '30-ml-lossfn-huber-tasks', excerpt: 'Робастная регрессия и обучение с шумом.' },
            { title: '31. Что такое log loss (cross entropy)?', slug: '31-ml-lossfn-log-loss-cross-entropy', excerpt: 'Отрицательное лог-правдоподобие для вероятностных предсказаний.' },
            { title: '32. Почему cross entropy используется в классификации?', slug: '32-ml-lossfn-why-cross-entropy', excerpt: 'Согласованность с дискретными метками и вероятностной моделью.' },
            { title: '33. Чем cross entropy отличается от MSE?', slug: '33-ml-lossfn-ce-vs-mse', excerpt: 'Штраф за распределение классов vs квадратичная ошибка вероятностей.' },
            { title: '34. Почему cross entropy лучше подходит для вероятностей?', slug: '34-ml-lossfn-ce-probabilities', excerpt: 'Логарифм и выпуклость для p∈(0,1).' },
            { title: '35. Что такое binary cross entropy?', slug: '35-ml-lossfn-binary-cross-entropy', excerpt: 'CE для двух классов и одной вероятности.' },
            { title: '36. Что такое categorical cross entropy?', slug: '36-ml-lossfn-categorical-cross-entropy', excerpt: 'CE для one-hot распределения по K классам.' },
            { title: '37. Когда используется categorical cross entropy?', slug: '37-ml-lossfn-when-categorical-ce', excerpt: 'Многоклассовая классификация без взаимоисключающих дублирований.' },
            { title: '38. Что такое logits?', slug: '38-ml-lossfn-logits', excerpt: 'Некалиброванные скоры до softmax/sigmoid.' },
            { title: '39. Почему некоторые loss функции работают с logits, а не с вероятностями?', slug: '39-ml-lossfn-logits-not-probs', excerpt: 'Численная стабильность и точный градиент.' },
            { title: '40. Что такое softmax cross entropy?', slug: '40-ml-lossfn-softmax-cross-entropy', excerpt: 'CE, объединённая с softmax по логитам.' },
            { title: '41. Что такое regularization loss?', slug: '41-ml-lossfn-regularization-loss', excerpt: 'Добавочный штраф к величине параметров.' },
            { title: '42. Почему регуляризация добавляется к функции потерь?', slug: '42-ml-lossfn-why-regularization-in-loss', excerpt: 'Единая цель: fit + сложность модели.' },
            { title: '43. Что такое L1 регуляризация?', slug: '43-ml-lossfn-l1', excerpt: 'Сумма модулей весов как штраф.' },
            { title: '44. Что такое L2 регуляризация?', slug: '44-ml-lossfn-l2', excerpt: 'Сумма квадратов весов как штраф.' },
            { title: '45. Как регуляризация влияет на параметры модели?', slug: '45-ml-lossfn-regularization-params', excerpt: 'Сжатие весов и снижение нормы.' },
            { title: '46. Почему регуляризация помогает бороться с переобучением?', slug: '46-ml-lossfn-regularization-overfitting', excerpt: 'Ограничивает ёмкость гипотез.' },
            { title: '47. Что такое weighted loss?', slug: '47-ml-lossfn-weighted-loss', excerpt: 'Взвешенная сумма ошибок по примерам или классам.' },
            { title: '48. Когда используется weighted loss?', slug: '48-ml-lossfn-when-weighted', excerpt: 'Неравные цены ошибок и дисбаланс классов.' },
            { title: '49. Почему class imbalance требует изменения функции потерь?', slug: '49-ml-lossfn-class-imbalance-loss', excerpt: 'Иначе класс-миноритет игнорируется.' },
            { title: '50. Как выбор loss функции влияет на конечное поведение модели?', slug: '50-ml-lossfn-choice-affects-model', excerpt: 'Задаёт, что оптимизировать: калибровку, порог, робастность.' }
          ]
        },
        {
          name: '7. Gradient Descent',
          slug: '7-gradient-descent',
          posts: [
            { title: '1. Что такое оптимизация в machine learning?', slug: '1-ml-gdopt-what-is-optimization', excerpt: 'Поиск параметров, минимизирующих целевую функцию.' },
            { title: '2. Почему обучение модели формулируется как задача оптимизации?', slug: '2-ml-gdopt-why-training-is-optimization', excerpt: 'Нужно найти веса с минимальной ошибкой.' },
            { title: '3. Что такое функция потерь в контексте оптимизации?', slug: '3-ml-gdopt-loss-in-optimization', excerpt: 'Скалярная цель, отражающая ошибку предсказаний.' },
            { title: '4. Что означает минимизация функции потерь?', slug: '4-ml-gdopt-loss-minimization-meaning', excerpt: 'Снижение средней ошибки на данных.' },
            { title: '5. Что такое параметр модели?', slug: '5-ml-gdopt-model-parameter', excerpt: 'Настраиваемый коэффициент, влияющий на выход модели.' },
            { title: '6. Как параметры модели влияют на предсказания?', slug: '6-ml-gdopt-parameters-affect-predictions', excerpt: 'Определяют форму функции и итоговый прогноз.' },
            { title: '7. Что такое градиент функции?', slug: '7-ml-gdopt-gradient-definition', excerpt: 'Вектор частных производных по параметрам.' },
            { title: '8. Почему градиент показывает направление наибольшего изменения функции?', slug: '8-ml-gdopt-gradient-largest-change', excerpt: 'Он указывает направление максимального локального роста.' },
            { title: '9. Почему градиент используется для обновления параметров модели?', slug: '9-ml-gdopt-gradient-for-updates', excerpt: 'Даёт направление шага к уменьшению loss.' },
            { title: '10. Что такое gradient descent?', slug: '10-ml-gdopt-what-is-gradient-descent', excerpt: 'Итеративный метод минимизации через антиградиент.' },
            { title: '11. Как работает gradient descent?', slug: '11-ml-gdopt-how-gradient-descent-works', excerpt: 'Повторяет шаги обновления весов по градиенту.' },
            { title: '12. Почему gradient descent позволяет находить минимум функции?', slug: '12-ml-gdopt-why-finds-minimum', excerpt: 'Последовательные шаги уменьшают значение целевой функции.' },
            { title: '13. Что такое learning rate?', slug: '13-ml-gdopt-learning-rate', excerpt: 'Коэффициент размера шага обновления.' },
            { title: '14. Как learning rate влияет на процесс обучения?', slug: '14-ml-gdopt-lr-effect', excerpt: 'Баланс скорости сходимости и стабильности.' },
            { title: '15. Что произойдёт, если learning rate слишком большой?', slug: '15-ml-gdopt-lr-too-large', excerpt: 'Перескоки минимума и расходимость.' },
            { title: '16. Что произойдёт, если learning rate слишком маленький?', slug: '16-ml-gdopt-lr-too-small', excerpt: 'Очень медленное обучение и долгий тренинг.' },
            { title: '17. Что такое шаг обновления параметров?', slug: '17-ml-gdopt-update-step', excerpt: 'Изменение весов на одной итерации.' },
            { title: '18. Как параметры модели обновляются на каждой итерации?', slug: '18-ml-gdopt-parameter-update-each-iter', excerpt: 'Новый вес = старый вес минус learning rate на градиент.' },
            { title: '19. Что такое итерация обучения?', slug: '19-ml-gdopt-training-iteration', excerpt: 'Один проход обновления по batch.' },
            { title: '20. Что такое epoch?', slug: '20-ml-gdopt-epoch', excerpt: 'Полный проход по всему тренировочному набору.' },
            { title: '21. Что такое batch gradient descent?', slug: '21-ml-gdopt-batch-gd', excerpt: 'Градиент по всему датасету на шаг.' },
            { title: '22. Чем batch gradient descent отличается от stochastic gradient descent?', slug: '22-ml-gdopt-batch-vs-sgd', excerpt: 'Полный набор против одного примера.' },
            { title: '23. Что такое stochastic gradient descent (SGD)?', slug: '23-ml-gdopt-sgd', excerpt: 'Обновления по одному примеру или очень малому batch.' },
            { title: '24. Почему SGD часто используется на практике?', slug: '24-ml-gdopt-why-sgd-practical', excerpt: 'Масштабируемость и частые обновления на больших данных.' },
            { title: '25. Что такое mini-batch gradient descent?', slug: '25-ml-gdopt-mini-batch', excerpt: 'Компромисс между batch GD и SGD.' },
            { title: '26. Почему mini-batch обучение является стандартом в deep learning?', slug: '26-ml-gdopt-why-minibatch-standard', excerpt: 'Эффективно на GPU и стабильно по градиенту.' },
            { title: '27. Как размер batch влияет на обучение модели?', slug: '27-ml-gdopt-batch-size-effect', excerpt: 'Влияет на скорость, шум и обобщение.' },
            { title: '28. Как batch size влияет на стабильность градиентов?', slug: '28-ml-gdopt-batch-size-gradient-stability', excerpt: 'Больший batch снижает дисперсию оценок градиента.' },
            { title: '29. Что такое шум в градиентах?', slug: '29-ml-gdopt-gradient-noise', excerpt: 'Случайные колебания оценки истинного градиента.' },
            { title: '30. Почему шум может помогать оптимизации?', slug: '30-ml-gdopt-why-noise-helps', excerpt: 'Помогает выходить из острых локальных областей.' },
            { title: '31. Что такое локальный минимум?', slug: '31-ml-gdopt-local-minimum', excerpt: 'Точка меньше соседних значений функции.' },
            { title: '32. Чем локальный минимум отличается от глобального?', slug: '32-ml-gdopt-local-vs-global-minimum', excerpt: 'Глобальный минимум на всей области, не только локально.' },
            { title: '33. Что такое saddle point?', slug: '33-ml-gdopt-saddle-point', excerpt: 'Точка с нулевым градиентом без минимума.' },
            { title: '34. Почему saddle points могут замедлять обучение?', slug: '34-ml-gdopt-why-saddles-slow', excerpt: 'Плоские направления уменьшают величину шагов.' },
            { title: '35. Почему оптимизация нейронных сетей сложнее линейных моделей?', slug: '35-ml-gdopt-why-nn-optimization-harder', excerpt: 'Невыпуклый ландшафт и много параметров.' },
            { title: '36. Что такое convergence в оптимизации?', slug: '36-ml-gdopt-convergence', excerpt: 'Состояние, когда улучшения становятся малыми.' },
            { title: '37. Как понять, что обучение модели сошлось?', slug: '37-ml-gdopt-how-detect-convergence', excerpt: 'Loss и метрики стабилизировались на валидации.' },
            { title: '38. Почему loss может перестать уменьшаться?', slug: '38-ml-gdopt-why-loss-stops-decreasing', excerpt: 'Плато, слишком маленький lr или достигнут предел.' },
            { title: '39. Что такое learning rate schedule?', slug: '39-ml-gdopt-learning-rate-schedule', excerpt: 'Правило изменения learning rate во времени.' },
            { title: '40. Зачем изменять learning rate во время обучения?', slug: '40-ml-gdopt-why-change-lr', excerpt: 'Быстрый старт и точная донастройка к концу.' },
            { title: '41. Что такое momentum в оптимизации?', slug: '41-ml-gdopt-momentum', excerpt: 'Накопление скорости обновлений по прошлым градиентам.' },
            { title: '42. Как momentum ускоряет обучение?', slug: '42-ml-gdopt-how-momentum-speeds', excerpt: 'Сглаживает шум и ускоряет движение в согласованных направлениях.' },
            { title: '43. Что такое AdaGrad?', slug: '43-ml-gdopt-adagrad', excerpt: 'Адаптивный метод с индивидуальными шагами по параметрам.' },
            { title: '44. Что такое RMSProp?', slug: '44-ml-gdopt-rmsprop', excerpt: 'Адаптивный оптимизатор с экспоненциальным средним квадратов градиента.' },
            { title: '45. Что такое Adam optimizer?', slug: '45-ml-gdopt-adam', excerpt: 'Комбинация momentum и RMSProp с bias correction.' },
            { title: '46. Почему Adam широко используется в deep learning?', slug: '46-ml-gdopt-why-adam-used', excerpt: 'Быстро сходится и мало требует тюнинга.' },
            { title: '47. Чем Adam отличается от SGD?', slug: '47-ml-gdopt-adam-vs-sgd', excerpt: 'Адаптивные шаги и моменты против фиксированного шага.' },
            { title: '48. Когда SGD может работать лучше Adam?', slug: '48-ml-gdopt-when-sgd-better', excerpt: 'Иногда даёт лучшее обобщение на финальных эпохах.' },
            { title: '49. Что такое gradient clipping?', slug: '49-ml-gdopt-gradient-clipping', excerpt: 'Ограничение нормы или значения градиента.' },
            { title: '50. Почему gradient clipping используется при обучении нейронных сетей?', slug: '50-ml-gdopt-why-gradient-clipping', excerpt: 'Защищает от взрыва градиентов и нестабильности.' }
          ]
        },
        {
          name: '8. Overfitting',
          slug: '8-overfitting',
          posts: [
            { title: '1. Что такое overfitting в machine learning?', slug: '1-ml-overfit-what-is-overfitting', excerpt: 'Переобучение: модель запоминает шум train-данных.' },
            { title: '2. Почему модель может переобучиться?', slug: '2-ml-overfit-why-happens', excerpt: 'Слишком высокая сложность и слабая регуляризация.' },
            { title: '3. Чем overfitting отличается от underfitting?', slug: '3-ml-overfit-vs-underfit', excerpt: 'Переобучение: высокая variance; недообучение: высокий bias.' },
            { title: '4. Что означает underfitting?', slug: '4-ml-overfit-what-is-underfitting', excerpt: 'Модель слишком проста и не улавливает закономерности.' },
            { title: '5. Как выглядит overfitting на графике обучения?', slug: '5-ml-overfit-training-curve-look', excerpt: 'Train error падает, validation error растёт.' },
            { title: '6. Как выглядит underfitting на графике обучения?', slug: '6-ml-overfit-underfitting-curve-look', excerpt: 'И train, и validation ошибки остаются высокими.' },
            { title: '7. Как связаны сложность модели и переобучение?', slug: '7-ml-overfit-complexity-link', excerpt: 'Рост сложности часто повышает риск overfitting.' },
            { title: '8. Почему слишком сложная модель может переобучиться?', slug: '8-ml-overfit-too-complex-model', excerpt: 'Легко подгоняет шум вместо сигнала.' },
            { title: '9. Как размер dataset влияет на переобучение?', slug: '9-ml-overfit-dataset-size-effect', excerpt: 'Больше данных обычно снижает риск переобучения.' },
            { title: '10. Почему маленькие datasets чаще приводят к overfitting?', slug: '10-ml-overfit-small-datasets', excerpt: 'Недостаточно разнообразия для устойчивого обобщения.' },
            { title: '11. Что такое generalization?', slug: '11-ml-overfit-what-is-generalization', excerpt: 'Способность работать на невидимых данных.' },
            { title: '12. Почему цель обучения модели — хорошая generalization?', slug: '12-ml-overfit-why-generalization-goal', excerpt: 'Ценность модели проявляется на новых примерах.' },
            { title: '13. Как измерить способность модели обобщать?', slug: '13-ml-overfit-measure-generalization', excerpt: 'Оценка на validation/test или через кросс-валидацию.' },
            { title: '14. Почему модель может показывать отличные результаты на train данных и плохие на test?', slug: '14-ml-overfit-train-good-test-bad', excerpt: 'Модель выучила особенности train-набора.' },
            { title: '15. Что такое training error?', slug: '15-ml-overfit-training-error', excerpt: 'Ошибка модели на обучающей выборке.' },
            { title: '16. Что такое validation error?', slug: '16-ml-overfit-validation-error', excerpt: 'Ошибка на отложенной валидационной выборке.' },
            { title: '17. Почему важно отслеживать обе ошибки?', slug: '17-ml-overfit-track-both-errors', excerpt: 'Разрыв между ними выявляет переобучение.' },
            { title: '18. Что такое bias-variance tradeoff?', slug: '18-ml-overfit-bias-variance-tradeoff', excerpt: 'Компромисс между недо- и переобучением.' },
            { title: '19. Что такое bias в модели?', slug: '19-ml-overfit-what-is-bias', excerpt: 'Систематическая ошибка из-за упрощений модели.' },
            { title: '20. Что такое variance в модели?', slug: '20-ml-overfit-what-is-variance', excerpt: 'Чувствительность предсказаний к изменению train-данных.' },
            { title: '21. Почему высокая variance связана с переобучением?', slug: '21-ml-overfit-high-variance-link', excerpt: 'Модель слишком зависит от конкретных примеров.' },
            { title: '22. Почему высокий bias связан с недообучением?', slug: '22-ml-overfit-high-bias-link', excerpt: 'Модель слишком грубо описывает зависимость.' },
            { title: '23. Как баланс bias и variance влияет на качество модели?', slug: '23-ml-overfit-bias-variance-balance', excerpt: 'Оптимальный баланс минимизирует общую ошибку.' },
            { title: '24. Какие признаки указывают на переобучение?', slug: '24-ml-overfit-signs', excerpt: 'Низкий train loss и заметно худший validation.' },
            { title: '25. Как обнаружить переобучение во время обучения модели?', slug: '25-ml-overfit-detect-during-training', excerpt: 'Следить за динамикой validation метрик.' },
            { title: '26. Что такое learning curves?', slug: '26-ml-overfit-learning-curves', excerpt: 'Графики ошибок в зависимости от эпох/объёма данных.' },
            { title: '27. Как learning curves помогают обнаружить переобучение?', slug: '27-ml-overfit-curves-detect', excerpt: 'Показывают разрыв train/validation качества.' },
            { title: '28. Как увеличение данных влияет на переобучение?', slug: '28-ml-overfit-more-data-effect', excerpt: 'Снижает variance и улучшает обобщение.' },
            { title: '29. Когда добавление данных помогает модели?', slug: '29-ml-overfit-when-more-data-helps', excerpt: 'Когда текущая модель ограничена данными, не bias.' },
            { title: '30. Почему иногда данные важнее архитектуры модели?', slug: '30-ml-overfit-data-vs-architecture', excerpt: 'Качество и покрытие данных часто доминируют.' },
            { title: '31. Что такое regularization?', slug: '31-ml-overfit-what-is-regularization', excerpt: 'Методы ограничения сложности модели при обучении.' },
            { title: '32. Почему регуляризация помогает бороться с переобучением?', slug: '32-ml-overfit-why-regularization-helps', excerpt: 'Штрафует избыточно сложные решения.' },
            { title: '33. Что такое L1 regularization?', slug: '33-ml-overfit-l1-regularization', excerpt: 'Штраф по сумме модулей весов.' },
            { title: '34. Что такое L2 regularization?', slug: '34-ml-overfit-l2-regularization', excerpt: 'Штраф по сумме квадратов весов.' },
            { title: '35. Чем L1 отличается от L2 регуляризации?', slug: '35-ml-overfit-l1-vs-l2', excerpt: 'L1 разреживает, L2 плавно сжимает веса.' },
            { title: '36. Как регуляризация влияет на веса модели?', slug: '36-ml-overfit-regularization-on-weights', excerpt: 'Снижает их величину и вариативность.' },
            { title: '37. Почему регуляризация уменьшает сложность модели?', slug: '37-ml-overfit-regularization-reduces-complexity', excerpt: 'Ограничивает пространство допустимых параметров.' },
            { title: '38. Что такое weight decay?', slug: '38-ml-overfit-weight-decay', excerpt: 'Постепенное уменьшение весов при обновлениях.' },
            { title: '39. Чем weight decay связан с L2 регуляризацией?', slug: '39-ml-overfit-weight-decay-l2', excerpt: 'В SGD часто эквивалентен L2-штрафу.' },
            { title: '40. Когда регуляризация может ухудшить модель?', slug: '40-ml-overfit-when-regularization-hurts', excerpt: 'При слишком сильном штрафе возникает underfitting.' },
            { title: '41. Что такое dropout?', slug: '41-ml-overfit-dropout', excerpt: 'Случайное отключение нейронов во время обучения.' },
            { title: '42. Почему dropout помогает уменьшить переобучение?', slug: '42-ml-overfit-why-dropout-helps', excerpt: 'Уменьшает коадаптацию и повышает робастность.' },
            { title: '43. Как dropout работает во время обучения?', slug: '43-ml-overfit-how-dropout-works', excerpt: 'На каждом шаге зануляет случайную долю активаций.' },
            { title: '44. Почему dropout отключается во время inference?', slug: '44-ml-overfit-why-disable-dropout-inference', excerpt: 'Нужны детерминированные предсказания полной моделью.' },
            { title: '45. Что такое early stopping?', slug: '45-ml-overfit-early-stopping', excerpt: 'Остановка обучения при ухудшении validation.' },
            { title: '46. Как early stopping помогает избежать переобучения?', slug: '46-ml-overfit-why-early-stopping', excerpt: 'Фиксирует веса до начала деградации обобщения.' },
            { title: '47. Почему иногда лучше остановить обучение раньше?', slug: '47-ml-overfit-why-stop-earlier', excerpt: 'Дальнейшие эпохи улучшают train, но портят test.' },
            { title: '48. Как выбор архитектуры влияет на переобучение?', slug: '48-ml-overfit-architecture-choice', excerpt: 'Более ёмкие сети требуют больше данных и регуляризации.' },
            { title: '49. Какие практические методы чаще всего используют для борьбы с overfitting?', slug: '49-ml-overfit-practical-methods', excerpt: 'Data augmentation, dropout, weight decay, early stopping.' },
            { title: '50. Почему борьба с переобучением остаётся одной из главных задач в machine learning?', slug: '50-ml-overfit-why-core-problem', excerpt: 'Обобщение на новые данные остаётся ключевой целью.' }
          ]
        },
        {
          name: '9. Metrics',
          slug: '9-metrics',
          posts: [
            { title: '1. Что такое метрика в machine learning?', slug: '1-ml-metrics-what-is-metric', excerpt: 'Численный показатель качества модели.' },
            { title: '2. Чем метрика отличается от функции потерь (loss function)?', slug: '2-ml-metrics-vs-loss', excerpt: 'Loss оптимизируют при обучении, метрикой оценивают результат.' },
            { title: '3. Почему метрики используются для оценки модели?', slug: '3-ml-metrics-why-used', excerpt: 'Они показывают качество на целевых данных.' },
            { title: '4. Почему одна метрика редко отражает всю картину качества модели?', slug: '4-ml-metrics-one-not-enough', excerpt: 'Разные аспекты ошибок требуют разных показателей.' },
            { title: '5. Как выбор метрики зависит от задачи?', slug: '5-ml-metrics-choice-by-task', excerpt: 'От типа задачи и стоимости ошибок.' },
            { title: '6. Почему важно выбрать правильную метрику для задачи?', slug: '6-ml-metrics-choose-right', excerpt: 'Иначе можно оптимизировать не то, что важно бизнесу.' },
            { title: '7. Что такое baseline метрика?', slug: '7-ml-metrics-baseline', excerpt: 'Качество простой опорной модели для сравнения.' },
            { title: '8. Почему baseline важен при оценке модели?', slug: '8-ml-metrics-why-baseline', excerpt: 'Показывает, есть ли реальное улучшение.' },
            { title: '9. Что такое evaluation dataset?', slug: '9-ml-metrics-evaluation-dataset', excerpt: 'Набор данных для независимой проверки качества.' },
            { title: '10. Почему метрики обычно вычисляются на validation или test dataset?', slug: '10-ml-metrics-why-val-test', excerpt: 'Чтобы оценка отражала обобщающую способность.' },
            { title: '11. Что такое accuracy?', slug: '11-ml-metrics-accuracy', excerpt: 'Доля верных предсказаний среди всех.' },
            { title: '12. Как вычисляется accuracy?', slug: '12-ml-metrics-accuracy-formula', excerpt: '(TP+TN)/(TP+TN+FP+FN).' },
            { title: '13. Когда accuracy является хорошей метрикой?', slug: '13-ml-metrics-when-accuracy-good', excerpt: 'При сбалансированных классах и равной цене ошибок.' },
            { title: '14. Когда accuracy может быть вводящей в заблуждение?', slug: '14-ml-metrics-accuracy-misleading', excerpt: 'При сильном дисбалансе классов.' },
            { title: '15. Почему accuracy плохо работает при дисбалансе классов?', slug: '15-ml-metrics-accuracy-imbalanced', excerpt: 'Мажоритарный класс может доминировать показатель.' },
            { title: '16. Что такое confusion matrix?', slug: '16-ml-metrics-confusion-matrix', excerpt: 'Таблица распределения истинных и предсказанных классов.' },
            { title: '17. Какие элементы входят в confusion matrix?', slug: '17-ml-metrics-confusion-elements', excerpt: 'TP, TN, FP, FN.' },
            { title: '18. Что такое true positives?', slug: '18-ml-metrics-true-positives', excerpt: 'Положительные объекты, верно классифицированные как положительные.' },
            { title: '19. Что такое true negatives?', slug: '19-ml-metrics-true-negatives', excerpt: 'Отрицательные объекты, верно классифицированные как отрицательные.' },
            { title: '20. Что такое false positives и false negatives?', slug: '20-ml-metrics-fp-fn', excerpt: 'Ошибки первого и второго рода в классификации.' },
            { title: '21. Что такое precision?', slug: '21-ml-metrics-precision', excerpt: 'Доля верных среди предсказанных положительных.' },
            { title: '22. Как вычисляется precision?', slug: '22-ml-metrics-precision-formula', excerpt: 'TP/(TP+FP).' },
            { title: '23. В каких задачах важна высокая precision?', slug: '23-ml-metrics-precision-use-cases', excerpt: 'Где ложноположительные особенно дорогие.' },
            { title: '24. Что такое recall?', slug: '24-ml-metrics-recall', excerpt: 'Доля найденных положительных среди всех положительных.' },
            { title: '25. Как вычисляется recall?', slug: '25-ml-metrics-recall-formula', excerpt: 'TP/(TP+FN).' },
            { title: '26. В каких задачах важен высокий recall?', slug: '26-ml-metrics-recall-use-cases', excerpt: 'Где критично не пропускать положительные случаи.' },
            { title: '27. Чем precision отличается от recall?', slug: '27-ml-metrics-precision-vs-recall', excerpt: 'Precision про FP, recall про FN.' },
            { title: '28. Почему между precision и recall существует компромисс?', slug: '28-ml-metrics-precision-recall-tradeoff', excerpt: 'Сдвиг порога обычно улучшает одно за счёт другого.' },
            { title: '29. Что такое F1 score?', slug: '29-ml-metrics-f1', excerpt: 'Гармоническое среднее precision и recall.' },
            { title: '30. Почему F1 score объединяет precision и recall?', slug: '30-ml-metrics-f1-why-combines', excerpt: 'Наказывает дисбаланс между двумя метриками.' },
            { title: '31. Что такое ROC curve?', slug: '31-ml-metrics-roc-curve', excerpt: 'Кривая TPR против FPR при разных порогах.' },
            { title: '32. Что показывает ROC curve?', slug: '32-ml-metrics-roc-shows', excerpt: 'Способность ранжировать классы по порогам.' },
            { title: '33. Что такое true positive rate?', slug: '33-ml-metrics-tpr', excerpt: 'То же, что recall: TP/(TP+FN).' },
            { title: '34. Что такое false positive rate?', slug: '34-ml-metrics-fpr', excerpt: 'FP/(FP+TN), доля ложных тревог.' },
            { title: '35. Что такое AUC (Area Under the Curve)?', slug: '35-ml-metrics-auc', excerpt: 'Площадь под ROC-кривой.' },
            { title: '36. Почему AUC используется для оценки классификаторов?', slug: '36-ml-metrics-why-auc', excerpt: 'Порог-независимая оценка качества ранжирования.' },
            { title: '37. Когда AUC полезнее accuracy?', slug: '37-ml-metrics-auc-vs-accuracy', excerpt: 'При дисбалансе и анализе по всем порогам.' },
            { title: '38. Что такое PR curve (precision–recall curve)?', slug: '38-ml-metrics-pr-curve', excerpt: 'Кривая precision против recall.' },
            { title: '39. Когда PR curve полезнее ROC curve?', slug: '39-ml-metrics-pr-vs-roc', excerpt: 'При редком положительном классе.' },
            { title: '40. Как threshold влияет на метрики классификации?', slug: '40-ml-metrics-threshold-impact', excerpt: 'Меняет баланс FP/FN и итоговые метрики.' },
            { title: '41. Какие метрики используются для задач regression?', slug: '41-ml-metrics-regression-metrics', excerpt: 'MSE, RMSE, MAE, R2 и другие.' },
            { title: '42. Что такое Mean Squared Error (MSE)?', slug: '42-ml-metrics-mse', excerpt: 'Средний квадрат ошибки прогноза.' },
            { title: '43. Что такое Root Mean Squared Error (RMSE)?', slug: '43-ml-metrics-rmse', excerpt: 'Квадратный корень из MSE в единицах таргета.' },
            { title: '44. Что такое Mean Absolute Error (MAE)?', slug: '44-ml-metrics-mae', excerpt: 'Средний модуль ошибки прогноза.' },
            { title: '45. Чем RMSE отличается от MAE?', slug: '45-ml-metrics-rmse-vs-mae', excerpt: 'RMSE сильнее штрафует большие ошибки.' },
            { title: '46. Когда RMSE предпочтительнее MAE?', slug: '46-ml-metrics-when-rmse', excerpt: 'Когда крупные промахи нужно штрафовать сильнее.' },
            { title: '47. Что такое R² (coefficient of determination)?', slug: '47-ml-metrics-r2', excerpt: 'Доля объяснённой вариации относительно baseline.' },
            { title: '48. Как интерпретировать значение R²?', slug: '48-ml-metrics-interpret-r2', excerpt: 'Чем ближе к 1, тем лучше объяснение вариации.' },
            { title: '49. Почему разные задачи требуют разные метрики?', slug: '49-ml-metrics-different-tasks', excerpt: 'Разные цели, классы ошибок и требования.' },
            { title: '50. Почему важно анализировать несколько метрик одновременно?', slug: '50-ml-metrics-multiple-metrics', excerpt: 'Чтобы видеть качество модели с разных сторон.' }
          ]
        },
        {
          name: '10. Neural Networks',
          slug: '10-neural-networks',
          posts: [
            { title: '1. Что такое нейронная сеть (neural network)?', slug: '1-ml-nn-what-is-neural-network', excerpt: 'Модель из слоёв нейронов для нелинейных зависимостей.' },
            { title: '2. Почему нейронные сети используются в machine learning?', slug: '2-ml-nn-why-used', excerpt: 'Хорошо аппроксимируют сложные функции на больших данных.' },
            { title: '3. Чем нейронные сети отличаются от линейных моделей?', slug: '3-ml-nn-vs-linear-models', excerpt: 'Наличие нелинейностей и иерархии признаков.' },
            { title: '4. Что такое искусственный нейрон?', slug: '4-ml-nn-artificial-neuron', excerpt: 'Блок, который суммирует входы и применяет активацию.' },
            { title: '5. Из каких компонентов состоит нейрон?', slug: '5-ml-nn-neuron-components', excerpt: 'Входы, веса, bias, линейная часть и активация.' },
            { title: '6. Что такое входы (inputs) нейрона?', slug: '6-ml-nn-neuron-inputs', excerpt: 'Признаки или выходы предыдущего слоя.' },
            { title: '7. Что такое веса (weights) нейрона?', slug: '7-ml-nn-neuron-weights', excerpt: 'Коэффициенты важности входных сигналов.' },
            { title: '8. Что такое bias в нейроне?', slug: '8-ml-nn-neuron-bias', excerpt: 'Смещение, позволяющее сдвигать активацию.' },
            { title: '9. Как нейрон вычисляет своё значение?', slug: '9-ml-nn-neuron-computation', excerpt: 'Линейная комбинация входов плюс bias и активация.' },
            { title: '10. Что такое линейная комбинация входов?', slug: '10-ml-nn-linear-combination', excerpt: 'Сумма произведений входов на их веса.' },
            { title: '11. Что такое activation function?', slug: '11-ml-nn-activation-function', excerpt: 'Нелинейное преобразование выхода нейрона.' },
            { title: '12. Зачем нужна функция активации?', slug: '12-ml-nn-why-activation', excerpt: 'Чтобы сеть могла моделировать нелинейности.' },
            { title: '13. Почему без активации сеть остаётся линейной моделью?', slug: '13-ml-nn-no-activation-linear', excerpt: 'Композиция линейных слоёв остаётся линейной.' },
            { title: '14. Что такое ReLU функция?', slug: '14-ml-nn-relu', excerpt: 'max(0, x), простая и эффективная активация.' },
            { title: '15. Почему ReLU широко используется в нейронных сетях?', slug: '15-ml-nn-why-relu', excerpt: 'Быстрая, устойчивая и уменьшает затухание градиента.' },
            { title: '16. Что такое sigmoid функция?', slug: '16-ml-nn-sigmoid', excerpt: 'S-образная функция, сжимающая в диапазон (0,1).' },
            { title: '17. Когда sigmoid используется в нейронных сетях?', slug: '17-ml-nn-when-sigmoid', excerpt: 'Часто в бинарном выходном слое.' },
            { title: '18. Что такое tanh функция?', slug: '18-ml-nn-tanh', excerpt: 'Активация в диапазоне (-1,1).' },
            { title: '19. Чем tanh отличается от sigmoid?', slug: '19-ml-nn-tanh-vs-sigmoid', excerpt: 'tanh нуле-центрирована и имеет другой диапазон.' },
            { title: '20. Как выбор функции активации влияет на обучение?', slug: '20-ml-nn-activation-choice-impact', excerpt: 'Влияет на градиенты, скорость и стабильность.' },
            { title: '21. Что такое слой (layer) в нейронной сети?', slug: '21-ml-nn-what-is-layer', excerpt: 'Группа нейронов, работающих на одном уровне.' },
            { title: '22. Что такое входной слой (input layer)?', slug: '22-ml-nn-input-layer', excerpt: 'Слой, принимающий исходные признаки.' },
            { title: '23. Что такое скрытый слой (hidden layer)?', slug: '23-ml-nn-hidden-layer', excerpt: 'Промежуточный слой для извлечения представлений.' },
            { title: '24. Что такое выходной слой (output layer)?', slug: '24-ml-nn-output-layer', excerpt: 'Слой, формирующий финальный прогноз.' },
            { title: '25. Как информация проходит через слои сети?', slug: '25-ml-nn-info-through-layers', excerpt: 'Последовательно через линейные преобразования и активации.' },
            { title: '26. Что такое forward pass?', slug: '26-ml-nn-forward-pass', excerpt: 'Прямой проход данных от входа к выходу.' },
            { title: '27. Что происходит во время forward pass?', slug: '27-ml-nn-forward-pass-what-happens', excerpt: 'Вычисляются активации всех слоёв и итоговый выход.' },
            { title: '28. Что такое предсказание нейронной сети?', slug: '28-ml-nn-prediction', excerpt: 'Числовой выход модели для заданного входа.' },
            { title: '29. Как вычисляется выход нейронной сети?', slug: '29-ml-nn-output-computation', excerpt: 'Последовательное применение слоёв к входному вектору.' },
            { title: '30. Как архитектура сети влияет на её возможности?', slug: '30-ml-nn-architecture-capabilities', excerpt: 'Определяет выразительность и индуктивные ограничения.' },
            { title: '31. Что такое параметры нейронной сети?', slug: '31-ml-nn-parameters', excerpt: 'Все обучаемые веса и смещения слоёв.' },
            { title: '32. Почему нейронные сети могут иметь миллионы параметров?', slug: '32-ml-nn-millions-parameters', excerpt: 'Большая ширина/глубина и плотные связи.' },
            { title: '33. Что означает обучение нейронной сети?', slug: '33-ml-nn-what-is-training', excerpt: 'Подбор параметров для минимизации loss.' },
            { title: '34. Что такое backpropagation?', slug: '34-ml-nn-backpropagation', excerpt: 'Алгоритм вычисления градиентов через правило цепочки.' },
            { title: '35. Почему backpropagation используется для обучения сетей?', slug: '35-ml-nn-why-backprop', excerpt: 'Эффективно считает производные для всех параметров.' },
            { title: '36. Как ошибка распространяется назад по сети?', slug: '36-ml-nn-error-backward', excerpt: 'Градиенты передаются от выхода к ранним слоям.' },
            { title: '37. Что такое градиенты в нейронных сетях?', slug: '37-ml-nn-gradients', excerpt: 'Производные loss по параметрам модели.' },
            { title: '38. Как градиенты используются для обновления весов?', slug: '38-ml-nn-gradients-update-weights', excerpt: 'Оптимизатор делает шаг против градиента.' },
            { title: '39. Почему gradient descent используется для обучения сетей?', slug: '39-ml-nn-why-gradient-descent', excerpt: 'Масштабируемый метод оптимизации больших моделей.' },
            { title: '40. Что происходит после многих итераций обучения?', slug: '40-ml-nn-after-many-iterations', excerpt: 'Параметры стабилизируются около минимума loss.' },
            { title: '41. Что такое vanishing gradient problem?', slug: '41-ml-nn-vanishing-gradient', excerpt: 'Градиенты затухают и ранние слои учатся медленно.' },
            { title: '42. Что такое exploding gradients?', slug: '42-ml-nn-exploding-gradients', excerpt: 'Градиенты растут слишком сильно и делают обучение нестабильным.' },
            { title: '43. Почему глубокие сети сложнее обучать?', slug: '43-ml-nn-why-deep-hard', excerpt: 'Больше слоёв, сложнее оптимизация и перенос градиента.' },
            { title: '44. Как архитектура сети влияет на стабильность обучения?', slug: '44-ml-nn-architecture-stability', excerpt: 'Нормализации, skip-связи и активации меняют динамику.' },
            { title: '45. Что такое deep learning?', slug: '45-ml-nn-what-is-deep-learning', excerpt: 'Обучение многослойных нейросетей на данных.' },
            { title: '46. Чем глубокие сети отличаются от простых нейронных сетей?', slug: '46-ml-nn-deep-vs-shallow', excerpt: 'Большей глубиной и иерархией признаков.' },
            { title: '47. Почему глубокие сети могут моделировать сложные зависимости?', slug: '47-ml-nn-why-deep-complex-deps', excerpt: 'Многоуровневые представления повышают выразительность.' },
            { title: '48. Какие области используют deep learning?', slug: '48-ml-nn-deep-learning-domains', excerpt: 'CV, NLP, speech, рекомендательные и биомед задачи.' },
            { title: '49. Какие ограничения есть у нейронных сетей?', slug: '49-ml-nn-limitations', excerpt: 'Требуют данных, вычислений и чувствительны к настройке.' },
            { title: '50. Почему нейронные сети стали основой современной AI индустрии?', slug: '50-ml-nn-why-ai-foundation', excerpt: 'Высокая эффективность на сложных прикладных задачах.' }
          ]
        }
      ]
    },
    'stage-1-pytorch-engineering': {
      name: 'Stage 1 — PyTorch Engineering',
      slug: 'stage-1-pytorch-engineering',
      subcategories: [
        { name: '1. Tensor Basics', slug: '1-tensor-basics', posts: [] },
        { name: '2. Autograd', slug: '2-autograd', posts: [] },
        { name: '3. Models', slug: '3-models', posts: [] },
        { name: '4. Loss Functions', slug: '4-loss-functions', posts: [] },
        { name: '5. Optimizers', slug: '5-optimizers', posts: [] },
        { name: '6. Training Loop', slug: '6-training-loop', posts: [] },
        { name: '7. Data Loading', slug: '7-data-loading', posts: [] },
        { name: '8. Checkpoints and Infrastructure', slug: '8-checkpoints-and-infrastructure', posts: [] }
      ]
    },
    'stage-2-gpu-and-performance-engineering': {
      name: 'Stage 2 — GPU and Performance Engineering',
      slug: 'stage-2-gpu-and-performance-engineering',
      subcategories: [
        { name: '1. GPU Fundamentals', slug: '1-gpu-fundamentals', posts: [] },
        { name: '2. GPU Architecture', slug: '2-gpu-architecture', posts: [] },
        { name: '3. GPU Memory', slug: '3-gpu-memory', posts: [] },
        { name: '4. Tensor Operations', slug: '4-tensor-operations', posts: [] },
        { name: '5. PyTorch and GPU', slug: '5-pytorch-and-gpu', posts: [] },
        { name: '6. Memory Bottlenecks', slug: '6-memory-bottlenecks', posts: [] },
        { name: '7. GPU Utilization', slug: '7-gpu-utilization', posts: [] },
        { name: '8. Data Loading Performance', slug: '8-data-loading-performance', posts: [] },
        { name: '9. Profiling', slug: '9-profiling', posts: [] },
        { name: '10. Mixed Precision', slug: '10-mixed-precision', posts: [] },
        { name: '11. Engineering Questions', slug: '11-engineering-questions', posts: [] }
      ]
    },
    'stage-3-distributed-training': {
      name: 'Stage 3 — Distributed Training',
      slug: 'stage-3-distributed-training',
      subcategories: [
        { name: '1. Basic Concepts of Distributed Training', slug: '1-basic-concepts-of-distributed-training', posts: [] },
        { name: '2. Data Parallelism', slug: '2-data-parallelism', posts: [] },
        { name: '3. All-Reduce and Communication', slug: '3-all-reduce-and-communication', posts: [] },
        { name: '4. PyTorch Distributed', slug: '4-pytorch-distributed', posts: [] },
        { name: '5. Distributed Data Parallel (DDP)', slug: '5-distributed-data-parallel-ddp', posts: [] },
        { name: '6. Batch Size and Scaling', slug: '6-batch-size-and-scaling', posts: [] },
        { name: '7. Model Parallelism', slug: '7-model-parallelism', posts: [] },
        { name: '8. Pipeline Parallelism', slug: '8-pipeline-parallelism', posts: [] },
        { name: '9. FSDP and ZeRO', slug: '9-fsdp-and-zero', posts: [] },
        { name: '10. Memory and Optimization', slug: '10-memory-and-optimization', posts: [] },
        { name: '11. Multi-Node Training', slug: '11-multi-node-training', posts: [] },
        { name: '12. Debugging Distributed Training', slug: '12-debugging-distributed-training', posts: [] },
        { name: '13. Engineering Questions', slug: '13-engineering-questions', posts: [] }
      ]
    },
    'stage-4-ml-infrastructure': {
      name: 'Stage 4 — ML Infrastructure',
      slug: 'stage-4-ml-infrastructure',
      subcategories: [
        { name: '1. Basics of ML Infrastructure', slug: '1-basics-of-ml-infrastructure', posts: [] },
        { name: '2. Data Pipelines', slug: '2-data-pipelines', posts: [] },
        { name: '3. Dataset Storage', slug: '3-dataset-storage', posts: [] },
        { name: '4. Experiment Tracking', slug: '4-experiment-tracking', posts: [] },
        { name: '5. Hyperparameter Tuning', slug: '5-hyperparameter-tuning', posts: [] },
        { name: '6. Training Orchestration', slug: '6-training-orchestration', posts: [] },
        { name: '7. Containerization', slug: '7-containerization', posts: [] },
        { name: '8. Kubernetes and GPU Orchestration', slug: '8-kubernetes-and-gpu-orchestration', posts: [] },
        { name: '9. Model Artifacts', slug: '9-model-artifacts', posts: [] },
        { name: '10. Inference Systems', slug: '10-inference-systems', posts: [] },
        { name: '11. Monitoring', slug: '11-monitoring', posts: [] },
        { name: '12. Production Reliability', slug: '12-production-reliability', posts: [] },
        { name: '13. Architecture Questions', slug: '13-architecture-questions', posts: [] }
      ]
    },
    'stage-5-large-model-training': {
      name: 'Stage 5 — Large Model Training',
      slug: 'stage-5-large-model-training',
      subcategories: [
        { name: '1. Basics of Large Models', slug: '1-basics-of-large-models', posts: [] },
        { name: '2. Scaling Laws', slug: '2-scaling-laws', posts: [] },
        { name: '3. Architectures of Large Models', slug: '3-architectures-of-large-models', posts: [] },
        { name: '4. Memory in Training', slug: '4-memory-in-training', posts: [] },
        { name: '5. Mixed Precision', slug: '5-mixed-precision', posts: [] },
        { name: '6. Gradient Accumulation', slug: '6-gradient-accumulation', posts: [] },
        { name: '7. Checkpointing', slug: '7-checkpointing', posts: [] },
        { name: '8. Distributed Training of Large Models', slug: '8-distributed-training-of-large-models', posts: [] },
        { name: '9. Training Stability', slug: '9-training-stability', posts: [] },
        { name: '10. Datasets for Large Models', slug: '10-datasets-for-large-models', posts: [] },
        { name: '11. Training Cost', slug: '11-training-cost', posts: [] },
        { name: '12. Engineering Problems', slug: '12-engineering-problems', posts: [] }
      ]
    },
    'stage-6-advanced-ml-systems': {
      name: 'Stage 6 — Advanced ML Systems',
      slug: 'stage-6-advanced-ml-systems',
      subcategories: [
        { name: '1. AI System Architecture', slug: '1-ai-system-architecture', posts: [] },
        { name: '2. LLM Inference', slug: '2-llm-inference', posts: [] },
        { name: '3. LLM Serving', slug: '3-llm-serving', posts: [] },
        { name: '4. Inference Optimization', slug: '4-inference-optimization', posts: [] },
        { name: '5. Memory Optimization', slug: '5-memory-optimization', posts: [] },
        { name: '6. Inference Architectures', slug: '6-inference-architectures', posts: [] },
        { name: '7. Retrieval-Augmented Generation (RAG)', slug: '7-retrieval-augmented-generation-rag', posts: [] },
        { name: '8. Multimodal Systems', slug: '8-multimodal-systems', posts: [] },
        { name: '9. Cost Engineering', slug: '9-cost-engineering', posts: [] },
        { name: '10. Reliability in Production', slug: '10-reliability-in-production', posts: [] },
        { name: '11. Architectural Questions', slug: '11-architectural-questions', posts: [] }
      ]
    }
  };


  private compareBySlugOrder(order: string[], aSlug: string, bSlug: string): number {
    const ia = order.indexOf(aSlug);
    const ib = order.indexOf(bSlug);
    return (ia === -1 ? 1000 : ia) - (ib === -1 ? 1000 : ib);
  }

  constructor(
    private http: HttpClient,
    private themeService: ThemeService
  ) {
    // Configure marked for better security and formatting
    marked.setOptions({
      gfm: true,
      breaks: true
    });

    // Configure custom renderer for syntax highlighting and image handling
    const renderer = new marked.Renderer();
    
    renderer.code = ({ text, lang }: { text: string; lang?: string }): string => {
      if (lang && hljs.getLanguage(lang)) {
        try {
          const highlighted = hljs.highlight(text, { language: lang }).value;
          const themeClass = this.themeService.currentTheme === 'dark' ? 'hljs-dark' : 'hljs-light';
          return `<pre><code class="hljs language-${lang} ${themeClass}">${highlighted}</code></pre>`;
        } catch (err) {
          console.warn('Highlight.js error:', err);
        }
      }
      
      // Fallback to auto-detection
      const autoDetected = hljs.highlightAuto(text);
      const themeClass = this.themeService.currentTheme === 'dark' ? 'hljs-dark' : 'hljs-light';
      return `<pre><code class="hljs ${themeClass}">${autoDetected.value}</code></pre>`;
    };

    // Configure image renderer for proper path resolution and click handling
    renderer.image = function({ href, title, text }: { href: string; title: string | null; text: string }): string {
      // If the image path doesn't start with http/https, treat it as a local image
      if (!href.startsWith('http://') && !href.startsWith('https://')) {
        // Resolve relative image paths to the data/images directory
        href = `/data/images/${href}`;
      }
      
      const titleAttr = title ? ` title="${title}"` : '';
      const altAttr = text ? ` alt="${text}"` : '';
      
      return `<img src="${href}"${altAttr}${titleAttr} class="post-image clickable-image">`;
    };

    marked.setOptions({ renderer });
  }

  private loadAllPostsMetadata(): Observable<PostMetadata[]> {
    if (this.allPostsCache) {
      return of(this.allPostsCache);
    }

    const allPosts: PostMetadata[] = [];
    const orderedSlugs = new Set(this.studyOrderCategorySlugs);
    const categorySlugs = [
      ...this.studyOrderCategorySlugs,
      ...Object.keys(this.predefinedCategories).filter(s => !orderedSlugs.has(s))
    ];

    for (const categorySlug of categorySlugs) {
      const category = this.predefinedCategories[categorySlug];
      if (!category) {
        continue;
      }
      const subOrder = this.studyOrderSubcategorySlugs[categorySlug] ?? [];
      const subcategories = [...category.subcategories].sort((a, b) =>
        this.compareBySlugOrder(subOrder, a.slug, b.slug)
      );
      for (const subcategory of subcategories) {
        for (const post of subcategory.posts) {
          allPosts.push({
            ...post,
            category: category.name,
            subcategory: subcategory.name,
            categorySlug: category.slug,
            subcategorySlug: subcategory.slug
          });
        }
      }
    }

    this.allPostsCache = allPosts;
    return of(this.allPostsCache);
  }

  getAllPosts(): Observable<PostMetadata[]> {
    return this.loadAllPostsMetadata();
  }

  getPostsByCategory(categorySlug: string): Observable<PostMetadata[]> {
    const categoryData = this.predefinedCategories[categorySlug];
    if (!categoryData) {
      return of([]);
    }

    const subOrder = this.studyOrderSubcategorySlugs[categorySlug] ?? [];
    const subcategories = [...categoryData.subcategories].sort((a, b) =>
      this.compareBySlugOrder(subOrder, a.slug, b.slug)
    );

    const posts: PostMetadata[] = [];
    for (const subcategory of subcategories) {
      for (const post of subcategory.posts) {
        posts.push({
          ...post,
          category: categoryData.name,
          subcategory: subcategory.name,
          categorySlug: categoryData.slug,
          subcategorySlug: subcategory.slug
        });
      }
    }

    return of(posts);
  }

  getPost(slug: string): Observable<Post | null> {
    // Find the post metadata from predefined categories
    let foundPost: PostMetadata | null = null;
    let foundCategory: PredefinedCategory | null = null;
    let foundSubcategory: PredefinedSubcategory | null = null;

    Object.values(this.predefinedCategories).forEach(category => {
      category.subcategories.forEach(subcategory => {
        const post = subcategory.posts.find(p => p.slug === slug);
        if (post) {
          foundPost = post;
          foundCategory = category;
          foundSubcategory = subcategory;
        }
      });
    });

    if (!foundPost || !foundCategory || !foundSubcategory) {
      return of(null);
    }

    // Build file path: /data/posts/{category-slug}/{subcategory-slug}/{post-slug}.md
    const categorySlug = (foundCategory as PredefinedCategory).slug;
    const subcategorySlug = (foundSubcategory as PredefinedSubcategory).slug;
    const filePath = `/data/posts/${categorySlug}/${subcategorySlug}/${slug}.md`;

    return this.http.get(filePath, { responseType: 'text' }).pipe(
      map(markdown => {
        const content = this.parseMarkdown(markdown);
        return { 
          ...foundPost!, 
          category: foundCategory!.name,
          subcategory: foundSubcategory!.name,
          categorySlug: (foundCategory as PredefinedCategory).slug,
          subcategorySlug: (foundSubcategory as PredefinedSubcategory).slug,
          content 
        } as Post;
      }),
      catchError(() => of(null))
    );
  }

  getCategories(): Observable<string[]> {
    const orderedSlugs = new Set(this.studyOrderCategorySlugs);
    const names: string[] = [];
    for (const slug of this.studyOrderCategorySlugs) {
      const cat = this.predefinedCategories[slug];
      if (cat) {
        names.push(cat.name);
      }
    }
    for (const slug of Object.keys(this.predefinedCategories)) {
      if (!orderedSlugs.has(slug)) {
        names.push(this.predefinedCategories[slug].name);
      }
    }
    return of(names);
  }

  getCategoryTree(): Observable<CategoryTree[]> {
    const result: CategoryTree[] = [];
    const orderedSlugs = new Set(this.studyOrderCategorySlugs);

    const pushCategoryTree = (categorySlug: string): void => {
      const category = this.predefinedCategories[categorySlug];
      if (!category) {
        return;
      }

      const subOrder = this.studyOrderSubcategorySlugs[categorySlug] ?? [];
      const sortedSubcategories = [...category.subcategories].sort((a, b) =>
        this.compareBySlugOrder(subOrder, a.slug, b.slug)
      );

      const categoryTree: CategoryTree = {
        name: category.name,
        slug: category.slug,
        count: 0,
        subcategories: []
      };

      for (const subcategory of sortedSubcategories) {
        const subCategoryData: SubCategory = {
          name: subcategory.name,
          slug: subcategory.slug,
          count: subcategory.posts.length,
          posts: subcategory.posts
        };

        categoryTree.subcategories.push(subCategoryData);
        categoryTree.count += subcategory.posts.length;
      }

      result.push(categoryTree);
    };

    for (const slug of this.studyOrderCategorySlugs) {
      pushCategoryTree(slug);
    }
    for (const slug of Object.keys(this.predefinedCategories)) {
      if (!orderedSlugs.has(slug)) {
        pushCategoryTree(slug);
      }
    }

    return of(result);
  }

  getPostsBySubcategory(categorySlug: string, subcategorySlug: string): Observable<PostMetadata[]> {
    const categoryData = this.predefinedCategories[categorySlug];
    if (!categoryData) {
      return of([]);
    }

    const subcategoryData = categoryData.subcategories.find(sub => sub.slug === subcategorySlug);
    if (!subcategoryData) {
      return of([]);
    }

    const posts: PostMetadata[] = subcategoryData.posts.map(post => ({
      ...post,
      category: categoryData.name,
      subcategory: subcategoryData.name,
      categorySlug: categoryData.slug,
      subcategorySlug: subcategoryData.slug
    }));

    return of(posts);
  }

  private parseMarkdown(markdown: string): string {
    // Add standard footer section to all posts
    const footerSection = `
<br/><br/>
<br/><br/>
## Let's Connect!

Found this helpful? Have questions or spotted something I missed? 

I'd love to hear from you on [LinkedIn](https://www.linkedin.com/in/dmitrygrinko/)! Your feedback helps me write better posts and keeps the conversation going.`;

    const contentWithFooter = markdown + footerSection;
    return marked(contentWithFooter) as string;
  }
} 