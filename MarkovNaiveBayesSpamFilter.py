import csv
import math
import re

class MarkovNaiveBayesSpamFilter:
    def __init__(self, alpha=1, n=2):
        """
        alpha: коэффициент сглаживания (Лаплас)
        n: порядок n-грамм; если n == 1, используется классический наивный байес,
           при n >= 2 происходит моделирование переходов по n-граммам.
        """
        self.alpha = alpha
        self.n = n
        self.num_spam = 0
        self.num_ham = 0
        self.vocab = set()  # множество уникальных токенов (слов)

        if self.n == 1:
            # Стандартный подход: отдельные слова
            self.spam_unigram = {}
            self.ham_unigram = {}
            self.spam_total = 0
            self.ham_total = 0
        else:
            # Для n-грамм (при n>=2):
            # Для первого n-грамма (начало текста)
            self.spam_first_ngram = {}
            self.ham_first_ngram = {}
            self.spam_total_first = 0
            self.ham_total_first = 0
            # Для переходов: разбиваем n-граммы на префикс и последнее слово
            self.spam_ngram = {}      # счётчики для n-грамм (все, кроме первого окна)
            self.ham_ngram = {}
            self.spam_prefix = {}     # счётчики для префиксов (первые n-1 слов n-граммы)
            self.ham_prefix = {}

    def tokenize(self, text):
        """Токенизация: приведение к нижнему регистру и выделение слов."""
        return re.findall(r'\b\w+\b', text.lower())

    def generate_ngrams(self, tokens):
        """
        Генерирует список n-грамм из списка токенов.
        Если длина текста меньше n, возвращается один n-грамм (tuple(tokens)).
        """
        if len(tokens) < self.n:
            return [tuple(tokens)]
        return [tuple(tokens[i:i+self.n]) for i in range(len(tokens) - self.n + 1)]

    def train(self, messages, labels):
        """
        Обучает фильтр, обновляя статистику по обучающим примерам.
        messages: список текстов сообщений.
        labels: список меток ('spam' или другой, считаем ham).
        """
        for text, label in zip(messages, labels):
            tokens = self.tokenize(text)
            if not tokens:
                continue
            # Обновляем общий словарь
            self.vocab.update(tokens)
            ngrams = self.generate_ngrams(tokens)
            if label == "spam":
                self.num_spam += 1
                if self.n == 1:
                    for gram in ngrams:
                        self.spam_unigram[gram] = self.spam_unigram.get(gram, 0) + 1
                        self.spam_total += 1
                else:
                    # Первый n-грамм
                    first = ngrams[0]
                    self.spam_first_ngram[first] = self.spam_first_ngram.get(first, 0) + 1
                    self.spam_total_first += 1
                    # Оставшиеся n-граммы: учитываем переходы по префиксу
                    for gram in ngrams[1:]:
                        prefix = gram[:-1]
                        self.spam_ngram[gram] = self.spam_ngram.get(gram, 0) + 1
                        self.spam_prefix[prefix] = self.spam_prefix.get(prefix, 0) + 1
            else:
                self.num_ham += 1
                if self.n == 1:
                    for gram in ngrams:
                        self.ham_unigram[gram] = self.ham_unigram.get(gram, 0) + 1
                        self.ham_total += 1
                else:
                    first = ngrams[0]
                    self.ham_first_ngram[first] = self.ham_first_ngram.get(first, 0) + 1
                    self.ham_total_first += 1
                    for gram in ngrams[1:]:
                        prefix = gram[:-1]
                        self.ham_ngram[gram] = self.ham_ngram.get(gram, 0) + 1
                        self.ham_prefix[prefix] = self.ham_prefix.get(prefix, 0) + 1


    def _calculate_probability(self, text, label):
        """
        Вычисляет логарифм апостериорной вероятности для данного текста с
        использованием либо униграмм (n == 1), либо n-грамм с переходами (n >= 2).
        """
        log_prob = 0.0
        tokens = self.tokenize(text)
        if not tokens:
            return float('-inf')

        total_messages = self.num_spam + self.num_ham
        num_classes = 2

        # Априорная вероятность класса с сглаживанием
        if label == "spam":
            prior = (self.num_spam + self.alpha) / (total_messages + self.alpha * num_classes)
        else:
            prior = (self.num_ham + self.alpha) / (total_messages + self.alpha * num_classes)
        log_prob += math.log(prior)

        ngrams = self.generate_ngrams(tokens)
        if self.n == 1:
            # Модель на основе отдельных слов (унграмм)
            for gram in ngrams:
                if label == "spam":
                    count = self.spam_unigram.get(gram, 0)
                    total = self.spam_total
                else:
                    count = self.ham_unigram.get(gram, 0)
                    total = self.ham_total
                vocab_size = len(self.vocab)
                prob = (count + self.alpha) / (total + self.alpha * vocab_size)
                log_prob += math.log(prob)
        else:
            # Модель для n-грамм: сначала вероятность первого n-грамма, потом переходы
            first = ngrams[0]
            if label == "spam":
                count = self.spam_first_ngram.get(first, 0)
                total = self.spam_total_first
                # Оценка размерности пространства n-грамм (приблизительно)
                vocab_ngram = (len(self.vocab)) ** self.n
            else:
                count = self.ham_first_ngram.get(first, 0)
                total = self.ham_total_first
                vocab_ngram = (len(self.vocab)) ** self.n
            first_prob = (count + self.alpha) / (total + self.alpha * vocab_ngram)
            log_prob += math.log(first_prob)

            # Переходы: для каждой позиции считаем вероятность появления последнего слова по данному префиксу
            for gram in ngrams[1:]:
                prefix = gram[:-1]
                if label == "spam":
                    count = self.spam_ngram.get(gram, 0)
                    total_prefix = self.spam_prefix.get(prefix, 0)
                else:
                    count = self.ham_ngram.get(gram, 0)
                    total_prefix = self.ham_prefix.get(prefix, 0)
                vocab_size = len(self.vocab)
                # Если префикс не встречался, используем лишь сглаживание
                if total_prefix == 0:
                    prob = self.alpha / (self.alpha * vocab_size)
                else:
                    prob = (count + self.alpha) / (total_prefix + self.alpha * vocab_size)
                log_prob += math.log(prob)
        return log_prob

    def predict(self, text):
        """
        Предсказывает метку ('spam' или 'ham') для заданного текста,
        сравнивая логарифмы апостериорных вероятностей.
        """
        if self.num_spam + self.num_ham == 0:
            return "ham"
        spam_score = self._calculate_probability(text, "spam")
        ham_score = self._calculate_probability(text, "ham")
        return "spam" if spam_score > ham_score else "ham"

if __name__ == "__main__":
    # Пример использования
    messages = []
    labels = []
    with open('spam.csv', newline='', encoding='windows-1251') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) >= 2:
                labels.append(row[0])
                messages.append(row[1])

    filter = MarkovNaiveBayesSpamFilter(n=2)
    filter.train(messages, labels)

    test_msg = "Get cheap drugs"
    print(f"Prediction: {filter.predict(test_msg)}")

    # Расширенное использование с файлами
    try:
        with open('spam.txt') as f:
            spam_text = f.read()
        print(f"Spam file prediction: {filter.predict(spam_text)}")

        with open('hum.txt') as f:
            ham_text = f.read()
        print(f"Ham file prediction: {filter.predict(ham_text)}")
    except FileNotFoundError:
        print("Example files not found")

    test_message = "Hi NICK, go play slizerIO!"
    result = filter.predict(test_message)
    print(f"Сообщение '{test_message}' классифицировано как: {result}")
    test_message = "You win iphone!!!!"
    result = filter.predict(test_message)
    print(f"Сообщение '{test_message}' классифицировано как: {result}")
    test_message = "You have won a lot of money"
    result = filter.predict(test_message)
    print(f"Сообщение '{test_message}' классифицировано как: {result}")
    test_message = "Congratulations, you won the phone! Follow the link"
    result = filter.predict(test_message)
    print(f"Сообщение '{test_message}' классифицировано как: {result}")
    test_message = "Congratulations, you won the Olympics, tell me your phone number."
    result = filter.predict(test_message)
    print(f"Сообщение '{test_message}' классифицировано как: {result}")

    test_message = "Congrats N1ck! You've been selected for exclusive rewards. Confirm NOW: hxxps://susp1cious.lnk. Limited-time: 24h!"
    result = filter.predict(test_message)
    print(f"Сообщение '{test_message}' классифицировано как: {result}")

    test_message = "Nick, the quokka exhibit at the zoo was amazing! Let's discuss the funding proposal tomorrow. PS: Don’t forget about the 5PM deadline."
    result = filter.predict(test_message)
    print(f"Сообщение '{test_message}' классифицировано как: {result}")
