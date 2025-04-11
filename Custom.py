import csv
import math
import re
from collections import defaultdict
from nltk.stem import PorterStemmer

class HybridMarkovSpamFilter:
    def __init__(self, max_n=3, alpha=1, unk_threshold=2):
        self.max_n = max_n
        self.alpha = alpha
        self.unk_threshold = unk_threshold

        # Инициализация стеммера
        self.stemmer = PorterStemmer()

        # Структуры данных
        self.vocab = defaultdict(int)
        self.models = []
        for n in range(1, max_n + 1):
            self.models.append({
                'n': n,
                'spam_counts': defaultdict(int),
                'ham_counts': defaultdict(int),
                'spam_total': 0,
                'ham_total': 0
            })

        self.num_spam = 0
        self.num_ham = 0

    def tokenize(self, text):
        tokens = re.findall(r'\b[\w$€£%]+\b|\d+', text.lower())  # Выделяем числа отдельно
        return [self.stemmer.stem(t) if not t.isdigit() else '<NUM>' for t in tokens]

    # Остальные методы остаются без изменений
    def _replace_rare_words(self, tokens):
        return [t if self.vocab[t] >= self.unk_threshold else '<UNK>' for t in tokens]

    def generate_ngrams(self, tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)] if len(tokens) >= n else []

    def train(self, messages, labels):
        for text in messages:
            tokens = self.tokenize(text)
            for t in tokens:
                self.vocab[t] += 1

        for text, label in zip(messages, labels):
            tokens = self.tokenize(text)
            tokens = self._replace_rare_words(tokens)

            if label == "spam":
                self.num_spam += 1
            else:
                self.num_ham += 1

            for model in self.models:
                n = model['n']
                ngrams = self.generate_ngrams(tokens, n)

                if label == "spam":
                    model['spam_total'] += len(ngrams)
                    for gram in ngrams:
                        model['spam_counts'][gram] += 1
                else:
                    model['ham_total'] += len(ngrams)
                    for gram in ngrams:
                        model['ham_counts'][gram] += 1

    def _ngram_probability(self, ngram, model, label):
        vocab_size = len(self.vocab)
        n = model['n']
        count = model[f"{label}_counts"].get(ngram, 0)
        total = model[f"{label}_total"]
    
        # Адаптивное сглаживание
        alpha = self.alpha / (1 + math.log(1 + total))
        return (count + alpha) / (total + alpha * (vocab_size ** n))

    def predict(self, text, weights=None):
        tokens = self._replace_rare_words(self.tokenize(text))
        if weights is None:
            weights = {1: 0.2, 2: 0.3, 3: 0.5}

        log_spam = math.log(self.num_spam / (self.num_spam + self.num_ham))
        log_ham = math.log(self.num_ham / (self.num_spam + self.num_ham))

        for model in self.models:
            n = model['n']
            ngrams = self.generate_ngrams(tokens, n)
            if not ngrams:
                continue

            weight = weights.get(n, 0)
            spam_prob = sum(math.log(self._ngram_probability(g, model, "spam")) for g in ngrams) * weight
            ham_prob = sum(math.log(self._ngram_probability(g, model, "ham")) for g in ngrams) * weight

            log_spam += spam_prob
            log_ham += ham_prob

        return "spam" if log_spam > log_ham else "ham"

    def evaluate(self, test_messages, test_labels):
        """Оценка точности модели"""
        correct = 0
        for msg, lbl in zip(test_messages, test_labels):
            prediction = self.predict(msg)
            if prediction == lbl:
                correct += 1
        return correct / len(test_labels)
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
    
    filter = HybridMarkovSpamFilter()
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
