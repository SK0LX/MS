import csv
import math
import re

class NaiveBayesSpamFilter:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.spam_counts = {}
        self.ham_counts = {}
        self.spam_total = 0
        self.ham_total = 0
        self.num_spam = 0
        self.num_ham = 0
        self.vocab = set()

    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())

    def train(self, messages, labels):
        for text, label in zip(messages, labels):
            words = self.tokenize(text)
            self.vocab.update(words)

            if label == "spam":
                self.num_spam += 1
                for word in words:
                    self.spam_counts[word] = self.spam_counts.get(word, 0) + 1
                    self.spam_total += 1
            else:
                self.num_ham += 1
                for word in words:
                    self.ham_counts[word] = self.ham_counts.get(word, 0) + 1
                    self.ham_total += 1

    def _calculate_probability(self, text, label):
        log_prob = 0.0
        words = self.tokenize(text)

        # Исправление: Сглаживание для априорных вероятностей классов
        total_messages = self.num_spam + self.num_ham
        num_classes = 2

        if label == "spam":
            prior = (self.num_spam + self.alpha) / (total_messages + self.alpha * num_classes)
        else:
            prior = (self.num_ham + self.alpha) / (total_messages + self.alpha * num_classes)

        log_prob += math.log(prior)

        for word in words:
            if label == "spam":
                count = self.spam_counts.get(word, 0)
                total = self.spam_total
            else:
                count = self.ham_counts.get(word, 0)
                total = self.ham_total

            prob = (count + self.alpha) / (total + self.alpha * len(self.vocab))
            log_prob += math.log(prob)

        return log_prob

    def predict(self, text):
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

    filter = NaiveBayesSpamFilter()
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

    test_message = "Congratulations, you won the phone! Follow the link"
    result = filter.predict(test_message)
    print(f"Сообщение '{test_message}' классифицировано как: {result}")
    test_message = "Congratulations, you won the Olympics, tell me your phone number."
    result = filter.predict(test_message)
    print(f"Сообщение '{test_message}' классифицировано как: {result}")
    test_message = "You have won a lot of money"
    result = filter.predict(test_message)
    print(f"Сообщение '{test_message}' классифицировано как: {result}")

    test_message = "Congrats N1ck! You've been selected for exclusive rewards. Confirm NOW: hxxps://susp1cious.lnk. Limited-time: 24h!"
    result = filter.predict(test_message)
    print(f"Сообщение '{test_message}' классифицировано как: {result}")

    test_message = "Nick, the quokka exhibit at the zoo was amazing! Let's discuss the funding proposal tomorrow. PS: Don’t forget about the 5PM deadline."
    result = filter.predict(test_message)
    print(f"Сообщение '{test_message}' классифицировано как: {result}")
