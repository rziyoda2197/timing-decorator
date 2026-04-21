import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Feedbacklar uchun ma'lumotlar
feedbacklar = {
    'Feedback': ['Mening sevimli dasturim!', 'Bu dastur juda yomon.', 'Men juda xursandman.', 'Dastur yomon ishlaydi.', 'Men juda g'azablanganman.'],
    'Sentiment': [1, 0, 1, 0, 0]
}

# Ma'lumotlarni DataFrame ga aylantirish
df = pd.DataFrame(feedbacklar)

# Sentiment ni kategoriya sifatida qabul qilish
df['Sentiment'] = pd.Categorical(df['Sentiment']).codes

# Sentiment ni kategoriya sifatida qabul qilish
df['Sentiment'] = pd.Categorical(df['Sentiment']).codes

# Vectorizatsiya qilish
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df['Feedback'])

# Test va train ma'lumotlarini ajratish
x_train, x_test, y_train, y_test = train_test_split(x, df['Sentiment'], test_size=0.2, random_state=42)

# Naive Bayes modeli yaratish
model = MultinomialNB()

# Modelni o'rganish
model.fit(x_train, y_train)

# Modelni tekshirish
y_pred = model.predict(x_test)

# Modelni baholash
print('To'g'rilik darajasi:', accuracy_score(y_test, y_pred))
print('Klassifikatsiya hisoboti:')
print(classification_report(y_test, y_pred))
print('Konfuziya matritsa:')
print(confusion_matrix(y_test, y_pred))
```

Kodda quyidagi amallar amalga oshiriladi:

1. Feedbacklar uchun ma'lumotlar yaratiladi.
2. Ma'lumotlar DataFrame ga aylantiriladi.
3. Sentiment ni kategoriya sifatida qabul qilish uchun kodi qo'llaniladi.
4. Vectorizatsiya qilish uchun TfidfVectorizer qo'llaniladi.
5. Test va train ma'lumotlarini ajratish uchun train_test_split qo'llaniladi.
6. Naive Bayes modeli yaratiladi.
7. Modelni o'rganish uchun fit qo'llaniladi.
8. Modelni tekshirish uchun predict qo'llaniladi.
9. Modelni baholash uchun accuracy_score, classification_report va confusion_matrix qo'llaniladi.
